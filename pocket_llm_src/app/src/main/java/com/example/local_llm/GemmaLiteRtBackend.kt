package com.example.local_llm

import android.content.Context
import android.os.Build
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile

/** Shared LiteRT multimodal engine; custom models use no Gemma-specific thinking or tool directives. */
class GemmaLiteRtBackend(
    private val context: Context,
    private val spec: ModelDescriptor,
    private val modelFileResolver: ModelFileResolver,
    private val runtimeSettings: ModelRuntimeSettings = ModelRuntimeSettings(
        ModelRuntimeSettingsLimits.LITERT_DEFAULT_CONTEXT_LENGTH
    ),
    private val initializationPolicy: BackendInitializationPolicy = BackendInitializationPolicy()
) : ChatBackend {

    companion object {
        private const val TAG = "GemmaLiteRtBackend"
        private const val THOUGHT_CHANNEL_NAME = "thought"
        private const val DEFAULT_MAX_NUM_IMAGES = 1
        private const val CPU_THREAD_COUNT = 4
        private const val AUDIO_COMPATIBILITY_PREFS = "gemma_audio_compatibility"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null
    private var directImageInputInitialized = false
    private var directAudioInputInitialized = false
    private var importedInputs: CustomModelCapabilities? = (spec as? CustomLiteRtSpec)?.detectedInputs
    private val imageInputRequested: Boolean
        get() = if (spec is CustomLiteRtSpec) importedInputs?.vision == true else spec.directImageInputAvailable
    private val audioInputRequested: Boolean
        get() = if (spec is CustomLiteRtSpec) importedInputs?.audio == true else spec.directAudioInputAvailable

    override val capabilities: BackendCapabilities
        get() = BackendCapabilities(
            supportsNativeImage = directImageInputInitialized,
            supportsNativeAudio = directAudioInputInitialized,
            supportsNativeToolCalling = spec is GemmaLiteRtSpec,
            contextWindowTokens = runtimeSettings.contextLengthTokens
        )

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)
        val modelPath = modelFile.absolutePath
        if (spec is CustomLiteRtSpec) {
            // Reinspect on load, including old imports with no saved input metadata.
            importedInputs = inspectCustomModel(modelFile)
            require(importedInputs?.text != false) { "This model does not declare text input support." }
            if (importedInputs == null) Log.w(TAG, "Could not inspect ${spec.displayName}; attempting text only.")
        }

        directImageInputInitialized = false
        directAudioInputInitialized = false
        val audioModelSha256 = if (audioInputRequested) {
            GemmaNativeAudioCompatibility.modelSha256(modelFile)
        } else {
            null
        }
        val failures = mutableListOf<EngineInitFailure>()
        for (attempt in buildEngineInitAttempts()) {
            Log.i(
                TAG,
                "Initializing ${spec.displayName} from $modelPath (${modelFile.length()} bytes) " +
                "with ${attempt.label}, maxNumTokens=${runtimeSettings.contextLengthTokens}."
            )
            val result = createInitializedEngine(modelPath, attempt)
            val initializedEngine = result.getOrNull()
            if (initializedEngine != null) {
                engine = initializedEngine
                directImageInputInitialized = attempt.visionBackend != null
                val audioVerified = attempt.audioBackend == null || verifyNativeAudioCompatibility(
                    attempt = attempt,
                    modelSha256 = checkNotNull(audioModelSha256)
                )
                if (!audioVerified) {
                    failures += EngineInitFailure(
                        attempt.label,
                        IllegalStateException("Native audio compatibility smoke test failed.")
                    )
                    runCatching { engine.close() }
                    continue
                }
                directAudioInputInitialized = attempt.audioBackend != null
                if (!directImageInputInitialized && imageInputRequested) {
                    Log.w(
                        TAG,
                        "${spec.displayName} initialized without vision. Direct image input is disabled on this device/backend."
                    )
                }
                if (!directAudioInputInitialized && audioInputRequested) {
                    Log.w(
                        TAG,
                        "${spec.displayName} initialized without an audio backend. Native audio is disabled."
                    )
                }
                return@withContext
            }

            val error = result.exceptionOrNull()
                ?: IllegalStateException("Unknown LiteRT-LM initialization failure.")
            failures += EngineInitFailure(attempt.label, error)
            Log.w(
                TAG,
                "LiteRT-LM initialization failed for ${attempt.label}: ${error.shortDescription()}",
                error
            )
        }

        directImageInputInitialized = false
        directAudioInputInitialized = false
        throw IllegalStateException(
            "Failed to initialize ${spec.displayName}. ${formatInitFailures(failures)}",
            failures.lastOrNull()?.error
        )
    }

    override suspend fun resetConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String
    ) {
        val boundedHistory = fitHistoryWithinContext(
            InferenceRequest(
                history = history,
                thinkingEnabled = thinkingEnabled,
                modelInstruction = modelInstruction
            )
        )
        recreateConversation(boundedHistory, thinkingEnabled, modelInstruction)
    }

    override suspend fun streamReply(
        request: InferenceRequest,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(request.imageFilePaths.isEmpty() || supportsDirectImageInput) {
            "Direct image input is not available on this device/backend. Switch image input to OCR."
        }
        require(request.nativeAudioInputs.isEmpty() || supportsNativeAudioInput) {
            "Native audio is not initialized on this device/backend."
        }
        require(request.imageFilePaths.isEmpty() || request.nativeAudioInputs.isEmpty()) {
            "Image and document/audio attachments cannot be mixed in the same send."
        }
        require(request.imageFilePaths.size <= DEFAULT_MAX_NUM_IMAGES) { "Send one image at a time with native image input." }
        val boundedHistory = fitHistoryWithinContext(request)
        require(
            boundedHistory.isNotEmpty() &&
                (boundedHistory.last().role == ChatRole.USER || boundedHistory.last().isToolResult)
        ) {
            "LiteRT backend expects the final history turn to be a user or tool message."
        }

        val initialHistory = boundedHistory.dropLast(1)
        val userTurn = boundedHistory.last()
        recreateConversation(
            initialHistory,
            request.thinkingEnabled,
            request.modelInstruction,
            request.tools
        )

        val activeConversation = conversation
            ?: throw IllegalStateException("Conversation was not created.")

        val textBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()

        val messageForModel = buildUserMessage(
            userTurn.text,
            request.imageFilePaths,
            request.nativeAudioInputs
        )
        var externalToolCalls = emptyList<ExternalToolCall>()
        activeConversation.sendMessageAsync(
            if (userTurn.isToolResult) userTurn.toLiteRtMessage() else messageForModel
        ).collect { message ->
            val chunkText = extractTextContent(message)
            if (chunkText.isNotEmpty()) {
                textBuilder.append(chunkText)
            }

            val thoughtChunk = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
            if (thoughtChunk.isNotEmpty()) {
                thinkingBuilder.append(thoughtChunk)
            }

            if (message.toolCalls.isNotEmpty()) {
                externalToolCalls = nativeToolCallsToExternal(
                    message.toolCalls,
                    externalToolCalls,
                    request.parallelToolCalls
                )
            }
            onPartial(
                BackendResponse(
                    text = textBuilder.toString(),
                    thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() },
                    toolCalls = externalToolCalls
                )
            )
        }

        BackendResponse(
            text = textBuilder.toString(),
            thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() },
            toolCalls = externalToolCalls
        )
    }

    private fun createInitializedEngine(
        modelPath: String,
        attempt: EngineInitAttempt
    ): Result<Engine> {
        return createUsableLiteRtRuntime(
            create = { Engine(
                EngineConfig(
                    modelPath = modelPath,
                    backend = attempt.backend,
                    visionBackend = attempt.visionBackend,
                    audioBackend = attempt.audioBackend,
                    maxNumTokens = runtimeSettings.contextLengthTokens,
                    maxNumImages = if (attempt.visionBackend != null) DEFAULT_MAX_NUM_IMAGES else null,
                    cacheDir = context.cacheDir.absolutePath
                )
            ) },
            initializeAndValidate = { candidate ->
                candidate.initialize()
                // Vision executors can compile lazily during conversation creation. Keep this
                // inside the attempt so unsupported GPU ops reach the CPU vision fallback.
                candidate.createConversation(
                    ConversationConfig(channels = emptyList(), automaticToolCalling = false)
                ).use { }
            }
        )
    }

    private suspend fun verifyNativeAudioCompatibility(
        attempt: EngineInitAttempt,
        modelSha256: String
    ): Boolean {
        if (!audioInputRequested) return false
        val preferences = context.getSharedPreferences(AUDIO_COMPATIBILITY_PREFS, Context.MODE_PRIVATE)
        val cache = GemmaNativeAudioCompatibilityCache(
            SharedPreferencesGemmaNativeAudioCompatibilityStore(preferences)
        )
        val runtimeIdentity = GemmaNativeAudioRuntimeIdentity(
            modelSha256 = modelSha256,
            backendAttempt = attempt.runtimeIdentity,
            liteRtLmVersion = GemmaNativeAudioCompatibility.LITERT_LM_RUNTIME_VERSION,
            smokeTestContractVersion = if (spec is CustomLiteRtSpec) "custom-native-audio-v1" else GemmaNativeAudioCompatibility.SMOKE_TEST_CONTRACT_VERSION,
            buildFingerprint = Build.FINGERPRINT
        )
        if (cache.isAuthorized(runtimeIdentity)) return true

        val smokeFile = File(context.cacheDir, "gemma_native_audio_smoke.wav")
        return try {
            writeSilentSmokeWav(smokeFile)
            val smokeConversation = engine.createConversation(
                ConversationConfig(
                    systemInstruction = Contents.of("Audio compatibility test."),
                    channels = emptyList(),
                    maxOutputToken = 16
                )
            )
            try {
                smokeConversation.sendMessageAsync(
                    Message.user(
                        Contents.of(
                            Content.AudioFile(smokeFile.absolutePath),
                            Content.Text(
                                "Acknowledge this silent audio test briefly."
                            )
                        )
                    )
                ).collect { }
            } finally {
                runCatching { smokeConversation.close() }
            }
            cache.recordSuccess(runtimeIdentity)
            Log.i(
                TAG,
                "Native audio compatibility smoke test passed for ${spec.displayName} " +
                    "with ${attempt.runtimeIdentity}."
            )
            true
        } catch (error: CancellationException) {
            throw error
        } catch (error: Throwable) {
            cache.recordFailure(runtimeIdentity)
            Log.e(
                TAG,
                "Native audio compatibility smoke test failed for ${attempt.runtimeIdentity}. " +
                    "Audio remains disabled.",
                error
            )
            false
        } finally {
            smokeFile.delete()
        }
    }

    private fun writeSilentSmokeWav(file: File) {
        val sampleCount = 1_600
        val dataBytes = sampleCount * 2
        RandomAccessFile(file, "rw").use { wav ->
            wav.setLength(0L)
            wav.writeBytes("RIFF")
            wav.writeInt(Integer.reverseBytes(dataBytes + 36))
            wav.writeBytes("WAVEfmt ")
            wav.writeInt(Integer.reverseBytes(16))
            wav.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
            wav.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
            wav.writeInt(Integer.reverseBytes(16_000))
            wav.writeInt(Integer.reverseBytes(32_000))
            wav.writeShort(java.lang.Short.reverseBytes(2.toShort()).toInt())
            wav.writeShort(java.lang.Short.reverseBytes(16.toShort()).toInt())
            wav.writeBytes("data")
            wav.writeInt(Integer.reverseBytes(dataBytes))
            wav.write(ByteArray(dataBytes))
        }
    }

    override fun cancelGeneration() {
        conversation?.cancelProcess()
    }

    override fun close() {
        closeConversation()
        if (::engine.isInitialized && engine.isInitialized()) {
            engine.close()
        }
        directImageInputInitialized = false
        directAudioInputInitialized = false
    }

    private fun recreateConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String,
        tools: List<ExternalToolDefinition> = emptyList()
    ) {
        closeConversation()
        conversation = engine.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(buildSystemInstruction(thinkingEnabled, modelInstruction)),
                initialMessages = history.map { turn ->
                    turn.toLiteRtMessage()
                },
                tools = externalToolProviders(if (capabilities.supportsNativeToolCalling) tools else emptyList()),
                automaticToolCalling = false,
                channels = if (spec is GemmaLiteRtSpec && thinkingEnabled) null else emptyList()
            )
        )
    }

    private fun buildSystemInstruction(thinkingEnabled: Boolean, modelInstruction: String): String {
        return if (spec is GemmaLiteRtSpec && thinkingEnabled) {
            "<|think|>\n$modelInstruction"
        } else {
            modelInstruction
        }
    }

    private fun buildUserMessage(
        text: String,
        imageFilePaths: List<String>,
        nativeAudioInputs: List<NativeAudioInput>
    ): Message {
        if (imageFilePaths.isEmpty() && nativeAudioInputs.isEmpty()) {
            return Message.user(text)
        }

        val textContent = text.ifBlank {
            if (nativeAudioInputs.isNotEmpty()) {
                "Describe and transcribe the attached audio."
            } else {
                "Describe the attached image."
            }
        }
        val contents = buildList<Content> {
            imageFilePaths.forEach { path -> add(Content.ImageFile(path)) }
            nativeAudioInputs.forEach { input -> add(Content.AudioFile(input.filePath)) }
            add(Content.Text(textContent))
        }
        return Message.user(Contents.of(*contents.toTypedArray()))
    }

    private fun extractTextContent(message: Message): String {
        val text = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString(separator = "") { content -> content.text }
        return if (message.toolCalls.isNotEmpty()) text else text.ifBlank { message.contents.toString() }
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }

    private fun buildEngineInitAttempts(): List<EngineInitAttempt> {
        val attempts = mutableListOf<EngineInitAttempt>()
        val cpuBackend = Backend.CPU(numOfThreads = CPU_THREAD_COUNT)
        if (imageInputRequested || audioInputRequested) {
            attempts += EngineInitAttempt(
                "GPU text + GPU multimodal",
                Backend.GPU(),
                Backend.GPU().takeIf { imageInputRequested },
                Backend.GPU().takeIf { audioInputRequested },
                runtimeIdentity(
                    text = "gpu",
                    vision = "gpu".takeIf { imageInputRequested },
                    audio = "gpu".takeIf { audioInputRequested }
                )
            )
            if (audioInputRequested) {
                // Gemma 4 E2B declares a CPU-only audio encoder. Keep text and vision on the
                // GPU, which has already initialized successfully on the device, and move only
                // the constrained audio encoder to CPU. This is required even when broader CPU
                // fallbacks are disabled for a memory-risky model load.
                attempts += EngineInitAttempt(
                    "GPU text + GPU vision + CPU audio",
                    Backend.GPU(),
                    Backend.GPU().takeIf { imageInputRequested },
                    cpuBackend,
                    runtimeIdentity(
                        text = "gpu",
                        vision = "gpu".takeIf { imageInputRequested },
                        audio = "cpu-$CPU_THREAD_COUNT"
                    )
                )
            }
            if (initializationPolicy.allowCpuFallback) {
                attempts += EngineInitAttempt(
                    "GPU text + CPU multimodal",
                    Backend.GPU(),
                    cpuBackend.takeIf { imageInputRequested },
                    cpuBackend.takeIf { audioInputRequested },
                    runtimeIdentity(
                        text = "gpu",
                        vision = "cpu-$CPU_THREAD_COUNT".takeIf { imageInputRequested },
                        audio = "cpu-$CPU_THREAD_COUNT".takeIf { audioInputRequested }
                    )
                )
                attempts += EngineInitAttempt(
                    "CPU text + CPU multimodal",
                    cpuBackend,
                    cpuBackend.takeIf { imageInputRequested },
                    cpuBackend.takeIf { audioInputRequested },
                    runtimeIdentity(
                        text = "cpu-$CPU_THREAD_COUNT",
                        vision = "cpu-$CPU_THREAD_COUNT".takeIf { imageInputRequested },
                        audio = "cpu-$CPU_THREAD_COUNT".takeIf { audioInputRequested }
                    )
                )
            }
        }
        if (imageInputRequested) {
            attempts += EngineInitAttempt(
                "GPU text + GPU vision",
                Backend.GPU(),
                Backend.GPU(),
                null,
                runtimeIdentity("gpu", "gpu", null)
            )
        }
        if (audioInputRequested) {
            // Audio may be CPU-only even when vision cannot initialize on this device.
            attempts += EngineInitAttempt(
                "GPU text + CPU audio",
                Backend.GPU(),
                null,
                cpuBackend,
                runtimeIdentity("gpu", null, "cpu-$CPU_THREAD_COUNT")
            )
            attempts += EngineInitAttempt(
                "GPU text + GPU audio",
                Backend.GPU(),
                null,
                Backend.GPU(),
                runtimeIdentity("gpu", null, "gpu")
            )
        }
        attempts += EngineInitAttempt(
            "GPU text only",
            Backend.GPU(),
            null,
            null,
            runtimeIdentity("gpu", null, null)
        )
        if (initializationPolicy.allowCpuFallback) {
            attempts += EngineInitAttempt(
                "CPU text only",
                cpuBackend,
                null,
                null,
                runtimeIdentity("cpu-$CPU_THREAD_COUNT", null, null)
            )
        }
        return attempts
    }

    private fun runtimeIdentity(text: String, vision: String?, audio: String?): String =
        "text=$text;vision=${vision ?: "none"};audio=${audio ?: "none"}"

    private fun formatInitFailures(failures: List<EngineInitFailure>): String {
        if (failures.isEmpty()) {
            return "No backend attempts were made."
        }
        return failures.joinToString(separator = " ") { failure ->
            "${failure.label}: ${failure.error.shortDescription()}."
        }
    }

    private fun Throwable.shortDescription(): String {
        val message = message?.takeIf { it.isNotBlank() } ?: "no message"
        return "${javaClass.simpleName}($message)"
    }

    private data class EngineInitAttempt(
        val label: String,
        val backend: Backend,
        val visionBackend: Backend?,
        val audioBackend: Backend?,
        val runtimeIdentity: String
    )

    private data class EngineInitFailure(
        val label: String,
        val error: Throwable
    )
}
