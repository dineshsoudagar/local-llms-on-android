package com.example.local_llm

import android.content.Context
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
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile

class GemmaLiteRtBackend(
    private val context: Context,
    private val spec: GemmaLiteRtSpec,
    private val modelFileResolver: ModelFileResolver,
    private val initializationPolicy: BackendInitializationPolicy = BackendInitializationPolicy()
) : ChatBackend {

    companion object {
        private const val TAG = "GemmaLiteRtBackend"
        private const val THOUGHT_CHANNEL_NAME = "thought"
        private const val DEFAULT_MAX_NUM_TOKENS = 2048
        private const val DEFAULT_MAX_NUM_IMAGES = 1
        private const val CPU_THREAD_COUNT = 4
        private const val AUDIO_COMPATIBILITY_PREFS = "gemma_audio_compatibility"
        private const val AUDIO_SMOKE_TEST_VERSION = "litertlm-0.14.0-gemma4-audio-v1"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null
    private var directImageInputInitialized = false
    private var directAudioInputInitialized = false

    override val capabilities: BackendCapabilities
        get() = BackendCapabilities(
            supportsNativeImage = directImageInputInitialized,
            supportsNativeAudio = directAudioInputInitialized,
            contextWindowTokens = DEFAULT_MAX_NUM_TOKENS
        )

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)
        val modelPath = modelFile.absolutePath

        directImageInputInitialized = false
        directAudioInputInitialized = false
        val failures = mutableListOf<EngineInitFailure>()
        for (attempt in buildEngineInitAttempts()) {
            Log.i(
                TAG,
                "Initializing ${spec.displayName} from $modelPath (${modelFile.length()} bytes) " +
                    "with ${attempt.label}, maxNumTokens=$DEFAULT_MAX_NUM_TOKENS."
            )
            val result = createInitializedEngine(modelPath, attempt)
            val initializedEngine = result.getOrNull()
            if (initializedEngine != null) {
                engine = initializedEngine
                directImageInputInitialized = attempt.visionBackend != null
                val audioVerified = attempt.audioBackend == null || verifyNativeAudioCompatibility(modelFile)
                if (!audioVerified) {
                    failures += EngineInitFailure(
                        attempt.label,
                        IllegalStateException("Native audio compatibility smoke test failed.")
                    )
                    runCatching { engine.close() }
                    continue
                }
                directAudioInputInitialized = attempt.audioBackend != null
                if (!directImageInputInitialized && spec.directImageInputAvailable) {
                    Log.w(
                        TAG,
                        "Gemma initialized in text-only mode. Direct image input is disabled on this device/backend."
                    )
                }
                if (!directAudioInputInitialized && spec.directAudioInputAvailable) {
                    Log.w(
                        TAG,
                        "Gemma initialized without an audio backend. Native audio is disabled; Whisper fallback is forbidden."
                    )
                }
                return@withContext
            }

            val error = result.exceptionOrNull()
                ?: IllegalStateException("Unknown LiteRT-LM initialization failure.")
            failures += EngineInitFailure(attempt.label, error)
            Log.w(
                TAG,
                "Gemma LiteRT-LM initialization failed for ${attempt.label}: ${error.shortDescription()}",
                error
            )
        }

        directImageInputInitialized = false
        directAudioInputInitialized = false
        throw IllegalStateException(
            "Failed to initialize Gemma LiteRT-LM. ${formatInitFailures(failures)}",
            failures.lastOrNull()?.error
        )
    }

    override suspend fun resetConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String
    ) {
        recreateConversation(history, thinkingEnabled, modelInstruction)
    }

    override suspend fun streamReply(
        request: InferenceRequest,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(request.imageFilePaths.isEmpty() || supportsDirectImageInput) {
            "Direct Gemma image input is not available on this device/backend. Switch image input to OCR."
        }
        require(request.nativeAudioInputs.isEmpty() || supportsNativeAudioInput) {
            "Native Gemma audio is not initialized on this device/backend."
        }
        require(request.imageFilePaths.isEmpty() || request.nativeAudioInputs.isEmpty()) {
            "Image and document/audio attachments cannot be mixed in the same send."
        }
        require(request.history.isNotEmpty() && request.history.last().role == ChatRole.USER) {
            "Gemma backend expects the final history turn to be the user's prompt."
        }

        val initialHistory = request.history.dropLast(1)
        val userTurn = request.history.last()
        recreateConversation(initialHistory, request.thinkingEnabled, request.modelInstruction)

        val activeConversation = conversation
            ?: throw IllegalStateException("Conversation was not created.")

        val textBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()

        val messageForModel = buildUserMessage(
            userTurn.text,
            request.imageFilePaths,
            request.nativeAudioInputs
        )
        activeConversation.sendMessageAsync(messageForModel).collect { message ->
            val chunkText = extractTextContent(message)
            if (chunkText.isNotEmpty()) {
                textBuilder.append(chunkText)
            }

            val thoughtChunk = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
            if (thoughtChunk.isNotEmpty()) {
                thinkingBuilder.append(thoughtChunk)
            }

            onPartial(
                BackendResponse(
                    text = textBuilder.toString(),
                    thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() }
                )
            )
        }

        BackendResponse(
            text = textBuilder.toString(),
            thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() }
        )
    }

    private fun createInitializedEngine(
        modelPath: String,
        attempt: EngineInitAttempt
    ): Result<Engine> {
        var candidate: Engine? = null
        return runCatching {
            candidate = Engine(
                EngineConfig(
                    modelPath = modelPath,
                    backend = attempt.backend,
                    visionBackend = attempt.visionBackend,
                    audioBackend = attempt.audioBackend,
                    maxNumTokens = DEFAULT_MAX_NUM_TOKENS,
                    maxNumImages = if (attempt.visionBackend != null) DEFAULT_MAX_NUM_IMAGES else null,
                    cacheDir = context.cacheDir.absolutePath
                )
            )
            candidate!!.initialize()
            candidate!!
        }.onFailure {
            candidate?.let { failedEngine ->
                runCatching {
                    failedEngine.close()
                }
            }
        }
    }

    private suspend fun verifyNativeAudioCompatibility(modelFile: File): Boolean {
        if (!spec.directAudioInputAvailable) return false
        val preferences = context.getSharedPreferences(AUDIO_COMPATIBILITY_PREFS, Context.MODE_PRIVATE)
        val gateKey = "$AUDIO_SMOKE_TEST_VERSION:${spec.id}:${modelFile.length()}"
        if (preferences.getBoolean(gateKey, false)) return true

        val smokeFile = File(context.cacheDir, "gemma_native_audio_smoke.wav")
        return try {
            writeSilentSmokeWav(smokeFile)
            val smokeConversation = engine.createConversation(
                ConversationConfig(
                    systemInstruction = Contents.of("Audio compatibility test."),
                    channels = emptyList()
                )
            )
            try {
                smokeConversation.sendMessageAsync(
                    Message.user(
                        Contents.of(
                            Content.Text("Acknowledge this silent audio test briefly."),
                            Content.AudioFile(smokeFile.absolutePath)
                        )
                    )
                ).collect { }
            } finally {
                runCatching { smokeConversation.close() }
            }
            preferences.edit().putBoolean(gateKey, true).apply()
            Log.i(TAG, "Native Gemma audio compatibility smoke test passed for ${spec.displayName}.")
            true
        } catch (error: Throwable) {
            Log.e(
                TAG,
                "Native Gemma audio compatibility smoke test failed. Audio remains disabled and will not use Whisper.",
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
        modelInstruction: String
    ) {
        closeConversation()
        conversation = engine.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(buildSystemInstruction(thinkingEnabled, modelInstruction)),
                initialMessages = history.map { turn ->
                    when (turn.role) {
                        ChatRole.USER -> Message.user(turn.text)
                        ChatRole.ASSISTANT -> Message.model(turn.text)
                    }
                },
                channels = if (thinkingEnabled) null else emptyList()
            )
        )
    }

    private fun buildSystemInstruction(thinkingEnabled: Boolean, modelInstruction: String): String {
        return if (thinkingEnabled) {
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
            add(Content.Text(textContent))
            nativeAudioInputs.forEach { input -> add(Content.AudioFile(input.filePath)) }
        }
        return Message.user(Contents.of(*contents.toTypedArray()))
    }

    private fun extractTextContent(message: Message): String {
        val text = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString(separator = "") { content -> content.text }
        return text.ifBlank { message.contents.toString() }
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }

    private fun buildEngineInitAttempts(): List<EngineInitAttempt> {
        val attempts = mutableListOf<EngineInitAttempt>()
        val cpuBackend = Backend.CPU(numOfThreads = CPU_THREAD_COUNT)
        if (spec.directImageInputAvailable || spec.directAudioInputAvailable) {
            attempts += EngineInitAttempt(
                "GPU text + GPU multimodal",
                Backend.GPU(),
                Backend.GPU().takeIf { spec.directImageInputAvailable },
                Backend.GPU().takeIf { spec.directAudioInputAvailable }
            )
            if (initializationPolicy.allowCpuFallback) {
                attempts += EngineInitAttempt(
                    "GPU text + CPU multimodal",
                    Backend.GPU(),
                    cpuBackend.takeIf { spec.directImageInputAvailable },
                    cpuBackend.takeIf { spec.directAudioInputAvailable }
                )
                attempts += EngineInitAttempt(
                    "CPU text + CPU multimodal",
                    cpuBackend,
                    cpuBackend.takeIf { spec.directImageInputAvailable },
                    cpuBackend.takeIf { spec.directAudioInputAvailable }
                )
            }
        }
        if (spec.directImageInputAvailable) {
            attempts += EngineInitAttempt("GPU text + GPU vision", Backend.GPU(), Backend.GPU(), null)
        }
        if (spec.directAudioInputAvailable) {
            attempts += EngineInitAttempt("GPU text + GPU audio", Backend.GPU(), null, Backend.GPU())
        }
        attempts += EngineInitAttempt("GPU text only", Backend.GPU(), null, null)
        if (initializationPolicy.allowCpuFallback) {
            attempts += EngineInitAttempt("CPU text only", cpuBackend, null, null)
        }
        return attempts
    }

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
        val audioBackend: Backend?
    )

    private data class EngineInitFailure(
        val label: String,
        val error: Throwable
    )
}
