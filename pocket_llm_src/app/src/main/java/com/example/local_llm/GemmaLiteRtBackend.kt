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
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null
    private var directImageInputInitialized = false

    override val supportsDirectImageInput: Boolean
        get() = directImageInputInitialized

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)
        val modelPath = modelFile.absolutePath

        directImageInputInitialized = false
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
                if (!directImageInputInitialized && spec.directImageInputAvailable) {
                    Log.w(
                        TAG,
                        "Gemma initialized in text-only mode. Direct image input is disabled on this device/backend."
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
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String,
        imageFilePaths: List<String>,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(imageFilePaths.isEmpty() || supportsDirectImageInput) {
            "Direct Gemma image input is not available on this device/backend. Switch image input to OCR."
        }
        require(history.isNotEmpty() && history.last().role == ChatRole.USER) {
            "Gemma backend expects the final history turn to be the user's prompt."
        }

        val initialHistory = history.dropLast(1)
        val userTurn = history.last()
        recreateConversation(initialHistory, thinkingEnabled, modelInstruction)

        val activeConversation = conversation
            ?: throw IllegalStateException("Conversation was not created.")

        val textBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()

        val messageForModel = buildUserMessage(userTurn.text, imageFilePaths)
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

    override fun cancelGeneration() {
        conversation?.cancelProcess()
    }

    override fun close() {
        closeConversation()
        if (::engine.isInitialized && engine.isInitialized()) {
            engine.close()
        }
        directImageInputInitialized = false
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

    private fun buildUserMessage(text: String, imageFilePaths: List<String>): Message {
        if (imageFilePaths.isEmpty()) {
            return Message.user(text)
        }

        val textContent = text.ifBlank { "Describe the attached image." }
        val contents = buildList<Content> {
            imageFilePaths.forEach { path -> add(Content.ImageFile(path)) }
            add(Content.Text(textContent))
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
        if (spec.directImageInputAvailable) {
            attempts += EngineInitAttempt("GPU text + GPU vision", Backend.GPU(), Backend.GPU())
            if (initializationPolicy.allowCpuFallback) {
                attempts += EngineInitAttempt("GPU text + CPU vision", Backend.GPU(), cpuBackend)
                attempts += EngineInitAttempt("CPU text + CPU vision", cpuBackend, cpuBackend)
            }
        }
        attempts += EngineInitAttempt("GPU text only", Backend.GPU(), null)
        if (initializationPolicy.allowCpuFallback) {
            attempts += EngineInitAttempt("CPU text only", cpuBackend, null)
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
        val visionBackend: Backend?
    )

    private data class EngineInitFailure(
        val label: String,
        val error: Throwable
    )
}
