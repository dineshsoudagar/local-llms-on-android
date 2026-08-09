package com.example.local_llm

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext

class QwenLiteRtBackend(
    private val context: Context,
    private val spec: QwenLiteRtSpec,
    private val modelFileResolver: ModelFileResolver,
    private val initializationPolicy: BackendInitializationPolicy = BackendInitializationPolicy()
) : ChatBackend {

    override val capabilities = BackendCapabilities(contextWindowTokens = 2048)

    companion object {
        private const val THOUGHT_CHANNEL_NAME = "thought"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)

        val gpuResult = createInitializedEngine(modelFile.absolutePath, Backend.GPU())

        engine = gpuResult.getOrElse { gpuError ->
            if (!initializationPolicy.allowCpuFallback) {
                throw IllegalStateException(
                    "Failed to initialize LiteRT-LM on GPU. CPU fallback was skipped because this load is already a memory risk: ${gpuError.message}",
                    gpuError
                )
            }
            createInitializedEngine(modelFile.absolutePath, Backend.CPU()).getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize LiteRT-LM GPU (${gpuError.message}) and CPU (${cpuError.message}).",
                    cpuError
                )
            }
        }
    }

    private fun createInitializedEngine(modelPath: String, backend: Backend): Result<Engine> {
        var candidate: Engine? = null
        return runCatching {
            candidate = Engine(
                EngineConfig(
                    modelPath = modelPath,
                    backend = backend,
                    cacheDir = context.cacheDir.absolutePath
                )
            )
            candidate!!.initialize()
            candidate!!
        }.onFailure {
            candidate?.let { failed ->
                runCatching { failed.close() }
            }
        }
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
        require(request.imageFilePaths.isEmpty()) {
            "Qwen LiteRT models do not support direct image input."
        }
        require(request.nativeAudioInputs.isEmpty()) {
            "Qwen LiteRT models do not support native audio input."
        }
        require(request.history.isNotEmpty() && request.history.last().role == ChatRole.USER) {
            "Qwen LiteRT backend expects the final history turn to be the user's prompt."
        }

        val initialHistory = request.history.dropLast(1)
        val userTurn = request.history.last()
        recreateConversation(initialHistory, request.thinkingEnabled, request.modelInstruction)

        val activeConversation = conversation
            ?: throw IllegalStateException("Conversation was not created.")

        val rawOutputBuilder = StringBuilder()
        val channelThinkingBuilder = StringBuilder()

        activeConversation.sendMessageAsync(userTurn.text).collect { message ->
            val chunkText = message.contents.toString()
            if (chunkText.isNotEmpty()) {
                rawOutputBuilder.append(chunkText)
            }

            val thoughtChunk = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
            if (thoughtChunk.isNotEmpty()) {
                channelThinkingBuilder.append(thoughtChunk)
            }

            onPartial(
                QwenResponseParser.parseVisibleResponse(
                    rawOutput = rawOutputBuilder.toString(),
                    channelThinking = channelThinkingBuilder.toString().takeIf { it.isNotBlank() }
                )
            )
        }

        QwenResponseParser.parseVisibleResponse(
            rawOutput = rawOutputBuilder.toString(),
            channelThinking = channelThinkingBuilder.toString().takeIf { it.isNotBlank() }
        )
    }

    override fun cancelGeneration() {
        conversation?.cancelProcess()
    }

    override fun close() {
        closeConversation()
        if (::engine.isInitialized && engine.isInitialized()) {
            engine.close()
        }
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
                channels = if (spec.thinkingModeAvailable && !thinkingEnabled) emptyList() else null
            )
        )
    }

    private fun buildSystemInstruction(thinkingEnabled: Boolean, modelInstruction: String): String {
        if (!spec.thinkingModeAvailable) {
            return modelInstruction.trim()
        }

        val thinkingDirective = if (thinkingEnabled) "/think" else "/no_think"
        return "$modelInstruction $thinkingDirective".trim()
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }
}
