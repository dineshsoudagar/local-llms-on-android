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
    private val spec: ModelDescriptor,
    private val modelFileResolver: ModelFileResolver,
    private val runtimeSettings: ModelRuntimeSettings = ModelRuntimeSettings(
        ModelRuntimeSettingsLimits.LITERT_DEFAULT_CONTEXT_LENGTH
    ),
    private val initializationPolicy: BackendInitializationPolicy = BackendInitializationPolicy()
) : ChatBackend {

    override val capabilities = BackendCapabilities(
        supportsNativeToolCalling = spec is QwenLiteRtSpec,
        contextWindowTokens = runtimeSettings.contextLengthTokens
    )

    companion object {
        private const val THOUGHT_CHANNEL_NAME = "thought"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)

        if (initializationPolicy.cpuOnly) {
            // A previous GPU load crashed the process in native code; never retry the GPU path.
            engine = createInitializedEngine(modelFile.absolutePath, Backend.CPU()).getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize LiteRT-LM on CPU (GPU is disabled after a previous crash): ${cpuError.message}",
                    cpuError
                )
            }
            return@withContext
        }

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
                    maxNumTokens = runtimeSettings.contextLengthTokens,
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
        require(request.imageFilePaths.isEmpty()) {
            "This text model does not support direct image input. Use OCR."
        }
        require(request.nativeAudioInputs.isEmpty()) {
            "This text model does not support native audio input."
        }
        val boundedHistory = fitHistoryWithinContext(request)
        require(
            boundedHistory.isNotEmpty() &&
                (boundedHistory.last().role == ChatRole.USER || boundedHistory.last().isToolResult)
        ) {
            "Qwen LiteRT backend expects the final history turn to be a user or tool message."
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

        val rawOutputBuilder = StringBuilder()
        val channelThinkingBuilder = StringBuilder()

        var externalToolCalls = emptyList<ExternalToolCall>()
        activeConversation.sendMessageAsync(userTurn.toLiteRtMessage()).collect { message ->
            val chunkText = message.visibleText()
            if (chunkText.isNotEmpty()) {
                rawOutputBuilder.append(chunkText)
            }

            val thoughtChunk = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
            if (thoughtChunk.isNotEmpty()) {
                channelThinkingBuilder.append(thoughtChunk)
            }

            val parsed = if (spec is CustomLiteRtSpec) {
                BackendResponse(rawOutputBuilder.toString(), toolCalls = externalToolCalls)
            } else QwenResponseParser.parseVisibleResponse(
                    rawOutput = rawOutputBuilder.toString(),
                    channelThinking = channelThinkingBuilder.toString().takeIf { it.isNotBlank() }
                )
            if (message.toolCalls.isNotEmpty()) {
                externalToolCalls = nativeToolCallsToExternal(
                    message.toolCalls,
                    externalToolCalls,
                    request.parallelToolCalls
                )
            }
            onPartial(
                parsed.copy(toolCalls = externalToolCalls)
            )
        }

        val finalResponse = if (spec is CustomLiteRtSpec) {
            BackendResponse(rawOutputBuilder.toString())
        } else QwenResponseParser.parseVisibleResponse(
            rawOutput = rawOutputBuilder.toString(),
            channelThinking = channelThinkingBuilder.toString().takeIf { it.isNotBlank() }
        )
        finalResponse.copy(toolCalls = externalToolCalls)
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
                tools = externalToolProviders(tools),
                automaticToolCalling = false,
                channels = if (spec is CustomLiteRtSpec || (spec is QwenLiteRtSpec && !thinkingEnabled)) emptyList() else null
            )
        )
    }

    private fun buildSystemInstruction(thinkingEnabled: Boolean, modelInstruction: String): String {
        if (spec !is QwenLiteRtSpec || !spec.thinkingModeAvailable) {
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
