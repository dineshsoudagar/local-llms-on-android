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
    private val modelFileResolver: ModelFileResolver
) : ChatBackend {

    companion object {
        private const val THOUGHT_CHANNEL_NAME = "thought"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveAssetToFile(spec.modelAssetName)

        val gpuResult = runCatching {
            Engine(
                EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = Backend.GPU(),
                    cacheDir = context.cacheDir.absolutePath
                )
            ).apply { initialize() }
        }

        engine = gpuResult.getOrElse { gpuError ->
            runCatching {
                Engine(
                    EngineConfig(
                        modelPath = modelFile.absolutePath,
                        backend = Backend.CPU(),
                        cacheDir = context.cacheDir.absolutePath
                    )
                ).apply { initialize() }
            }.getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize LiteRT-LM GPU (${gpuError.message}) and CPU (${cpuError.message}).",
                    cpuError
                )
            }
        }
    }

    override suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) {
        recreateConversation(history, thinkingEnabled)
    }

    override suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(history.isNotEmpty() && history.last().role == ChatRole.USER) {
            "Qwen LiteRT backend expects the final history turn to be the user's prompt."
        }

        val initialHistory = history.dropLast(1)
        val userTurn = history.last()
        recreateConversation(initialHistory, thinkingEnabled)

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

    private fun recreateConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) {
        closeConversation()
        conversation = engine.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(buildSystemInstruction(thinkingEnabled)),
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

    private fun buildSystemInstruction(thinkingEnabled: Boolean): String {
        val thinkingDirective = if (thinkingEnabled) "/think" else "/no_think"
        return "${spec.defaultSystemInstruction} $thinkingDirective".trim()
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }
}
