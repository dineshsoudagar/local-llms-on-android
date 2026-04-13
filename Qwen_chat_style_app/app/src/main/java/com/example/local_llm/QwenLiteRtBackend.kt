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
        private const val THINK_OPEN_TAG = "<think>"
        private const val THINK_CLOSE_TAG = "</think>"
        private val ASSISTANT_PREFIX_REGEX = Regex("^\\s*Assistant\\s*:\\s*", RegexOption.IGNORE_CASE)
        private val CONTROL_MARKER_REGEX = Regex("<\\|[^>]+\\|>")
        private val TRAILING_GARBAGE_REGEX = Regex(
            "(?<=[A-Za-z0-9.!?])(?:[\\u00A0-\\u024F\\uFFFD]{2,}|\\uFFFD+)$"
        )
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
                parseVisibleResponse(
                    rawOutput = rawOutputBuilder.toString(),
                    channelThinking = channelThinkingBuilder.toString().takeIf { it.isNotBlank() }
                )
            )
        }

        parseVisibleResponse(
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

    private fun parseVisibleResponse(
        rawOutput: String,
        channelThinking: String?
    ): BackendResponse {
        val normalizedOutput = rawOutput.replace("\r\n", "\n")
        val extracted = extractThinkSections(normalizedOutput)
        val visibleThinking = channelThinking
            ?.takeIf { it.isNotBlank() }
            ?: extracted.thinkingText?.let(::sanitizeThinkingText)

        return BackendResponse(
            text = sanitizeAnswerText(extracted.answerText),
            thinkingText = visibleThinking?.takeIf { it.isNotBlank() }
        )
    }

    private fun extractThinkSections(rawOutput: String): ParsedOutput {
        if (rawOutput.isBlank()) {
            return ParsedOutput(answerText = "", thinkingText = null)
        }

        val answerBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()
        var cursor = 0

        while (cursor < rawOutput.length) {
            val thinkStart = rawOutput.indexOf(THINK_OPEN_TAG, startIndex = cursor)
            if (thinkStart < 0) {
                answerBuilder.append(rawOutput.substring(cursor))
                break
            }

            if (thinkStart > cursor) {
                answerBuilder.append(rawOutput.substring(cursor, thinkStart))
            }

            val thoughtStart = thinkStart + THINK_OPEN_TAG.length
            val thinkEnd = rawOutput.indexOf(THINK_CLOSE_TAG, startIndex = thoughtStart)
            if (thinkEnd < 0) {
                thinkingBuilder.append(rawOutput.substring(thoughtStart))
                cursor = rawOutput.length
                break
            }

            thinkingBuilder.append(rawOutput.substring(thoughtStart, thinkEnd))
            cursor = thinkEnd + THINK_CLOSE_TAG.length
        }

        return ParsedOutput(
            answerText = answerBuilder.toString(),
            thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() }
        )
    }

    private fun sanitizeThinkingText(thinkingText: String): String {
        return thinkingText
            .replace(THINK_OPEN_TAG, "")
            .replace(THINK_CLOSE_TAG, "")
            .trim()
    }

    private fun sanitizeAnswerText(answerText: String): String {
        return answerText
            .replace(CONTROL_MARKER_REGEX, "")
            .replaceFirst(ASSISTANT_PREFIX_REGEX, "")
            .replace(TRAILING_GARBAGE_REGEX, "")
            .trim()
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }

    private data class ParsedOutput(
        val answerText: String,
        val thinkingText: String?
    )
}
