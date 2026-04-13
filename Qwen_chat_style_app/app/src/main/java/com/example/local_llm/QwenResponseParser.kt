package com.example.local_llm

object QwenResponseParser {

    private const val THINK_OPEN_TAG = "<think>"
    private const val THINK_CLOSE_TAG = "</think>"
    private val ASSISTANT_PREFIX_REGEX = Regex("^\\s*Assistant\\s*:\\s*", RegexOption.IGNORE_CASE)
    private val CONTROL_MARKER_REGEX = Regex("<\\|[^>]+\\|>")
    private val TRAILING_GARBAGE_REGEX = Regex(
        "(?<=[A-Za-z0-9.!?])(?:[\\u00A0-\\u024F\\uFFFD]{2,}|\\uFFFD+)$"
    )

    fun parseVisibleResponse(
        rawOutput: String,
        channelThinking: String? = null
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

    private data class ParsedOutput(
        val answerText: String,
        val thinkingText: String?
    )
}
