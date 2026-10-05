package com.example.local_llm

/** Older turns condensed into [summary]; turns up to and including [throughTurnId] are no longer sent verbatim. */
data class ChatCompaction(
    val summary: String,
    val throughTurnId: String
)

/**
 * Pure rules for keeping a long chat inside the context window. Instead of silently dropping the
 * oldest turns, they are summarized by the model and the summary rides along in the system
 * instruction while recent turns stay verbatim.
 */
object ChatCompactionPlanner {
    /** Compact once the prompt needs more than this share of the context. */
    private const val TRIGGER_PERCENT = 75
    /** Recent turns kept verbatim after compacting. */
    private const val KEEP_PERCENT = 35
    /** Largest slice of old turns sent to the model in one summarization call. */
    private const val BATCH_PERCENT = 45
    /** Hard ceiling on the stored summary, as a share of the context. */
    private const val SUMMARY_PERCENT = 12
    const val SUMMARY_OUTPUT_RESERVE_PERCENT = 15
    /** Share of the context that chat history may take when an attachment is sent. */
    const val ATTACHMENT_HISTORY_PERCENT = 25

    private val THINK_BLOCK = Regex("(?s)<think>.*?</think>")

    const val SUMMARY_INSTRUCTION =
        "You condense chat transcripts. Write a compact summary in plain sentences that keeps " +
            "facts, names, numbers, decisions, open questions, and anything the user asked to remember. " +
            "Do not add commentary, greetings, or new information."

    /** Turns still sent verbatim. A compaction whose turn no longer exists is ignored. */
    fun visibleTurns(turns: List<ChatTurn>, compaction: ChatCompaction?): List<ChatTurn> {
        if (compaction == null) return turns
        val index = turns.indexOfFirst { it.id == compaction.throughTurnId }
        return if (index < 0) turns else turns.subList(index + 1, turns.size)
    }

    fun isActive(turns: List<ChatTurn>, compaction: ChatCompaction?): Boolean =
        compaction != null && turns.any { it.id == compaction.throughTurnId }

    fun shouldCompact(promptTokens: Int, contextTokens: Int): Boolean =
        promptTokens.toLong() * 100 > contextTokens.toLong() * TRIGGER_PERCENT

    /**
     * Index where the verbatim tail starts. The tail always holds the final turn, starts at a
     * plain user turn, and fits the keep budget when possible. Returns 0 when nothing can be
     * summarized without splitting the latest exchange.
     */
    fun tailStart(turns: List<ChatTurn>, contextTokens: Int, estimateTurn: (ChatTurn) -> Int): Int {
        if (turns.size < 2) return 0
        val keepBudget = contextTokens.toLong() * KEEP_PERCENT / 100
        var used = 0L
        var start = turns.size
        var bestStart = -1
        for (index in turns.indices.reversed()) {
            used += estimateTurn(turns[index])
            if (index < turns.size - 1 && used > keepBudget) break
            start = index
            if (turns[start].isPlainUser) bestStart = start
        }
        if (bestStart < 0) {
            // Even the latest exchange exceeds the budget: keep from its user turn.
            bestStart = turns.indexOfLast { it.isPlainUser }.coerceAtLeast(0)
        }
        return bestStart
    }

    /**
     * Latest turns that fit [budgetTokens], starting at a plain user turn. Empty when not even
     * the latest exchange fits, so a long chat never crowds out an attachment.
     */
    fun recentTurns(turns: List<ChatTurn>, budgetTokens: Int, estimateTurn: (ChatTurn) -> Int): List<ChatTurn> {
        var used = 0L
        var bestStart = turns.size
        for (index in turns.indices.reversed()) {
            used += estimateTurn(turns[index])
            if (used > budgetTokens) break
            if (turns[index].isPlainUser) bestStart = index
        }
        return turns.subList(bestStart, turns.size).toList()
    }

    /** Splits [turns] into consecutive groups that each fit [budgetTokens]. */
    fun batches(turns: List<ChatTurn>, budgetTokens: Int, estimateTurn: (ChatTurn) -> Int): List<List<ChatTurn>> {
        val result = mutableListOf<List<ChatTurn>>()
        var current = mutableListOf<ChatTurn>()
        var used = 0
        for (turn in turns) {
            val cost = estimateTurn(turn)
            if (current.isNotEmpty() && used + cost > budgetTokens) {
                result += current
                current = mutableListOf()
                used = 0
            }
            current += turn
            used += cost
        }
        if (current.isNotEmpty()) result += current
        return result
    }

    fun batchBudget(contextTokens: Int): Int = contextTokens * BATCH_PERCENT / 100

    fun summaryCharLimit(contextTokens: Int): Int = contextTokens * SUMMARY_PERCENT / 100

    /** Target length handed to the model; the char limit is the backstop. */
    fun summaryWordTarget(contextTokens: Int): Int = (summaryCharLimit(contextTokens) / 7).coerceIn(80, 600)

    fun summaryPrompt(previousSummary: String?, turns: List<ChatTurn>, maxWords: Int, maxTurnChars: Int): String =
        buildString {
            append("Summarize the conversation below in at most ").append(maxWords).append(" words.\n")
            if (!previousSummary.isNullOrBlank()) {
                append("Merge it with this summary of what came before it:\n")
                append(previousSummary.trim()).append("\n\n")
            }
            append("Conversation:\n")
            turns.forEach { turn ->
                val speaker = when {
                    turn.isToolResult -> "Tool"
                    turn.role == ChatRole.USER -> "User"
                    else -> "Assistant"
                }
                append(speaker).append(": ").append(turn.text.trim().take(maxTurnChars)).append('\n')
            }
        }

    fun cleanSummary(raw: String, maxChars: Int): String {
        val text = raw.replace(THINK_BLOCK, "").substringAfterLast("</think>").trim()
        if (text.length <= maxChars) return text
        val cut = text.take(maxChars)
        val sentenceEnd = cut.lastIndexOf(". ")
        return if (sentenceEnd > maxChars / 2) cut.substring(0, sentenceEnd + 1) else cut.trimEnd()
    }

    fun instructionWithSummary(base: String, summary: String?): String {
        if (summary.isNullOrBlank()) return base
        val note = "Summary of the earlier part of this conversation (older messages were condensed " +
            "to fit the context window):\n$summary"
        return if (base.isBlank()) note else "$base\n\n$note"
    }

    private val ChatTurn.isPlainUser: Boolean
        get() = role == ChatRole.USER && !isToolResult
}
