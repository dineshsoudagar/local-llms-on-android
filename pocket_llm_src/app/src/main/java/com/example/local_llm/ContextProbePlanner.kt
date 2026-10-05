package com.example.local_llm

/**
 * Pure search rules for the on-device context test. Sizes grow along a ladder until one fails,
 * then a short bisection narrows the gap. Each size is run in a separate process so a native
 * out-of-memory crash only ends that step.
 */
object ContextProbePlanner {
    private val LADDER = listOf(4_096, 8_192, 12_288, 16_384, 24_576, 32_768, 49_152, 65_536, 98_304, 131_072)
    private const val GRANULARITY = 1_024
    private const val MAX_BISECTION_STEPS = 3
    private const val FILL_PERCENT = 80
    private const val RECOMMENDED_PERCENT = 80

    // Common words that each tokenize as a single token with a leading space in Gemma and Qwen.
    private val FILLER_WORDS = listOf(
        "the", "house", "river", "green", "small", "water", "light", "table", "story", "people",
        "music", "garden", "morning", "city", "road", "window", "paper", "friend", "family", "school"
    )

    /** Next context size to try, or null when the search is finished. */
    fun nextSize(passed: List<Int>, failed: List<Int>, maxTokens: Int): Int? {
        val bestPass = passed.maxOrNull()
        val lowestFail = failed.minOrNull()
        val ladder = LADDER.filter { it <= maxTokens }.ifEmpty { listOf(maxTokens) }.let {
            if (maxTokens > it.last()) it + maxTokens else it
        }
        if (lowestFail == null) {
            return ladder.firstOrNull { it > (bestPass ?: 0) }
        }
        if (bestPass == null) {
            // Even the smallest size failed: nothing below it is worth testing.
            return null
        }
        val bisections = (passed + failed).count { it !in ladder }
        if (bisections >= MAX_BISECTION_STEPS) return null
        val mid = (bestPass + lowestFail) / 2 / GRANULARITY * GRANULARITY
        return mid.takeIf { it > bestPass && it < lowestFail && lowestFail - bestPass > GRANULARITY }
    }

    fun recommended(largestPassed: Int?): Int? =
        largestPassed?.let { (it.toLong() * RECOMMENDED_PERCENT / 100 / GRANULARITY * GRANULARITY).toInt() }
            ?.takeIf { it > 0 }

    fun fillTarget(contextTokens: Int): Int = contextTokens * FILL_PERCENT / 100

    /** About [targetTokens] tokens of plain words for the prefill. */
    fun fillerText(targetTokens: Int): String {
        val builder = StringBuilder(targetTokens * 7)
        for (index in 0 until targetTokens.coerceAtLeast(1)) {
            if (index > 0) builder.append(if (index % 24 == 0) ".\n" else " ")
            builder.append(FILLER_WORDS[(index * 7 + index / FILLER_WORDS.size) % FILLER_WORDS.size])
        }
        return builder.append('.').toString()
    }
}

/** The context test decodes only a few tokens after the prefill. */
internal const val CONTEXT_PROBE_OUTPUT_TOKENS = 16
