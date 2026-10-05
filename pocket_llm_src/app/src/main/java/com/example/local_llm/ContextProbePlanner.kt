package com.example.local_llm

/**
 * Pure search rules for the on-device context test. Sizes grow along a ladder until one fails,
 * then a short bisection narrows the gap. Each size is run in a separate process so a native
 * out-of-memory crash only ends that step.
 */
object ContextProbePlanner {
    private val LADDER = listOf(
        4_096, 8_192, 12_288, 16_384, 20_480, 24_576, 28_672, 32_768,
        40_960, 49_152, 57_344, 65_536, 81_920, 98_304, 114_688, 131_072
    )
    private const val GRANULARITY = 1_024
    private const val MAX_BISECTION_STEPS = 3
    private const val FILL_PERCENT = 80
    private const val RECOMMENDED_PERCENT = 80

    // Common words that each tokenize as a single token with a leading space in Gemma and Qwen.
    private val FILLER_WORDS = listOf(
        "the", "house", "river", "green", "small", "water", "light", "table", "story", "people",
        "music", "garden", "morning", "city", "road", "window", "paper", "friend", "family", "school"
    )

    private const val FILL_CHUNKS = 8

    /**
     * Next context size to try, or null when the search is finished. A retest starts at
     * [startTokens], the largest size that worked last time, instead of climbing from the bottom.
     */
    fun nextSize(passed: List<Int>, failed: List<Int>, maxTokens: Int, startTokens: Int? = null): Int? {
        val tested = passed + failed
        val start = startTokens?.takeIf { it in (LADDER.first() + 1)..maxTokens }
        if (start != null && tested.isEmpty()) return start
        val bestPass = passed.maxOrNull()
        val lowestFail = failed.minOrNull()
        val ladder = LADDER.filter { it <= maxTokens }.ifEmpty { listOf(maxTokens) }.let {
            if (maxTokens > it.last()) it + maxTokens else it
        }
        // Untested rungs between the best pass and the first failure come before any bisection.
        ladder.firstOrNull { it > (bestPass ?: 0) && it < (lowestFail ?: Int.MAX_VALUE) && it !in tested }
            ?.let { return it }
        if (lowestFail == null || bestPass == null) {
            // Either the top was reached or even the smallest size failed.
            return null
        }
        val bisections = tested.count { it !in ladder && it != start }
        if (bisections >= MAX_BISECTION_STEPS) return null
        val mid = (bestPass + lowestFail) / 2 / GRANULARITY * GRANULARITY
        return mid.takeIf { it > bestPass && it < lowestFail && lowestFail - bestPass > GRANULARITY }
    }

    fun recommended(largestPassed: Int?): Int? =
        largestPassed?.let { (it.toLong() * RECOMMENDED_PERCENT / 100 / GRANULARITY * GRANULARITY).toInt() }
            ?.takeIf { it > 0 }

    fun fillTarget(contextTokens: Int): Int = contextTokens * FILL_PERCENT / 100

    /** The prefill split into parts, so the test can report how far the context is filled. */
    fun fillerChunks(targetTokens: Int): List<String> {
        val parts = FILL_CHUNKS.coerceAtMost(targetTokens.coerceAtLeast(1))
        return List(parts) { fillerText(targetTokens / parts) }
    }

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

/** The context test decodes only a few tokens after the prefill; parts before the last decode one. */
internal const val CONTEXT_PROBE_OUTPUT_TOKENS = 16
internal const val CONTEXT_PROBE_PART_OUTPUT_TOKENS = 1
