package com.example.local_llm

private const val KIB = 1024L
private const val GIB = 1024L * 1024L * 1024L

data class ContextDecision(
    val requestedTokens: Int,
    val effectiveTokens: Int,
    val deviceLimitTokens: Int,
    val learnedLimitTokens: Int?
) {
    val isCapped: Boolean get() = effectiveTokens < requestedTokens
    val cappedByCrashHistory: Boolean
        get() = isCapped && learnedLimitTokens != null && learnedLimitTokens < deviceLimitTokens
}

/**
 * Pure sizing rules for LiteRT context length. A large maxNumTokens can exhaust native or GPU
 * memory and crash the runtime with a native signal that Kotlin cannot catch, so the context is
 * capped before the engine is created instead of relying on exception-based fallback.
 */
object ContextMemoryBudget {
    /** Values at or below the tested baseline are never reduced by the memory estimate. */
    const val TESTED_BASELINE_TOKENS = ModelRuntimeSettingsLimits.HIGH_CONTEXT_WARNING_THRESHOLD
    const val MIN_LEARNED_TOKENS = ModelRuntimeSettingsLimits.LITERT_DEFAULT_CONTEXT_LENGTH
    private const val ROUNDING_TOKENS = 1_024
    private const val RUNTIME_OVERHEAD_BYTES = 1L * GIB
    private const val MIN_BYTES_PER_TOKEN = 64L * KIB

    // Estimated per-token cost (KV cache plus GPU working buffers that grow with context),
    // scaled from model size. Calibrated so Gemma 4 E2B (2.58 GB) on a 12 GB phone caps near
    // 15K: 10K-15K worked there and 20K crashed. Tune with on-device measurements.
    private const val MODEL_BYTES_PER_CONTEXT_TOKEN = 20_000L

    fun bytesPerToken(modelBytes: Long): Long =
        maxOf(MIN_BYTES_PER_TOKEN, modelBytes / MODEL_BYTES_PER_CONTEXT_TOKEN)

    /** Largest context the device is expected to hold for a model of [modelBytes]. */
    fun deviceLimit(modelBytes: Long, totalMemoryBytes: Long): Int {
        // Leave half of RAM to the OS, other apps and this app's own heap.
        val budget = totalMemoryBytes / 2L - modelBytes - RUNTIME_OVERHEAD_BYTES
        val tokens = (budget.coerceAtLeast(0L) / bytesPerToken(modelBytes))
            .coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        return maxOf(TESTED_BASELINE_TOKENS, roundDown(tokens))
    }

    fun decide(
        requestedTokens: Int,
        modelBytes: Long,
        totalMemoryBytes: Long,
        learnedLimitTokens: Int?
    ): ContextDecision {
        val deviceLimit = deviceLimit(modelBytes, totalMemoryBytes)
        val effective = minOf(requestedTokens, deviceLimit, learnedLimitTokens ?: Int.MAX_VALUE)
        return ContextDecision(requestedTokens, effective, deviceLimit, learnedLimitTokens)
    }

    /**
     * Limit to apply after the process died while running at [crashedTokens]. Steps down to the
     * tested baseline first, then halves, so one bad run never drops far below what works.
     */
    fun limitAfterCrash(crashedTokens: Int, previousLimit: Int?): Int {
        val stepped = if (crashedTokens > TESTED_BASELINE_TOKENS) {
            maxOf(TESTED_BASELINE_TOKENS, roundDown(crashedTokens / 2))
        } else {
            maxOf(MIN_LEARNED_TOKENS, roundDown(crashedTokens / 2))
        }
        return minOf(stepped, previousLimit ?: Int.MAX_VALUE)
    }

    /** Generation is only tracked when both the context and the prompt are past the baseline. */
    fun isHighContextRun(contextTokens: Int, promptTokens: Int): Boolean =
        contextTokens > TESTED_BASELINE_TOKENS && promptTokens > TESTED_BASELINE_TOKENS

    private fun roundDown(tokens: Int): Int = tokens / ROUNDING_TOKENS * ROUNDING_TOKENS
}
