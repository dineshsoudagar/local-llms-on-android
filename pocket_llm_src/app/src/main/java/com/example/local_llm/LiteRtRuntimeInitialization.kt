package com.example.local_llm

import kotlinx.coroutines.CancellationException

/** Lazy executor compilation must succeed before a backend attempt is accepted. */
internal fun <T : AutoCloseable> createUsableLiteRtRuntime(
    create: () -> T,
    initializeAndValidate: (T) -> Unit
): Result<T> {
    var candidate: T? = null
    return try {
        val runtime = create()
        candidate = runtime
        initializeAndValidate(runtime)
        Result.success(runtime)
    } catch (error: Throwable) {
        candidate?.let { runCatching { it.close() } }
        // Cancellation and VM failures must not trigger more expensive load attempts.
        if (error is CancellationException || error is Error) throw error
        Result.failure(error)
    }
}

/** Compares the runtime's KV-cache token count with the estimate so later estimates stay above it. */
internal fun ChatBackend.observeTokenUsage(
    calibration: TokenEstimateCalibration,
    conversation: com.google.ai.edge.litertlm.Conversation,
    request: InferenceRequest,
    responseText: String
) {
    val measured = runCatching { conversation.getTokenCount() }.getOrNull() ?: return
    val estimated = rawPromptTokenEstimate(request) + PromptTokenEstimator.estimate(responseText)
    calibration.observe(estimated, measured)
}
