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
