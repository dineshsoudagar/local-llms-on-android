package com.example.local_llm

import kotlinx.coroutines.CancellationException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtRuntimeInitializationTest {
    private class FakeRuntime : AutoCloseable {
        var initialized = false
        var closed = false
        override fun close() { closed = true }
    }

    @Test
    fun lazyConversationFailureClosesGpuRuntimeAndAllowsCpuRetry() {
        val gpu = FakeRuntime()
        val cpu = FakeRuntime()
        val tried = mutableListOf<String>()
        val error = IllegalStateException("Some ops are not accelerated")
        val gpuResult = createUsableLiteRtRuntime({ gpu }) {
            tried += "GPU vision"
            it.initialized = true
            throw error // Engine initialized; first conversation triggers vision compilation.
        }
        assertSame(error, gpuResult.exceptionOrNull())
        assertTrue(gpu.initialized)
        assertTrue(gpu.closed)
        val loaded = gpuResult.getOrElse {
            createUsableLiteRtRuntime({ cpu }) {
                tried += "CPU vision"
                it.initialized = true
            }.getOrThrow()
        }
        assertSame(cpu, loaded)
        assertEquals(listOf("GPU vision", "CPU vision"), tried)
        assertFalse(cpu.closed)
    }

    @Test
    fun engineInitializationFailureAlsoReleasesCandidate() {
        val runtime = FakeRuntime()
        val result = createUsableLiteRtRuntime({ runtime }) { throw IllegalStateException("engine failed") }
        assertTrue(result.isFailure)
        assertTrue(runtime.closed)
    }

    @Test
    fun cancellationReleasesCandidateAndPropagatesWithoutFallback() {
        val runtime = FakeRuntime()
        val cancelled = CancellationException("cancel load")
        val thrown = runCatching {
            createUsableLiteRtRuntime({ runtime }) { throw cancelled }
        }.exceptionOrNull()
        assertSame(cancelled, thrown)
        assertTrue(runtime.closed)
    }

    @Test
    fun outOfMemoryReleasesCandidateAndDoesNotBecomeRetryableFailure() {
        val runtime = FakeRuntime()
        val oom = OutOfMemoryError("no memory")
        val thrown = runCatching {
            createUsableLiteRtRuntime({ runtime }) { throw oom }
        }.exceptionOrNull()
        assertSame(oom, thrown)
        assertTrue(runtime.closed)
    }
}
