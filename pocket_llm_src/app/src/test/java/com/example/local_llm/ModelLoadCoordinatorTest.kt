package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelLoadCoordinatorTest {
    @Test
    fun interruptedInitializationBecomesFailureAndSuppressesAutoLoad() {
        val coordinator = ModelLoadCoordinator(
            ModelLoadRecord("qwen", "attempt-1", ModelLoadPhase.INITIALIZING, 10L)
        )

        val recovered = coordinator.recoverInterrupted(20L)

        assertEquals(ModelLoadPhase.FAILED, recovered?.phase)
        assertTrue(recovered?.failureReason.orEmpty().contains("stopped while"))
        assertFalse(coordinator.shouldAutoLoad("qwen"))
        assertTrue(coordinator.shouldAutoLoad("gemma"))
    }

    @Test
    fun successClearsOnlyTheMatchingAttempt() {
        val coordinator = ModelLoadCoordinator()
        val first = coordinator.begin("qwen", 1L, "first")
        val second = coordinator.begin("gemma", 2L, "second")

        assertFalse(coordinator.succeed(first.modelId, first.attemptId))
        assertEquals(second, coordinator.record)
        assertTrue(coordinator.succeed(second.modelId, second.attemptId))
        assertNull(coordinator.record)
    }

    @Test
    fun cancellationAndLateFailureCannotPoisonNewAttempt() {
        val coordinator = ModelLoadCoordinator()
        val old = coordinator.begin("qwen", 1L, "old")
        assertTrue(coordinator.cancel(old.modelId, old.attemptId))
        val current = coordinator.begin("gemma", 2L, "current")

        assertFalse(coordinator.fail("qwen", "old", "late error", 3L))
        assertEquals(current, coordinator.record)
    }

    @Test
    fun retryFailureAndDeleteTransitionsRemainExplicit() {
        val coordinator = ModelLoadCoordinator()
        val attempt = coordinator.begin("qwen", 1L, "retry")
        assertTrue(coordinator.fail("qwen", attempt.attemptId, "bad model", 2L))
        assertEquals(ModelLoadPhase.FAILED, coordinator.record?.phase)

        // Continuing without a model deliberately leaves this record unchanged.
        assertFalse(coordinator.shouldAutoLoad("qwen"))
        assertTrue(coordinator.clearForModel("qwen"))
        assertNull(coordinator.record)
    }
}
