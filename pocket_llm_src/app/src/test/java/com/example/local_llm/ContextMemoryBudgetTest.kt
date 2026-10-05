package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextMemoryBudgetTest {
    private val gemmaE2bBytes = ModelRegistry.gemma4E2B.approxDownloadBytes
    private val gemmaE4bBytes = ModelRegistry.gemma4E4B.approxDownloadBytes

    // ActivityManager reports a little under the marketed size.
    private val twelveGbPhone = 11_200_000_000L
    private val eightGbPhone = 7_400_000_000L

    @Test
    fun gemmaE2bOnTwelveGbPhoneCapsBelowTheObservedCrash() {
        val limit = ContextMemoryBudget.deviceLimit(gemmaE2bBytes, twelveGbPhone)

        assertTrue("limit $limit should keep 10K working", limit >= 10_240)
        assertTrue("limit $limit should stay below the 20K crash", limit < 20_000)
    }

    @Test
    fun estimateNeverCapsBelowTheTestedBaseline() {
        assertEquals(
            ContextMemoryBudget.TESTED_BASELINE_TOKENS,
            ContextMemoryBudget.deviceLimit(gemmaE4bBytes, eightGbPhone)
        )
        assertEquals(
            ContextMemoryBudget.TESTED_BASELINE_TOKENS,
            ContextMemoryBudget.deviceLimit(gemmaE2bBytes, 2_000_000_000L)
        )
    }

    @Test
    fun requestsWithinTheLimitAreUnchanged() {
        val decision = ContextMemoryBudget.decide(8_192, gemmaE2bBytes, twelveGbPhone, null)

        assertEquals(8_192, decision.effectiveTokens)
        assertFalse(decision.isCapped)
    }

    @Test
    fun oversizedRequestLoadsAtTheDeviceLimit() {
        val decision = ContextMemoryBudget.decide(20_000, gemmaE2bBytes, twelveGbPhone, null)

        assertTrue(decision.isCapped)
        assertFalse(decision.cappedByCrashHistory)
        assertEquals(decision.deviceLimitTokens, decision.effectiveTokens)
    }

    @Test
    fun learnedCrashLimitWinsWhenLower() {
        val decision = ContextMemoryBudget.decide(20_000, gemmaE2bBytes, twelveGbPhone, 4_096)

        assertEquals(4_096, decision.effectiveTokens)
        assertTrue(decision.cappedByCrashHistory)
    }

    @Test
    fun crashStepsDownToBaselineBeforeHalvingFurther() {
        assertEquals(10_240, ContextMemoryBudget.limitAfterCrash(20_480, null))
        assertEquals(8_192, ContextMemoryBudget.limitAfterCrash(12_288, null))
        assertEquals(4_096, ContextMemoryBudget.limitAfterCrash(8_192, null))
        assertEquals(
            ContextMemoryBudget.MIN_LEARNED_TOKENS,
            ContextMemoryBudget.limitAfterCrash(2_048, null)
        )
    }

    @Test
    fun crashNeverRaisesAnEarlierLimit() {
        assertEquals(6_144, ContextMemoryBudget.limitAfterCrash(20_480, 6_144))
    }

    @Test
    fun onlyLargePromptsInLargeContextsAreTracked() {
        assertFalse(ContextMemoryBudget.isHighContextRun(8_192, 8_000))
        assertFalse(ContextMemoryBudget.isHighContextRun(16_384, 4_000))
        assertTrue(ContextMemoryBudget.isHighContextRun(16_384, 12_000))
    }
}
