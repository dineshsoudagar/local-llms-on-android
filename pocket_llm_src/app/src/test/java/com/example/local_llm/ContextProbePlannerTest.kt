package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextProbePlannerTest {
    @Test
    fun climbsTheLadderWhileSizesPass() {
        assertEquals(4_096, ContextProbePlanner.nextSize(emptyList(), emptyList(), 32_768))
        assertEquals(8_192, ContextProbePlanner.nextSize(listOf(4_096), emptyList(), 32_768))
        assertEquals(24_576, ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384), emptyList(), 32_768))
    }

    @Test
    fun endsAtTheModelMaximum() {
        assertEquals(20_000, ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384), emptyList(), 20_000))
        assertNull(ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384, 20_000), emptyList(), 20_000))
    }

    @Test
    fun stopsWhenTheSmallestSizeFails() {
        assertNull(ContextProbePlanner.nextSize(emptyList(), listOf(4_096), 32_768))
    }

    @Test
    fun bisectsBetweenTheLastPassAndFirstFailAtMostThreeTimes() {
        val passed = mutableListOf(4_096, 8_192, 12_288)
        val failed = mutableListOf(24_576)
        passed += 16_384

        val first = ContextProbePlanner.nextSize(passed, failed, 131_072)
        assertEquals(20_480, first)
        failed += first!!

        val second = ContextProbePlanner.nextSize(passed, failed, 131_072)
        assertEquals(18_432, second)
        passed += second!!

        val third = ContextProbePlanner.nextSize(passed, failed, 131_072)
        assertEquals(19_456, third)
        passed += third!!

        assertNull(ContextProbePlanner.nextSize(passed, failed, 131_072))
    }

    @Test
    fun recommendsEightyPercentRoundedDown() {
        assertEquals(15_360, ContextProbePlanner.recommended(19_456))
        assertEquals(3_072, ContextProbePlanner.recommended(4_096))
        assertNull(ContextProbePlanner.recommended(null))
    }

    @Test
    fun fillerHasAboutOneWordPerTargetToken() {
        val text = ContextProbePlanner.fillerText(1_000)
        val words = text.split(Regex("[\\s.]+")).filter { it.isNotEmpty() }
        assertEquals(1_000, words.size)
        assertTrue(PromptTokenEstimator.estimate(text) >= 1_000)
    }
}
