package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextProbePlannerTest {
    @Test
    fun climbsTheLadderInFourThousandStepsUpTo32K() {
        assertEquals(4_096, ContextProbePlanner.nextSize(emptyList(), emptyList(), 131_072))
        assertEquals(8_192, ContextProbePlanner.nextSize(listOf(4_096), emptyList(), 131_072))
        assertEquals(20_480, ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384), emptyList(), 131_072))
        assertEquals(24_576, ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384, 20_480), emptyList(), 131_072))
    }

    @Test
    fun endsAtTheModelMaximum() {
        assertEquals(22_000, ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384, 20_480), emptyList(), 22_000))
        assertNull(ContextProbePlanner.nextSize(listOf(4_096, 8_192, 12_288, 16_384, 20_480, 22_000), emptyList(), 22_000))
    }

    @Test
    fun stopsWhenTheSmallestSizeFails() {
        assertNull(ContextProbePlanner.nextSize(emptyList(), listOf(4_096), 32_768))
    }

    @Test
    fun bisectsBetweenTheLastPassAndFirstFailAtMostThreeTimes() {
        val passed = mutableListOf(4_096, 8_192, 12_288, 16_384, 20_480)
        val failed = mutableListOf(24_576)

        val first = ContextProbePlanner.nextSize(passed, failed, 131_072)
        assertEquals(22_528, first)
        passed += first!!

        val second = ContextProbePlanner.nextSize(passed, failed, 131_072)
        assertEquals(23_552, second)
        failed += second!!

        assertNull(ContextProbePlanner.nextSize(passed, failed, 131_072))
    }

    @Test
    fun retestStartsAtTheLastLargestSizeAndClimbsFromThere() {
        assertEquals(19_456, ContextProbePlanner.nextSize(emptyList(), emptyList(), 131_072, startTokens = 19_456))
        assertEquals(20_480, ContextProbePlanner.nextSize(listOf(19_456), emptyList(), 131_072, startTokens = 19_456))
    }

    @Test
    fun retestFallsBackToTheBottomWhenTheLastSizeFails() {
        assertEquals(4_096, ContextProbePlanner.nextSize(emptyList(), listOf(19_456), 131_072, startTokens = 19_456))
        val passed = listOf(4_096, 8_192, 12_288, 16_384)
        // The failed start size is not counted as a bisection step.
        assertEquals(17_408, ContextProbePlanner.nextSize(passed, listOf(19_456), 131_072, startTokens = 19_456))
    }

    @Test
    fun recommendsEightyPercentRoundedDown() {
        assertEquals(15_360, ContextProbePlanner.recommended(19_456))
        assertEquals(3_072, ContextProbePlanner.recommended(4_096))
        assertNull(ContextProbePlanner.recommended(null))
    }

    @Test
    fun fillerChunksAddUpToTheTarget() {
        val chunks = ContextProbePlanner.fillerChunks(8_000)
        assertEquals(8, chunks.size)
        val words = chunks.sumOf { chunk -> chunk.split(Regex("[\\s.]+")).count { it.isNotEmpty() } }
        assertEquals(8_000, words)
    }

    @Test
    fun fillerHasAboutOneWordPerTargetToken() {
        val text = ContextProbePlanner.fillerText(1_000)
        val words = text.split(Regex("[\\s.]+")).filter { it.isNotEmpty() }
        assertEquals(1_000, words.size)
        assertTrue(PromptTokenEstimator.estimate(text) >= 1_000)
    }
}
