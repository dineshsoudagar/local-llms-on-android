package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptTokenEstimatorTest {
    private val english = "The quick brown fox jumps over the lazy dog while the farmer watches from the porch. "

    @Test
    fun englishUsesFarLessThanTheByteCountButStaysAboveTypicalTokenizers() {
        val text = english.repeat(20)
        val estimate = PromptTokenEstimator.estimate(text)
        val bytes = text.toByteArray(Charsets.UTF_8).size

        // BPE tokenizers average about 4 characters per English token.
        assertTrue("estimate $estimate should be above ${text.length / 4}", estimate > text.length / 4)
        assertTrue("estimate $estimate should be well below $bytes bytes", estimate * 2 < bytes)
    }

    @Test
    fun digitsCountOnePerToken() {
        val digits = "1234567890".repeat(10)
        assertTrue(PromptTokenEstimator.estimate(digits) >= digits.length)
    }

    @Test
    fun nonLatinScriptsCountPerCharacter() {
        val hindi = "नमस्ते दुनिया"
        val chinese = "你好世界"
        assertTrue(PromptTokenEstimator.estimate(hindi) >= hindi.count { !it.isWhitespace() })
        assertTrue(PromptTokenEstimator.estimate(chinese) >= chinese.length)
    }

    @Test
    fun emojiCountsAsSeveralTokens() {
        assertTrue(PromptTokenEstimator.estimate("😀😀") >= 6)
    }

    @Test
    fun emptyTextIsFree() {
        assertEquals(0, PromptTokenEstimator.estimate(""))
    }

    @Test
    fun calibrationOnlyGrowsAndKeepsAMargin() {
        val calibration = TokenEstimateCalibration()
        assertEquals(100, calibration.apply(100))

        calibration.observe(rawEstimate = 100, measuredTokens = 80)
        assertEquals(100, calibration.apply(100))

        calibration.observe(rawEstimate = 100, measuredTokens = 150)
        val raised = calibration.apply(100)
        assertTrue("raised to $raised", raised in 165..166)

        calibration.observe(rawEstimate = 100, measuredTokens = 120)
        assertEquals(raised, calibration.apply(100))
    }

    @Test
    fun liteRtKeepsRoomForTheReply() {
        assertEquals(256, liteRtMinimumOutputReserve(512))
        assertEquals(1_792, liteRtMinimumOutputReserve(14_336))
        assertEquals(2_048, liteRtMinimumOutputReserve(128_000))
    }
}
