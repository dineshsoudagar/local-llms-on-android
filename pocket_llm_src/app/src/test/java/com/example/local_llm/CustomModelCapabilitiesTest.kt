package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CustomModelCapabilitiesTest {
    private fun model(inputs: CustomModelCapabilities? = null) =
        CustomLiteRtSpec("model.litertlm", 1024, "custom_test", "Test model", inputs)

    @Test
    fun legacyUnknownInputsDoNotEnableNativeEncoders() {
        assertFalse(model().directImageInputAvailable)
        assertFalse(model().directAudioInputAvailable)
        assertTrue(model().deviceRecommendation.contains("unknown"))
    }

    @Test
    fun independentModalitiesEnableOnlyTheirOwnEncoder() {
        val visionOnly = model(CustomModelCapabilities(true, true, false, false))
        val audioOnly = model(CustomModelCapabilities(true, false, true, false))
        assertTrue(visionOnly.directImageInputAvailable)
        assertFalse(visionOnly.directAudioInputAvailable)
        assertFalse(audioOnly.directImageInputAvailable)
        assertTrue(audioOnly.directAudioInputAvailable)
        assertEquals("text, image", visionOnly.detectedInputs!!.inputSummary())
    }

    @Test
    fun videoDetectionDoesNotEnableImageOrAudio() {
        val video = model(CustomModelCapabilities(true, false, false, true))
        assertFalse(video.directImageInputAvailable)
        assertFalse(video.directAudioInputAvailable)
        assertTrue(video.deviceRecommendation.contains("video (not enabled)"))
    }

    @Test
    fun customSegmentsCoverSourceWithOverlapAndRespectTenSecondLimit() {
        val limit = model().nativeAudioSegmentMillis
        val windows = GemmaAudioSegmenter.planDuration(31_000, limit).windows
        assertEquals(0L, windows.first().first)
        assertEquals(31_000L, windows.last().last)
        assertTrue(windows.all { it.last - it.first <= limit })
        windows.zipWithNext().forEach { (previous, next) ->
            assertEquals(AttachmentLimits.AUDIO_SEGMENT_OVERLAP_MILLIS, previous.last - next.first)
        }
        assertEquals(1, GemmaAudioSegmenter.planDuration(limit, limit).segmentCount)
        assertEquals(2, GemmaAudioSegmenter.planDuration(limit + 1, limit).segmentCount)
    }
}
