package com.example.local_llm

import java.io.ByteArrayOutputStream
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioNormalizationPolicyTest {
    @Test
    fun missingUnderstatedAndOverstatedMetadataDoNotControlDecodedDuration() {
        val estimates = listOf<Long?>(null, 1_000L, 99_000_000_000L)

        estimates.forEach { estimate ->
            val tracker = DecodedAudioDurationTracker(estimate, maximumSamples = 16_000L)
            tracker.recordSamples(16_000L)
            assertEquals(16_000L, tracker.normalizedSampleCount)
            assertEquals(1_000L, tracker.durationMillis)
        }
    }

    @Test
    fun exactTwoHourBoundaryIsAcceptedAndOneSampleOverIsRejected() {
        val tracker = DecodedAudioDurationTracker(containerDurationEstimateMicros = null)
        tracker.recordSamples(AttachmentLimits.MAX_NORMALIZED_AUDIO_SAMPLES)

        assertEquals(AttachmentLimits.MAX_AUDIO_DURATION_MILLIS, tracker.durationMillis)
        val error = runCatching { tracker.recordSamples(1L) }.exceptionOrNull()
        assertNotNull(error)
        assertTrue(error!!.message.orEmpty().contains("2 hours"))
    }

    @Test
    fun resamplerStopsBeforeWritingOneNormalizedSampleOverTheLimit() {
        val output = ByteArrayOutputStream()
        val tracker = DecodedAudioDurationTracker(null, maximumSamples = 2L)
        val resampler = Pcm16MonoResampler(output, tracker)

        val error = runCatching {
            resampler.consume(ByteArray(3 * 2), decodedSampleRate = 16_000, decodedChannels = 1)
        }.exceptionOrNull()

        assertNotNull(error)
        assertEquals(2L, tracker.normalizedSampleCount)
        assertEquals(4, output.size())
    }

    @Test
    fun stereo48kAndUnusual8kRatesProduceDeterministic16kMonoSamples() {
        val stereoOutput = ByteArrayOutputStream()
        val stereoTracker = DecodedAudioDurationTracker(null, maximumSamples = 100L)
        val stereo = Pcm16MonoResampler(stereoOutput, stereoTracker)
        val sixStereoFrames = ByteArray(6 * 2 * 2)
        stereo.consume(sixStereoFrames.copyOfRange(0, 7), decodedSampleRate = 48_000, decodedChannels = 2)
        stereo.consume(sixStereoFrames.copyOfRange(7, sixStereoFrames.size), decodedSampleRate = 48_000, decodedChannels = 2)
        stereo.finish()

        assertEquals(2L, stereoTracker.normalizedSampleCount)
        assertEquals(4, stereoOutput.size())

        val lowRateOutput = ByteArrayOutputStream()
        val lowRateTracker = DecodedAudioDurationTracker(null, maximumSamples = 100L)
        Pcm16MonoResampler(lowRateOutput, lowRateTracker).apply {
            consume(ByteArray(3 * 2), decodedSampleRate = 8_000, decodedChannels = 1)
            finish()
        }
        assertEquals(6L, lowRateTracker.normalizedSampleCount)
        assertEquals(12, lowRateOutput.size())
    }

    @Test
    fun partialOutputIsRemovedAfterFailureCleanup() {
        val partial = File.createTempFile("partial-normalized-", ".wav")
        partial.writeBytes(byteArrayOf(1, 2, 3))

        cleanupPartialAudioOutput(partial)

        assertFalse(partial.exists())
    }
}
