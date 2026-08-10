package com.example.local_llm

import java.io.File
import java.nio.file.Files
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GemmaAudioLifecycleTest {
    @Test
    fun planningBeforeConfirmationCreatesNoSegmentFiles() {
        val root = Files.createTempDirectory("gemma-plan").toFile()
        val segmentDirectory = File(root, "segments")

        val plan = GemmaAudioSegmenter.planDuration(60_000L)

        assertEquals(3, plan.segmentCount)
        assertEquals(4, plan.inferencePasses)
        assertFalse(segmentDirectory.exists())
        root.deleteRecursively()
    }

    @Test
    fun planRetainsThirtySecondDirectThresholdAndOverlapWindows() {
        assertEquals(1, GemmaAudioSegmenter.planDuration(30_000L).segmentCount)
        assertEquals(
            listOf(0L..28_000L, 26_000L..54_000L, 52_000L..60_000L),
            GemmaAudioSegmenter.planDuration(60_000L).windows
        )
    }

    @Test
    fun segmentationFailureDeletesEveryCompletedTemporarySegment() {
        val root = Files.createTempDirectory("gemma-failure").toFile()
        val segmentDirectory = File(root, "segments")
        val source = File(root, "source.wav")
        source.writeBytes(ByteArray(44 + 30 * 16_000 * 2))
        val descriptor = audioDescriptor(source, durationMillis = 60_000L)
        val segmenter = GemmaAudioSegmenter(segmentDirectory, useDirectDirectory = true)

        assertTrue(runCatching { runBlocking { segmenter.segment(descriptor) } }.isFailure)
        assertTrue(segmentDirectory.listFiles().orEmpty().isEmpty())
        root.deleteRecursively()
    }

    @Test
    fun activityRoutesSegmentationOffMainAndCleansCancellationAndDestruction() {
        val source = mainSourceFile("java/com/example/local_llm/PocketChatActivity.kt").readText()

        assertTrue(source.contains("withContext(Dispatchers.IO)"))
        assertTrue(source.contains("catch (_: CancellationException) {\n                cleanupPreparedGemmaAudio()"))
        assertTrue(source.contains("audioPreparationJob?.cancel()\n        cleanupPreparedGemmaAudio()"))
        assertTrue(source.indexOf("segmenter.plan(descriptor)") < source.indexOf("prepareGemmaAudioAfterConfirmation(descriptor)"))
    }

    private fun audioDescriptor(source: File, durationMillis: Long) = AttachmentDescriptor(
        id = "audio",
        sessionId = "session",
        displayName = "audio.wav",
        mimeType = "audio/wav",
        kind = AttachmentKind.AUDIO,
        status = AttachmentStatus.READY,
        processingRoute = AttachmentProcessingRoute.GEMMA_NATIVE_AUDIO,
        sourcePath = source.absolutePath,
        sizeBytes = source.length(),
        durationMillis = durationMillis,
        useGemmaNativeAudio = true
    )

    private fun mainSourceFile(relativePath: String): File {
        val moduleRelative = File("src/main", relativePath)
        return if (moduleRelative.exists()) moduleRelative else File("app/src/main", relativePath)
    }
}
