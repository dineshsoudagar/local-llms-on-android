package com.example.local_llm

import java.util.UUID
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class AttachmentWorkLifecycleTest {
    @Test
    fun activityRecreationReconnectsRunningImport() {
        val record = record(AttachmentWorkOperation.IMPORT, createdAtMillis = 10L)

        assertSame(record, AttachmentWorkPolicy.selectForRestoration(record.sessionId, listOf(record)))
    }

    @Test
    fun activityRecreationReconnectsRunningTranscription() {
        val record = record(AttachmentWorkOperation.WHISPER_TRANSCRIPTION, createdAtMillis = 10L)

        assertSame(record, AttachmentWorkPolicy.selectForRestoration(record.sessionId, listOf(record)))
    }

    @Test
    fun recreationCanReconnectToAlreadyCompletedWorkUntilActivityFinalizesIt() {
        val completedRecord = record(AttachmentWorkOperation.IMPORT, createdAtMillis = 20L)
        val unrelated = record(AttachmentWorkOperation.IMPORT, sessionId = "other", createdAtMillis = 30L)

        assertSame(
            completedRecord,
            AttachmentWorkPolicy.selectForRestoration(completedRecord.sessionId, listOf(unrelated, completedRecord))
        )
    }

    @Test
    fun duplicateWorkerPreventionReturnsExistingOperationIdentity() {
        val existing = record(AttachmentWorkOperation.WHISPER_TRANSCRIPTION)

        assertSame(
            existing,
            AttachmentWorkPolicy.findDuplicate(
                listOf(existing),
                existing.sessionId,
                existing.attachmentId,
                existing.operation
            )
        )
        assertNull(
            AttachmentWorkPolicy.findDuplicate(
                listOf(existing),
                existing.sessionId,
                existing.attachmentId,
                AttachmentWorkOperation.IMPORT
            )
        )
    }

    @Test
    fun retryAfterRecreationKeepsOwnedSourceAndAttachmentDirectoryIdentity() {
        val descriptor = descriptor(retryOperation = AttachmentWorkOperation.IMPORT)
        val failedAgain = AttachmentWorkPolicy.failedDescriptor(
            descriptor,
            AttachmentWorkOperation.IMPORT,
            "OCR failed",
            nowMillis = 50L
        )

        assertEquals(descriptor.id, failedAgain.id)
        assertEquals(descriptor.sourcePath, failedAgain.sourcePath)
        assertEquals(AttachmentWorkOperation.IMPORT, failedAgain.retryOperation)
        assertEquals("OCR failed", failedAgain.errorMessage)
    }

    @Test
    fun cancellationAfterRecreationPreservesReadableFailureAndCorrectRetryOperation() {
        val record = record(AttachmentWorkOperation.WHISPER_TRANSCRIPTION)
        val recovered = AttachmentWorkPolicy.selectForRestoration(null, listOf(record))!!
        val cancelled = AttachmentWorkPolicy.failedDescriptor(
            descriptor(),
            recovered.operation,
            "Processing cancelled. Retry or detach the attachment.",
            nowMillis = 100L
        )

        assertEquals(AttachmentStatus.FAILED, cancelled.status)
        assertEquals(AttachmentWorkOperation.WHISPER_TRANSCRIPTION, cancelled.retryOperation)
        assertTrue(cancelled.errorMessage!!.contains("cancelled"))
    }

    private fun record(
        operation: AttachmentWorkOperation,
        sessionId: String = "session",
        createdAtMillis: Long = 1L
    ) = AttachmentWorkRecord(
        workId = UUID.randomUUID().toString(),
        sessionId = sessionId,
        attachmentId = "attachment",
        operation = operation,
        requestedKind = if (operation == AttachmentWorkOperation.IMPORT) AttachmentKind.PDF else AttachmentKind.AUDIO,
        createdAtMillis = createdAtMillis
    )

    private fun descriptor(retryOperation: AttachmentWorkOperation? = null) = AttachmentDescriptor(
        id = "attachment",
        sessionId = "session",
        displayName = "sample.pdf",
        mimeType = "application/pdf",
        kind = AttachmentKind.PDF,
        status = AttachmentStatus.FAILED,
        processingRoute = AttachmentProcessingRoute.PDF_OCR,
        sourcePath = "chat_attachments/session/attachment/source.pdf",
        sizeBytes = 42L,
        retryOperation = retryOperation
    )
}
