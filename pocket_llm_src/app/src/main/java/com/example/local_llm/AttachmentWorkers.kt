package com.example.local_llm

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.net.Uri
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.OutOfQuotaPolicy
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.flow.first
import java.util.UUID

class AttachmentPreparationWorker(
    appContext: Context,
    parameters: WorkerParameters
) : CoroutineWorker(appContext, parameters) {
    override suspend fun doWork(): Result {
        setForeground(createForegroundInfo(applicationContext, "Preparing attachment…", id.hashCode()))
        val sessionId = inputData.getString(KEY_SESSION_ID) ?: return Result.failure()
        val uri = inputData.getString(KEY_URI)?.let(Uri::parse) ?: return Result.failure()
        val kind = inputData.getString(KEY_KIND)?.let { runCatching { AttachmentKind.valueOf(it) }.getOrNull() }
            ?: return Result.failure()
        return try {
            val descriptor = AttachmentImporter(applicationContext).import(
                uri = uri,
                sessionId = sessionId,
                requestedKind = kind,
                useGemmaNativeAudio = inputData.getBoolean(KEY_GEMMA_AUDIO, false)
            )
            Result.success(workDataOf(KEY_ATTACHMENT_ID to descriptor.id))
        } catch (error: AttachmentImportException) {
            Result.failure(
                workDataOf(
                    KEY_ATTACHMENT_ID to error.descriptor.id,
                    KEY_ERROR to error.message
                )
            )
        } catch (error: Throwable) {
            Result.failure(workDataOf(KEY_ERROR to (error.message ?: "Attachment preparation failed.")))
        }
    }

    companion object {
        const val KEY_SESSION_ID = "session_id"
        const val KEY_URI = "uri"
        const val KEY_KIND = "kind"
        const val KEY_GEMMA_AUDIO = "gemma_audio"
        const val KEY_ATTACHMENT_ID = "attachment_id"
        const val KEY_ERROR = "error"
    }
}

class WhisperTranscriptionWorker(
    appContext: Context,
    parameters: WorkerParameters
) : CoroutineWorker(appContext, parameters) {
    override suspend fun doWork(): Result {
        setForeground(createForegroundInfo(applicationContext, "Transcribing audio on device…", id.hashCode()))
        val sessionId = inputData.getString(AttachmentPreparationWorker.KEY_SESSION_ID) ?: return Result.failure()
        val attachmentId = inputData.getString(AttachmentPreparationWorker.KEY_ATTACHMENT_ID) ?: return Result.failure()
        val repository = AttachmentRepository(applicationContext)
        val descriptor = repository.loadDescriptor(sessionId, attachmentId) ?: return Result.failure()
        return try {
            val processing = descriptor.copy(
                status = AttachmentStatus.PROCESSING,
                updatedAtMillis = System.currentTimeMillis(),
                errorMessage = null
            )
            repository.saveDescriptor(processing)
            val existingChunks = repository.loadChunks(processing)
            val chunks = WhisperAttachmentTranscriber(applicationContext).transcribe(
                descriptor = processing,
                existingChunks = existingChunks,
                onCheckpoint = { checkpoint ->
                    repository.saveChunks(processing, checkpoint)
                    repository.saveBm25Index(processing, checkpoint)
                }
            )
            repository.saveChunks(processing, chunks)
            repository.saveBm25Index(processing, chunks)
            repository.saveExtractedText(
                processing,
                chunks.joinToString("\n\n") { "[${it.source.label()}]\n${it.text}" }
            )
            repository.saveDescriptor(
                processing.copy(
                    status = AttachmentStatus.READY,
                    updatedAtMillis = System.currentTimeMillis(),
                    extractedCharacters = chunks.sumOf { it.text.length }
                )
            )
            Result.success()
        } catch (error: Throwable) {
            repository.saveDescriptor(
                descriptor.copy(
                    status = AttachmentStatus.FAILED,
                    updatedAtMillis = System.currentTimeMillis(),
                    errorMessage = error.message ?: "Audio transcription failed."
                )
            )
            Result.failure(workDataOf(AttachmentPreparationWorker.KEY_ERROR to error.message))
        }
    }
}

class AttachmentWorkerCoordinator(context: Context) {
    private val workManager = WorkManager.getInstance(context.applicationContext)

    fun enqueueImport(
        sessionId: String,
        uri: Uri,
        kind: AttachmentKind,
        gemmaAudio: Boolean
    ): UUID {
        val request = OneTimeWorkRequestBuilder<AttachmentPreparationWorker>()
            .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
            .setInputData(
                workDataOf(
                    AttachmentPreparationWorker.KEY_SESSION_ID to sessionId,
                    AttachmentPreparationWorker.KEY_URI to uri.toString(),
                    AttachmentPreparationWorker.KEY_KIND to kind.name,
                    AttachmentPreparationWorker.KEY_GEMMA_AUDIO to gemmaAudio
                )
            )
            .build()
        workManager.enqueue(request)
        return request.id
    }

    fun enqueueWhisper(sessionId: String, attachmentId: String): UUID {
        val request = OneTimeWorkRequestBuilder<WhisperTranscriptionWorker>()
            .setInputData(
                workDataOf(
                    AttachmentPreparationWorker.KEY_SESSION_ID to sessionId,
                    AttachmentPreparationWorker.KEY_ATTACHMENT_ID to attachmentId
                )
            )
            .build()
        workManager.enqueue(request)
        return request.id
    }

    suspend fun await(id: UUID): WorkInfo {
        return requireNotNull(
            workManager.getWorkInfoByIdFlow(id).first { it?.state?.isFinished == true }
        )
    }

    fun cancel(id: UUID) {
        workManager.cancelWorkById(id)
    }
}

private fun createForegroundInfo(context: Context, text: String, notificationId: Int): ForegroundInfo {
    val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        manager.createNotificationChannel(
            NotificationChannel(
                ATTACHMENT_CHANNEL_ID,
                "Attachment processing",
                NotificationManager.IMPORTANCE_LOW
            )
        )
    }
    val notification = NotificationCompat.Builder(context, ATTACHMENT_CHANNEL_ID)
        .setSmallIcon(android.R.drawable.stat_sys_download)
        .setContentTitle("Pocket LLM")
        .setContentText(text)
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .build()
    return ForegroundInfo(notificationId, notification)
}

private const val ATTACHMENT_CHANNEL_ID = "attachment_processing"
