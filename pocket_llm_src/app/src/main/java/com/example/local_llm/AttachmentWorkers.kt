package com.example.local_llm

import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.OutOfQuotaPolicy
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.Flow
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
        val attachmentId = inputData.getString(KEY_ATTACHMENT_ID) ?: return Result.failure()
        return try {
            val importer = AttachmentImporter(applicationContext)
            val descriptor = if (inputData.getBoolean(KEY_RETRY_OWNED_SOURCE, false)) {
                val existing = AttachmentRepository(applicationContext)
                    .loadDescriptor(sessionId, attachmentId)
                    ?: return Result.failure(workDataOf(KEY_ERROR to "The attachment retry record is missing."))
                importer.retry(existing)
            } else {
                importer.import(
                    uri = uri,
                    sessionId = sessionId,
                    requestedKind = kind,
                    useGemmaNativeAudio = inputData.getBoolean(KEY_GEMMA_AUDIO, false),
                    attachmentId = attachmentId
                )
            }
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
        } finally {
            deleteRecordedAttachmentInput(applicationContext, uri)
        }
    }

    companion object {
        const val KEY_SESSION_ID = "session_id"
        const val KEY_URI = "uri"
        const val KEY_KIND = "kind"
        const val KEY_GEMMA_AUDIO = "gemma_audio"
        const val KEY_ATTACHMENT_ID = "attachment_id"
        const val KEY_ERROR = "error"
        const val KEY_RETRY_OWNED_SOURCE = "retry_owned_source"
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
                errorMessage = null,
                retryOperation = AttachmentWorkOperation.WHISPER_TRANSCRIPTION
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
                    extractedCharacters = chunks.sumOf { it.text.length },
                    retryOperation = null,
                    errorMessage = null
                )
            )
            Result.success()
        } catch (error: Throwable) {
            repository.saveDescriptor(
                descriptor.copy(
                    status = AttachmentStatus.FAILED,
                    updatedAtMillis = System.currentTimeMillis(),
                    errorMessage = error.message ?: "Audio transcription failed.",
                    retryOperation = AttachmentWorkOperation.WHISPER_TRANSCRIPTION
                )
            )
            Result.failure(
                workDataOf(
                    AttachmentPreparationWorker.KEY_ERROR to (error.message ?: "Audio transcription failed.")
                )
            )
        }
    }
}

class AttachmentWorkerCoordinator(context: Context) {
    private val workManager = WorkManager.getInstance(context.applicationContext)
    private val repository = AttachmentRepository(context.applicationContext)

    fun enqueueImport(
        sessionId: String,
        uri: Uri,
        kind: AttachmentKind,
        gemmaAudio: Boolean
    ): AttachmentWorkRecord {
        val attachmentId = UUID.randomUUID().toString()
        return enqueueImportWork(sessionId, attachmentId, uri, kind, gemmaAudio, retryOwnedSource = false)
    }

    fun enqueueImportRetry(descriptor: AttachmentDescriptor): AttachmentWorkRecord {
        return enqueueImportWork(
            sessionId = descriptor.sessionId,
            attachmentId = descriptor.id,
            uri = Uri.fromFile(repository.sourceFile(descriptor)),
            kind = descriptor.kind,
            gemmaAudio = descriptor.useGemmaNativeAudio,
            retryOwnedSource = true
        )
    }

    private fun enqueueImportWork(
        sessionId: String,
        attachmentId: String,
        uri: Uri,
        kind: AttachmentKind,
        gemmaAudio: Boolean,
        retryOwnedSource: Boolean
    ): AttachmentWorkRecord {
        existingRecord(sessionId, attachmentId, AttachmentWorkOperation.IMPORT)?.let { return it }
        val request = OneTimeWorkRequestBuilder<AttachmentPreparationWorker>()
            .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
            .setInputData(
                workDataOf(
                    AttachmentPreparationWorker.KEY_SESSION_ID to sessionId,
                    AttachmentPreparationWorker.KEY_URI to uri.toString(),
                    AttachmentPreparationWorker.KEY_KIND to kind.name,
                    AttachmentPreparationWorker.KEY_GEMMA_AUDIO to gemmaAudio,
                    AttachmentPreparationWorker.KEY_ATTACHMENT_ID to attachmentId,
                    AttachmentPreparationWorker.KEY_RETRY_OWNED_SOURCE to retryOwnedSource
                )
            )
            .addTag(sessionTag(sessionId))
            .addTag(attachmentTag(attachmentId))
            .addTag(operationTag(AttachmentWorkOperation.IMPORT))
            .build()
        val record = AttachmentWorkRecord(
            workId = request.id.toString(),
            sessionId = sessionId,
            attachmentId = attachmentId,
            operation = AttachmentWorkOperation.IMPORT,
            requestedKind = kind,
            useGemmaNativeAudio = gemmaAudio
        )
        enqueue(record, request)
        return record
    }

    fun enqueueWhisper(sessionId: String, attachmentId: String): AttachmentWorkRecord {
        existingRecord(sessionId, attachmentId, AttachmentWorkOperation.WHISPER_TRANSCRIPTION)?.let { return it }
        val request = OneTimeWorkRequestBuilder<WhisperTranscriptionWorker>()
            .setInputData(
                workDataOf(
                    AttachmentPreparationWorker.KEY_SESSION_ID to sessionId,
                    AttachmentPreparationWorker.KEY_ATTACHMENT_ID to attachmentId
                )
            )
            .addTag(sessionTag(sessionId))
            .addTag(attachmentTag(attachmentId))
            .addTag(operationTag(AttachmentWorkOperation.WHISPER_TRANSCRIPTION))
            .build()
        val descriptor = repository.loadDescriptor(sessionId, attachmentId)
        val record = AttachmentWorkRecord(
            workId = request.id.toString(),
            sessionId = sessionId,
            attachmentId = attachmentId,
            operation = AttachmentWorkOperation.WHISPER_TRANSCRIPTION,
            requestedKind = descriptor?.kind ?: AttachmentKind.AUDIO,
            useGemmaNativeAudio = descriptor?.useGemmaNativeAudio ?: false
        )
        enqueue(record, request)
        return record
    }

    private fun enqueue(record: AttachmentWorkRecord, request: androidx.work.OneTimeWorkRequest) {
        repository.saveWorkRecord(record)
        try {
            workManager.enqueueUniqueWork(uniqueWorkName(record), ExistingWorkPolicy.KEEP, request)
        } catch (error: Throwable) {
            repository.deleteWorkRecord(record)
            throw error
        }
    }

    private fun existingRecord(
        sessionId: String,
        attachmentId: String,
        operation: AttachmentWorkOperation
    ): AttachmentWorkRecord? = AttachmentWorkPolicy.findDuplicate(
        repository.listWorkRecords(sessionId),
        sessionId,
        attachmentId,
        operation
    )

    fun workInfoFlow(id: UUID): Flow<WorkInfo?> = workManager.getWorkInfoByIdFlow(id)

    suspend fun await(id: UUID): WorkInfo {
        return requireNotNull(
            workManager.getWorkInfoByIdFlow(id).first { it?.state?.isFinished == true }
        )
    }

    fun cancel(id: UUID) {
        workManager.cancelWorkById(id)
    }

    companion object {
        internal fun uniqueWorkName(record: AttachmentWorkRecord): String =
            "attachment:${record.sessionId}:${record.attachmentId}:${record.operation.name}"

        private fun sessionTag(sessionId: String) = "attachment-session:$sessionId"
        private fun attachmentTag(attachmentId: String) = "attachment-id:$attachmentId"
        private fun operationTag(operation: AttachmentWorkOperation) = "attachment-operation:${operation.name}"
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
    return ForegroundInfo(notificationId, notification, ATTACHMENT_FOREGROUND_SERVICE_TYPE)
}

@SuppressLint("InlinedApi")
internal const val ATTACHMENT_FOREGROUND_SERVICE_TYPE = ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
private const val ATTACHMENT_CHANNEL_ID = "attachment_processing"

private fun deleteRecordedAttachmentInput(context: Context, uri: Uri) {
    if (uri.scheme != "file") return
    val file = uri.path?.let { java.io.File(it) } ?: return
    val recordingsDirectory = java.io.File(context.cacheDir, "attachment_recordings")
    val safe = runCatching {
        file.canonicalPath.startsWith(recordingsDirectory.canonicalPath + java.io.File.separator)
    }.getOrDefault(false)
    if (safe) file.delete()
}
