package com.example.local_llm

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.text.format.Formatter
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class ModelDownloadService : Service() {

    companion object {
        private const val ACTION_CANCEL_DOWNLOAD = "com.example.local_llm.action.CANCEL_MODEL_DOWNLOAD"
        private const val EXTRA_MODEL_ID = "com.example.local_llm.extra.MODEL_ID"
        private const val CHANNEL_ID = "model_downloads"
        private const val NOTIFICATION_ID = 42
        private const val WAKE_LOCK_TIMEOUT_MS = 12 * 60 * 60 * 1000L

        fun start(context: Context, descriptor: ModelDescriptor) {
            val intent = Intent(context, ModelDownloadService::class.java)
                .putExtra(EXTRA_MODEL_ID, descriptor.id)
            ContextCompat.startForegroundService(context, intent)
        }

        fun cancel(context: Context) {
            val intent = Intent(context, ModelDownloadService::class.java)
                .setAction(ACTION_CANCEL_DOWNLOAD)
            context.startService(intent)
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private lateinit var modelFileResolver: ModelFileResolver
    private lateinit var modelDownloader: ModelDownloader
    private lateinit var notificationManager: NotificationManager
    private var downloadJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        modelFileResolver = ModelFileResolver(applicationContext)
        modelDownloader = ModelDownloader(modelFileResolver)
        notificationManager = getSystemService(NotificationManager::class.java)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL_DOWNLOAD) {
            cancelActiveDownload(startId)
            return START_NOT_STICKY
        }

        val descriptor = ModelRegistry.findById(intent?.getStringExtra(EXTRA_MODEL_ID))
        if (descriptor == null) {
            stopSelf(startId)
            return START_NOT_STICKY
        }

        if (downloadJob?.isActive == true) {
            return START_REDELIVER_INTENT
        }

        beginDownload(descriptor, startId)
        return START_REDELIVER_INTENT
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        downloadJob?.cancel()
        serviceScope.cancel()
        releaseWakeLock()
        super.onDestroy()
    }

    private fun cancelActiveDownload(startId: Int) {
        if (downloadJob?.isActive == true) {
            downloadJob?.cancel(CancellationException("Model download cancelled."))
            return
        }

        if (ModelDownloadStateStore.state.value is ModelDownloadState.Running) {
            ModelDownloadStateStore.update(ModelDownloadState.Idle)
        }
        removeForegroundNotification()
        releaseWakeLock()
        stopSelf(startId)
    }

    private fun beginDownload(descriptor: ModelDescriptor, startId: Int) {
        val initialState = ModelDownloadState.Running(
            modelId = descriptor.id,
            modelName = descriptor.displayName,
            fileName = null,
            bytesDownloaded = 0L,
            totalBytes = descriptor.approxDownloadBytes.takeIf { it > 0L }
        )
        ModelDownloadStateStore.update(initialState)
        startForegroundCompat(initialState)
        acquireWakeLock()

        downloadJob = serviceScope.launch {
            val fileTotals = mutableMapOf<String, Long>()
            var completedBytes = 0L
            var currentFileName: String? = null
            var currentFileBytes = 0L

            try {
                modelDownloader.downloadModel(descriptor) { progress ->
                    if (currentFileName != null && currentFileName != progress.fileName) {
                        completedBytes += fileTotals[currentFileName] ?: currentFileBytes
                        currentFileBytes = 0L
                    }

                    currentFileName = progress.fileName
                    currentFileBytes = progress.bytesDownloaded
                    progress.totalBytes?.let { fileTotals[progress.fileName] = it }

                    val aggregateTotalBytes = aggregateTotalBytes(descriptor, fileTotals)
                    val aggregateDownloadedBytes = (completedBytes + progress.bytesDownloaded)
                        .let { downloaded ->
                            if (aggregateTotalBytes != null) {
                                downloaded.coerceAtMost(aggregateTotalBytes)
                            } else {
                                downloaded
                            }
                        }

                    val runningState = ModelDownloadState.Running(
                        modelId = descriptor.id,
                        modelName = descriptor.displayName,
                        fileName = progress.fileName,
                        bytesDownloaded = aggregateDownloadedBytes,
                        totalBytes = aggregateTotalBytes
                    )
                    ModelDownloadStateStore.update(runningState)
                    notificationManager.notify(NOTIFICATION_ID, buildRunningNotification(runningState))
                }

                ModelSelectionStore(applicationContext).saveSelectedModel(descriptor.id)
                val completedState = ModelDownloadState.Completed(
                    modelId = descriptor.id,
                    modelName = descriptor.displayName
                )
                ModelDownloadStateStore.update(completedState)
                showTerminalNotification(buildCompletedNotification(completedState))
            } catch (_: CancellationException) {
                if (ModelDownloadStateStore.state.value is ModelDownloadState.Running) {
                    ModelDownloadStateStore.update(ModelDownloadState.Idle)
                }
                removeForegroundNotification()
            } catch (error: Exception) {
                val failedState = ModelDownloadState.Failed(
                    modelId = descriptor.id,
                    modelName = descriptor.displayName,
                    message = error.message ?: getString(R.string.model_download_failed_generic)
                )
                ModelDownloadStateStore.update(failedState)
                showTerminalNotification(buildFailedNotification(failedState))
            } finally {
                releaseWakeLock()
                downloadJob = null
                stopSelf(startId)
            }
        }
    }

    private fun aggregateTotalBytes(
        descriptor: ModelDescriptor,
        fileTotals: Map<String, Long>
    ): Long? {
        val allTotalsKnown = descriptor.downloadFiles.all { fileTotals.containsKey(it.localFileName) }
        if (allTotalsKnown) {
            return descriptor.downloadFiles.sumOf { fileTotals[it.localFileName] ?: 0L }
                .takeIf { it > 0L }
        }

        return descriptor.approxDownloadBytes.takeIf { it > 0L }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }

        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.download_notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            setShowBadge(false)
        }
        notificationManager.createNotificationChannel(channel)
    }

    private fun startForegroundCompat(state: ModelDownloadState.Running) {
        val notification = buildRunningNotification(state)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildRunningNotification(state: ModelDownloadState.Running): Notification {
        val totalBytes = state.totalBytes
        val contentText = if (totalBytes != null && totalBytes > 0L) {
            getString(
                R.string.download_notification_progress,
                formatFileSize(state.bytesDownloaded),
                formatFileSize(totalBytes)
            )
        } else {
            getString(
                R.string.download_notification_progress_unknown,
                formatFileSize(state.bytesDownloaded)
            )
        }

        val builder = baseNotificationBuilder()
            .setContentTitle(getString(R.string.download_notification_title, state.modelName))
            .setContentText(contentText)
            .setOngoing(true)
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                getString(R.string.cancel_download),
                cancelDownloadPendingIntent()
            )

        if (totalBytes != null && totalBytes > 0L) {
            builder.setProgress(
                1000,
                ((state.bytesDownloaded * 1000L) / totalBytes).toInt().coerceIn(0, 1000),
                false
            )
        } else {
            builder.setProgress(0, 0, true)
        }

        return builder.build()
    }

    private fun buildCompletedNotification(state: ModelDownloadState.Completed): Notification {
        return baseNotificationBuilder()
            .setContentTitle(getString(R.string.download_notification_complete_title))
            .setContentText(getString(R.string.download_notification_complete, state.modelName))
            .setOngoing(false)
            .setAutoCancel(true)
            .build()
    }

    private fun buildFailedNotification(state: ModelDownloadState.Failed): Notification {
        return baseNotificationBuilder()
            .setContentTitle(getString(R.string.download_notification_failed_title))
            .setContentText(getString(R.string.download_notification_failed, state.modelName))
            .setStyle(NotificationCompat.BigTextStyle().bigText(state.message))
            .setOngoing(false)
            .setAutoCancel(true)
            .build()
    }

    private fun baseNotificationBuilder(): NotificationCompat.Builder {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher_2)
            .setContentIntent(openAppPendingIntent())
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
    }

    private fun openAppPendingIntent(): PendingIntent {
        val intent = Intent(this, PocketChatActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        return PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun cancelDownloadPendingIntent(): PendingIntent {
        val intent = Intent(this, ModelDownloadService::class.java)
            .setAction(ACTION_CANCEL_DOWNLOAD)
        return PendingIntent.getService(
            this,
            1,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun showTerminalNotification(notification: Notification) {
        detachForegroundNotification()
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun detachForegroundNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_DETACH)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(false)
        }
    }

    private fun removeForegroundNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        notificationManager.cancel(NOTIFICATION_ID)
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) {
            return
        }

        val powerManager = getSystemService(PowerManager::class.java)
        wakeLock = powerManager
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:ModelDownload")
            .apply {
                setReferenceCounted(false)
                acquire(WAKE_LOCK_TIMEOUT_MS)
            }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    private fun formatFileSize(sizeBytes: Long): String {
        return Formatter.formatFileSize(this, sizeBytes)
    }
}
