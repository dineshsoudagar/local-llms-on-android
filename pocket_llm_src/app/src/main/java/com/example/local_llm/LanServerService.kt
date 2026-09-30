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
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout

class LanServerService : Service() {
    companion object {
        private const val ACTION_START = "com.example.local_llm.action.START_LAN_SERVER"
        private const val ACTION_STOP = "com.example.local_llm.action.STOP_LAN_SERVER"
        private const val EXTRA_MODEL_ID = "com.example.local_llm.extra.LAN_MODEL_ID"
        private const val CHANNEL_ID = "lan_server"
        private const val NOTIFICATION_ID = 43
        private const val STARTUP_TIMEOUT_MS = 10 * 60 * 1000L

        @Volatile
        private var activeInProcess = false

        fun start(context: Context, modelId: String) {
            activeInProcess = true
            val intent = Intent(context, LanServerService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_MODEL_ID, modelId)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, LanServerService::class.java).setAction(ACTION_STOP)
            )
        }

        fun isRunning(): Boolean {
            return activeInProcess
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var startupJob: Job? = null
    private var server: LanHttpServer? = null
    private var preserveFailureState = false
    private var stopRequested = false
    private var serverWakeLock: PowerManager.WakeLock? = null
    private lateinit var notificationManager: NotificationManager

    override fun onCreate() {
        super.onCreate()
        activeInProcess = true
        notificationManager = getSystemService(NotificationManager::class.java)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopServer(startId)
            return START_NOT_STICKY
        }

        val state = LanServerStateStore.read(applicationContext)
        val modelId = intent?.getStringExtra(EXTRA_MODEL_ID) ?: state.modelId
        if (modelId.isNullOrBlank()) {
            LanServerStateStore.markFailed(
                applicationContext,
                modelId = null,
                message = getString(R.string.lan_server_model_required)
            )
            preserveFailureState = true
            stopSelf(startId)
            return START_NOT_STICKY
        }

        startServer(modelId, startId)
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        activeInProcess = false
        server?.close()
        server = null
        startupJob?.cancel()
        serviceScope.cancel()
        releaseServerWakeLock()
        if (!preserveFailureState && !stopRequested) {
            LanServerStateStore.markStopped(applicationContext)
        }
        super.onDestroy()
    }

    private fun startServer(modelId: String, startId: Int) {
        if (server != null || startupJob?.isActive == true) {
            return
        }

        if (!LanServerStateStore.hasPassword(applicationContext)) {
            LanServerStateStore.markFailed(
                applicationContext,
                modelId,
                getString(R.string.lan_server_password_required)
            )
            preserveFailureState = true
            stopSelf(startId)
            return
        }

        LanServerStateStore.ensureApiKey(applicationContext)
        LanServerStateStore.markStarting(applicationContext, modelId)
        startForegroundCompat()

        startupJob = serviceScope.launch {
            try {
                acquireServerWakeLock()
                ModelRegistry.loadCustomModels(applicationContext)
                val descriptor = ModelRegistry.findById(modelId)
                    ?: error(getString(R.string.lan_server_model_required))
                val existingController = LanServerControllerRegistry.peek(modelId)
                val controller = LanServerControllerRegistry.getOrCreate(applicationContext, descriptor)

                if (existingController == null || !controller.state.value.isReady) {
                    if (!controller.state.value.isLoading) {
                        controller.initialize()
                    }
                    withTimeout(STARTUP_TIMEOUT_MS) {
                        controller.state.first { !it.isLoading }
                    }
                }

                check(controller.state.value.isReady) {
                    controller.state.value.statusMessage.ifBlank {
                        getString(R.string.lan_server_model_not_ready)
                    }
                }

                val startedServer = LanHttpServer(
                    context = applicationContext,
                    controller = controller,
                    modelId = modelId,
                    passwordVerifier = { password ->
                        LanServerStateStore.verifyPassword(applicationContext, password)
                    },
                    apiKeyVerifier = { apiKey ->
                        LanServerStateStore.verifyApiKey(applicationContext, apiKey)
                    }
                )
                startedServer.start()
                server = startedServer
                LanServerStateStore.markRunning(
                    applicationContext,
                    modelId,
                    startedServer.endpoint
                )
                updateNotification(startedServer.endpoint)
            } catch (_: CancellationException) {
                LanServerStateStore.markStopped(applicationContext)
            } catch (error: Exception) {
                server?.close()
                server = null
                LanServerStateStore.markFailed(
                    applicationContext,
                    modelId,
                    error.message ?: getString(R.string.lan_server_start_failed)
                )
                preserveFailureState = true
                releaseServerWakeLock()
                stopForegroundCompat()
                stopSelf(startId)
            }
        }
    }

    private fun stopServer(startId: Int) {
        activeInProcess = false
        stopRequested = true
        startupJob?.cancel()
        startupJob = null
        server?.close()
        server = null
        releaseServerWakeLock()
        LanServerStateStore.markStopped(applicationContext)
        stopForegroundCompat()
        stopSelf(startId)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.lan_server_notification_channel),
                NotificationManager.IMPORTANCE_LOW
            )
        )
    }

    private fun acquireServerWakeLock() {
        if (serverWakeLock?.isHeld == true) return
        val powerManager = getSystemService(PowerManager::class.java)
        serverWakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$packageName:LanServer"
        ).apply {
            setReferenceCounted(false)
            // The user explicitly enables an always-available LAN server. Keep the CPU
            // awake until Stop, startup failure, or service destruction; never the display.
            acquire()
        }
    }

    private fun releaseServerWakeLock() {
        serverWakeLock?.takeIf { it.isHeld }?.release()
        serverWakeLock = null
    }

    private fun startForegroundCompat() {
        val notification = buildNotification(getString(R.string.lan_server_starting))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun updateNotification(endpoint: String) {
        notificationManager.notify(
            NOTIFICATION_ID,
            buildNotification(getString(R.string.lan_server_notification_running, endpoint))
        )
    }

    private fun buildNotification(contentText: String): Notification {
        val intent = Intent(this, PocketChatActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        val pendingIntent = PendingIntent.getActivity(
            this,
            2,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopIntent = PendingIntent.getService(
            this,
            3,
            Intent(this, LanServerService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher_2)
            .setContentTitle(getString(R.string.lan_server_notification_title))
            .setContentText(contentText)
            .setContentIntent(pendingIntent)
            .addAction(0, getString(R.string.lan_server_stop), stopIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        notificationManager.cancel(NOTIFICATION_ID)
    }
}
