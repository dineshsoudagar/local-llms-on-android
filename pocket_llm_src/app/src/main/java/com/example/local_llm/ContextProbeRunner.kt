package com.example.local_llm

import android.app.ActivityManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process

data class ContextProbeStep(val contextTokens: Int, val passed: Boolean, val filledTokens: Int?, val error: String?)

data class ContextProbeResult(
    val steps: List<ContextProbeStep>,
    val largestPassedTokens: Int?,
    val recommendedTokens: Int?,
    val cancelled: Boolean
)

/**
 * Drives the context test from the app process: each size runs in [ContextProbeService]'s
 * process, and that process dying counts as a failed size.
 */
class ContextProbeRunner(
    context: Context,
    private val descriptor: ModelDescriptor,
    private val onStepStarted: (contextTokens: Int, steps: List<ContextProbeStep>) -> Unit,
    private val onFilling: (contextTokens: Int, filledTokens: Int, targetTokens: Int) -> Unit,
    private val onFinished: (ContextProbeResult) -> Unit,
    private val startTokens: Int? = null
) {
    private companion object {
        const val STEP_TIMEOUT_MILLIS = 5L * 60L * 1000L
        const val BETWEEN_STEPS_MILLIS = 1_500L
        const val CRASH_MESSAGE = "The test process stopped, most likely out of memory."
        const val TIMEOUT_MESSAGE = "The step took too long."
    }

    private val appContext = context.applicationContext
    private val handler = Handler(Looper.getMainLooper())
    private val steps = mutableListOf<ContextProbeStep>()
    private val maxTokens = ModelRuntimeSettingsLimits.maxFor(descriptor)
    private var currentSize: Int? = null
    private var bound = false
    private var finished = false

    private val replyMessenger = Messenger(Handler(Looper.getMainLooper()) { message ->
        if (message.what == ContextProbeService.MSG_PROGRESS) {
            val data = message.data
            val size = data.getInt(ContextProbeService.KEY_CONTEXT)
            if (size == currentSize) {
                onFilling(size, data.getInt(ContextProbeService.KEY_FILLED), data.getInt(ContextProbeService.KEY_TARGET))
            }
            true
        } else if (message.what == ContextProbeService.MSG_RESULT) {
            val data = message.data
            val size = data.getInt(ContextProbeService.KEY_CONTEXT)
            if (size == currentSize) {
                completeStep(
                    ContextProbeStep(
                        contextTokens = size,
                        passed = data.getBoolean(ContextProbeService.KEY_PASSED),
                        filledTokens = data.getInt(ContextProbeService.KEY_FILLED, 0).takeIf { it > 0 },
                        error = data.getString(ContextProbeService.KEY_ERROR)
                    )
                )
            }
            true
        } else {
            false
        }
    })

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, service: IBinder) {
            val size = currentSize ?: return
            val request = Message.obtain(null, ContextProbeService.MSG_PROBE).apply {
                data.putString(ContextProbeService.KEY_MODEL_ID, descriptor.id)
                data.putInt(ContextProbeService.KEY_CONTEXT, size)
                replyTo = replyMessenger
            }
            runCatching { Messenger(service).send(request) }
                .onFailure { failCurrent(CRASH_MESSAGE) }
        }

        override fun onServiceDisconnected(name: ComponentName) {
            failCurrent(CRASH_MESSAGE)
        }

        override fun onBindingDied(name: ComponentName) {
            failCurrent(CRASH_MESSAGE)
        }
    }

    private val timeout = Runnable {
        failCurrent(TIMEOUT_MESSAGE)
        killProbeProcess()
    }

    fun start() {
        runNextStep()
    }

    fun cancel() {
        if (finished) return
        currentSize = null
        cleanUpStep()
        killProbeProcess()
        finish(cancelled = true)
    }

    private fun runNextStep() {
        if (finished) return
        val size = ContextProbePlanner.nextSize(
            passed = steps.filter { it.passed }.map { it.contextTokens },
            failed = steps.filterNot { it.passed }.map { it.contextTokens },
            maxTokens = maxTokens,
            startTokens = startTokens
        )
        if (size == null) {
            finish(cancelled = false)
            return
        }
        currentSize = size
        onStepStarted(size, steps.toList())
        handler.postDelayed(timeout, STEP_TIMEOUT_MILLIS)
        bound = appContext.bindService(
            Intent(appContext, ContextProbeService::class.java),
            connection,
            Context.BIND_AUTO_CREATE
        )
        if (!bound) failCurrent("Could not start the test process.")
    }

    private fun failCurrent(message: String) {
        val size = currentSize ?: return
        completeStep(ContextProbeStep(size, passed = false, filledTokens = null, error = message))
    }

    private fun completeStep(step: ContextProbeStep) {
        if (currentSize != step.contextTokens) return
        currentSize = null
        steps += step
        cleanUpStep()
        // Give the probe process time to exit and return its memory before the next size.
        handler.postDelayed({ runNextStep() }, BETWEEN_STEPS_MILLIS)
    }

    private fun cleanUpStep() {
        handler.removeCallbacks(timeout)
        if (bound) {
            runCatching { appContext.unbindService(connection) }
            bound = false
        }
    }

    private fun finish(cancelled: Boolean) {
        if (finished) return
        finished = true
        handler.removeCallbacksAndMessages(null)
        val largest = steps.filter { it.passed }.maxOfOrNull { it.contextTokens }
        onFinished(
            ContextProbeResult(
                steps = steps.toList(),
                largestPassedTokens = largest,
                recommendedTokens = ContextProbePlanner.recommended(largest),
                cancelled = cancelled
            )
        )
    }

    private fun killProbeProcess() {
        val manager = appContext.getSystemService(ActivityManager::class.java) ?: return
        manager.runningAppProcesses
            ?.filter { it.processName.endsWith(ContextProbeService.PROCESS_SUFFIX) }
            ?.forEach { Process.killProcess(it.pid) }
    }
}
