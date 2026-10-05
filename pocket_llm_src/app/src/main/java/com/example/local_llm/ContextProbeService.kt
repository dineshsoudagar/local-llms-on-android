package com.example.local_llm

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.Process
import kotlinx.coroutines.runBlocking

/**
 * Runs one context-test step in its own process (see the manifest), so a native out-of-memory
 * crash at a size that is too large ends only this process and the app keeps running.
 */
class ContextProbeService : Service() {
    companion object {
        const val MSG_PROBE = 1
        const val MSG_RESULT = 2
        const val KEY_MODEL_ID = "model_id"
        const val KEY_CONTEXT = "context"
        const val KEY_PASSED = "passed"
        const val KEY_FILLED = "filled"
        const val KEY_ERROR = "error"
        const val PROCESS_SUFFIX = ":context_probe"
        private const val REPLY_FLUSH_MILLIS = 300L
    }

    private val worker = HandlerThread("context-probe").apply { start() }
    private val messenger = Messenger(Handler(worker.looper) { message ->
        if (message.what == MSG_PROBE) {
            runProbe(message.data, message.replyTo)
            true
        } else {
            false
        }
    })

    override fun onBind(intent: Intent): IBinder = messenger.binder

    private fun runProbe(request: Bundle, replyTo: Messenger?) {
        val contextTokens = request.getInt(KEY_CONTEXT)
        val result = Bundle().apply { putInt(KEY_CONTEXT, contextTokens) }
        try {
            ModelRegistry.loadCustomModels(this)
            val modelId = requireNotNull(request.getString(KEY_MODEL_ID)) { "No model was given." }
            val descriptor = requireNotNull(ModelRegistry.findById(modelId)) { "Unknown model $modelId." }
            val settings = ModelRuntimeSettings(contextTokens)
            // The test measures the GPU path the chat uses; a CPU fallback would hide the limit.
            val policy = BackendInitializationPolicy(allowCpuFallback = false)
            val resolver = ModelFileResolver(this)
            val backend: ChatBackend = when (descriptor) {
                is GemmaLiteRtSpec, is CustomLiteRtSpec -> GemmaLiteRtBackend(this, descriptor, resolver, settings, policy)
                is QwenLiteRtSpec -> QwenLiteRtBackend(this, descriptor, resolver, settings, policy)
                is OnnxQwenSpec -> throw IllegalArgumentException("ONNX models use a fixed context.")
            }
            backend.use {
                runBlocking { it.initialize() }
                val filler = ContextProbePlanner.fillerText(ContextProbePlanner.fillTarget(contextTokens))
                result.putInt(KEY_FILLED, it.probeContextFill(filler))
                result.putBoolean(KEY_PASSED, true)
            }
        } catch (error: Throwable) {
            result.putBoolean(KEY_PASSED, false)
            result.putString(KEY_ERROR, error.message ?: error.javaClass.simpleName)
        }
        runCatching { replyTo?.send(Message.obtain(null, MSG_RESULT).apply { data = result }) }
        // Exit so no native or GPU memory carries over into the next, larger step.
        Thread.sleep(REPLY_FLUSH_MILLIS)
        Process.killProcess(Process.myPid())
    }
}
