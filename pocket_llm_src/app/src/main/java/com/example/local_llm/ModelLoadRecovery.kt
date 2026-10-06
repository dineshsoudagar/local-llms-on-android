package com.example.local_llm

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import androidx.annotation.RequiresApi
import java.util.UUID

enum class ModelLoadPhase {
    INITIALIZING,
    FAILED
}

data class ModelLoadRecord(
    val modelId: String,
    val attemptId: String,
    val phase: ModelLoadPhase,
    val timestampMillis: Long,
    val failureReason: String? = null
)

/** Why the process died while a model load was in flight, as reported by Android. */
enum class InterruptedLoadCause {
    NATIVE_CRASH,
    LOW_MEMORY,
    UNKNOWN
}

class ModelLoadCoordinator(initialRecord: ModelLoadRecord? = null) {
    var record: ModelLoadRecord? = initialRecord
        private set

    fun recoverInterrupted(
        nowMillis: Long,
        cause: InterruptedLoadCause = InterruptedLoadCause.UNKNOWN
    ): ModelLoadRecord? {
        val current = record ?: return null
        if (current.phase == ModelLoadPhase.INITIALIZING) {
            record = current.copy(
                phase = ModelLoadPhase.FAILED,
                timestampMillis = nowMillis,
                failureReason = interruptedReason(cause)
            )
        }
        return record
    }

    private fun interruptedReason(cause: InterruptedLoadCause): String = when (cause) {
        InterruptedLoadCause.NATIVE_CRASH ->
            "The app stopped while this model was loading because the on-device runtime crashed. " +
                "GPU acceleration will be skipped for this model on the next attempt."
        InterruptedLoadCause.LOW_MEMORY ->
            "The app stopped while this model was loading because Android ran out of memory. " +
                "Try a smaller model, a shorter context length, or closing other apps."
        InterruptedLoadCause.UNKNOWN ->
            "The app stopped while this model was loading. It will not be retried automatically."
    }

    fun shouldAutoLoad(modelId: String): Boolean {
        val current = record ?: return true
        return current.modelId != modelId || current.phase != ModelLoadPhase.FAILED
    }

    fun begin(modelId: String, nowMillis: Long, attemptId: String = UUID.randomUUID().toString()): ModelLoadRecord {
        return ModelLoadRecord(
            modelId = modelId,
            attemptId = attemptId,
            phase = ModelLoadPhase.INITIALIZING,
            timestampMillis = nowMillis
        ).also { record = it }
    }

    fun succeed(modelId: String, attemptId: String): Boolean {
        if (!matchesInitializing(modelId, attemptId)) return false
        record = null
        return true
    }

    fun fail(modelId: String, attemptId: String, reason: String, nowMillis: Long): Boolean {
        if (!matchesInitializing(modelId, attemptId)) return false
        record = record!!.copy(
            phase = ModelLoadPhase.FAILED,
            timestampMillis = nowMillis,
            failureReason = reason
        )
        return true
    }

    fun cancel(modelId: String, attemptId: String): Boolean {
        if (!matchesInitializing(modelId, attemptId)) return false
        record = null
        return true
    }

    fun clearForModel(modelId: String): Boolean {
        if (record?.modelId != modelId) return false
        record = null
        return true
    }

    fun replace(record: ModelLoadRecord?) {
        this.record = record
    }

    private fun matchesInitializing(modelId: String, attemptId: String): Boolean {
        val current = record ?: return false
        return current.modelId == modelId &&
            current.attemptId == attemptId &&
            current.phase == ModelLoadPhase.INITIALIZING
    }
}

class ModelLoadRecoveryStore(context: Context) {
    companion object {
        private const val PREFS_NAME = "pocket_chat_model_load_recovery"
        private const val GPU_SAFETY_PREFS_NAME = "pocket_chat_gpu_safety"
        private const val GPU_UNSAFE_PREFIX = "gpu_unsafe_"
        private const val KEY_MODEL_ID = "model_id"
        private const val KEY_ATTEMPT_ID = "attempt_id"
        private const val KEY_PHASE = "phase"
        private const val KEY_TIMESTAMP = "timestamp"
        private const val KEY_FAILURE_REASON = "failure_reason"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    // Kept separate so clearing the load record does not re-enable a GPU path that crashed.
    private val gpuSafetyPrefs = context.getSharedPreferences(GPU_SAFETY_PREFS_NAME, Context.MODE_PRIVATE)
    private val runtimeIdentity = "${Build.FINGERPRINT}/${BuildConfig.VERSION_CODE}"

    /** A native crash during load disables GPU for this model until the OS or app is updated. */
    fun markGpuUnsafe(modelId: String) {
        gpuSafetyPrefs.edit().putString(GPU_UNSAFE_PREFIX + modelId, runtimeIdentity).commit()
    }

    fun isGpuUnsafe(modelId: String): Boolean {
        return gpuSafetyPrefs.getString(GPU_UNSAFE_PREFIX + modelId, null) == runtimeIdentity
    }

    fun load(): ModelLoadRecord? {
        val modelId = prefs.getString(KEY_MODEL_ID, null) ?: return null
        val attemptId = prefs.getString(KEY_ATTEMPT_ID, null) ?: return null
        val phase = prefs.getString(KEY_PHASE, null)
            ?.let { runCatching { ModelLoadPhase.valueOf(it) }.getOrNull() }
            ?: return null
        return ModelLoadRecord(
            modelId = modelId,
            attemptId = attemptId,
            phase = phase,
            timestampMillis = prefs.getLong(KEY_TIMESTAMP, 0L),
            failureReason = prefs.getString(KEY_FAILURE_REASON, null)
        )
    }

    fun save(record: ModelLoadRecord?) {
        if (record == null) {
            prefs.edit().clear().commit()
            return
        }
        prefs.edit()
            .putString(KEY_MODEL_ID, record.modelId)
            .putString(KEY_ATTEMPT_ID, record.attemptId)
            .putString(KEY_PHASE, record.phase.name)
            .putLong(KEY_TIMESTAMP, record.timestampMillis)
            .putString(KEY_FAILURE_REASON, record.failureReason)
            .commit()
    }
}

object ProcessExitInspector {
    /** Reads Android's record of the previous process death (API 30+) to explain an interrupted load. */
    fun interruptedLoadCause(context: Context, loadStartedAtMillis: Long): InterruptedLoadCause {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return InterruptedLoadCause.UNKNOWN
        return runCatching { lastExitCause(context, loadStartedAtMillis) }
            .getOrDefault(InterruptedLoadCause.UNKNOWN)
    }

    @RequiresApi(Build.VERSION_CODES.R)
    private fun lastExitCause(context: Context, loadStartedAtMillis: Long): InterruptedLoadCause {
        val exitInfo = context.getSystemService(ActivityManager::class.java)
            ?.getHistoricalProcessExitReasons(context.packageName, 0, 1)
            ?.firstOrNull()
            ?: return InterruptedLoadCause.UNKNOWN
        if (exitInfo.timestamp < loadStartedAtMillis) return InterruptedLoadCause.UNKNOWN
        return when (exitInfo.reason) {
            ApplicationExitInfo.REASON_CRASH_NATIVE -> InterruptedLoadCause.NATIVE_CRASH
            ApplicationExitInfo.REASON_LOW_MEMORY -> InterruptedLoadCause.LOW_MEMORY
            else -> InterruptedLoadCause.UNKNOWN
        }
    }
}
