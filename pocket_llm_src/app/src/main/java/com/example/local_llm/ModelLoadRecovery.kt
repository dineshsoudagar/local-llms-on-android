package com.example.local_llm

import android.content.Context
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

class ModelLoadCoordinator(initialRecord: ModelLoadRecord? = null) {
    var record: ModelLoadRecord? = initialRecord
        private set

    fun recoverInterrupted(nowMillis: Long): ModelLoadRecord? {
        val current = record ?: return null
        if (current.phase == ModelLoadPhase.INITIALIZING) {
            record = current.copy(
                phase = ModelLoadPhase.FAILED,
                timestampMillis = nowMillis,
                failureReason = "The app stopped while this model was loading. It will not be retried automatically."
            )
        }
        return record
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
        private const val KEY_MODEL_ID = "model_id"
        private const val KEY_ATTEMPT_ID = "attempt_id"
        private const val KEY_PHASE = "phase"
        private const val KEY_TIMESTAMP = "timestamp"
        private const val KEY_FAILURE_REASON = "failure_reason"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

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
