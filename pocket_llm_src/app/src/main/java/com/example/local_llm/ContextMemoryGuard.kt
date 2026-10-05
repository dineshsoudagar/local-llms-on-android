package com.example.local_llm

import android.app.ActivityManager
import android.content.Context
import java.util.UUID

/**
 * Applies [ContextMemoryBudget] on the device and learns from native crashes. A marker is written
 * before each LiteRT load and each high-context generation; if the process dies before the marker
 * is cleared, the next load for that model uses a smaller context.
 */
class ContextMemoryGuard(context: Context) {
    companion object {
        private const val PREFS_NAME = "context_memory_guard"
        private const val LEARNED_LIMIT_SUFFIX = "_learned_context_limit"
        private const val CAP_NOTICE_SUFFIX = "_context_cap_notice"
        private const val MEASURED_LIMIT_SUFFIX = "_measured_context_limit"
        private const val KEY_RUN_MODEL_ID = "run_model_id"
        private const val KEY_RUN_CONTEXT = "run_context"
        private const val KEY_RUN_PROCESS = "run_process"

        // A marker left by another process means that process died during the run.
        private val PROCESS_TOKEN = UUID.randomUUID().toString()
    }

    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun decide(descriptor: ModelDescriptor, requestedTokens: Int): ContextDecision {
        if (descriptor is OnnxQwenSpec) {
            return ContextDecision(requestedTokens, requestedTokens, requestedTokens, null)
        }
        recoverInterruptedRun()
        val memoryInfo = ActivityManager.MemoryInfo()
        appContext.getSystemService(ActivityManager::class.java).getMemoryInfo(memoryInfo)
        return ContextMemoryBudget.decide(
            requestedTokens = requestedTokens,
            modelBytes = descriptor.approxDownloadBytes,
            totalMemoryBytes = memoryInfo.totalMem,
            learnedLimitTokens = learnedLimit(descriptor.id),
            measuredLimitTokens = measuredLimit(descriptor.id)
        )
    }

    fun learnedLimit(modelId: String): Int? =
        prefs.getInt(modelId + LEARNED_LIMIT_SUFFIX, 0).takeIf { it > 0 }

    fun measuredLimit(modelId: String): Int? =
        prefs.getInt(modelId + MEASURED_LIMIT_SUFFIX, 0).takeIf { it > 0 }

    /** Stores the context test's recommendation; it supersedes the estimate and earlier crash history. */
    fun saveMeasuredLimit(modelId: String, tokens: Int) {
        prefs.edit()
            .putInt(modelId + MEASURED_LIMIT_SUFFIX, tokens)
            .remove(modelId + LEARNED_LIMIT_SUFFIX)
            .commit()
    }

    /** The user chose a new context explicitly, so earlier crash history no longer applies. */
    fun clearLearnedLimit(modelId: String) {
        prefs.edit().remove(modelId + LEARNED_LIMIT_SUFFIX).apply()
    }

    /** Returns true the first time a given cap is seen for [modelId], so the notice is not repeated on every load. */
    fun markCapNoticeShown(modelId: String, decision: ContextDecision): Boolean {
        val key = modelId + CAP_NOTICE_SUFFIX
        val signature = "${decision.requestedTokens}->${decision.effectiveTokens}"
        if (prefs.getString(key, null) == signature) return false
        prefs.edit().putString(key, signature).apply()
        return true
    }

    fun beginRun(modelId: String, contextTokens: Int) {
        // commit(): the marker must be on disk before native code can take the process down.
        prefs.edit()
            .putString(KEY_RUN_MODEL_ID, modelId)
            .putInt(KEY_RUN_CONTEXT, contextTokens)
            .putString(KEY_RUN_PROCESS, PROCESS_TOKEN)
            .commit()
    }

    fun endRun() {
        prefs.edit()
            .remove(KEY_RUN_MODEL_ID)
            .remove(KEY_RUN_CONTEXT)
            .remove(KEY_RUN_PROCESS)
            .apply()
    }

    inline fun <T> tracked(modelId: String, contextTokens: Int, block: () -> T): T {
        beginRun(modelId, contextTokens)
        try {
            return block()
        } finally {
            endRun()
        }
    }

    private fun recoverInterruptedRun() {
        val process = prefs.getString(KEY_RUN_PROCESS, null) ?: return
        if (process == PROCESS_TOKEN) return
        val modelId = prefs.getString(KEY_RUN_MODEL_ID, null)
        val contextTokens = prefs.getInt(KEY_RUN_CONTEXT, 0)
        val editor = prefs.edit()
            .remove(KEY_RUN_MODEL_ID)
            .remove(KEY_RUN_CONTEXT)
            .remove(KEY_RUN_PROCESS)
        if (modelId != null && contextTokens > 0) {
            editor.putInt(
                modelId + LEARNED_LIMIT_SUFFIX,
                ContextMemoryBudget.limitAfterCrash(contextTokens, learnedLimit(modelId))
            )
        }
        editor.commit()
    }
}
