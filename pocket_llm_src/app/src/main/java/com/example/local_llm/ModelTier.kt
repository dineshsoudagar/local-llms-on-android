package com.example.local_llm

import android.app.ActivityManager
import android.content.Context

/** Plain-language tier shown next to a built-in model's name. Models without a tier keep their raw name. */
enum class ModelTier(val labelRes: Int, val blurbRes: Int) {
    FAST(R.string.tier_fast, R.string.tier_fast_blurb),
    BALANCED(R.string.tier_balanced, R.string.tier_balanced_blurb),
    BEST(R.string.tier_best, R.string.tier_best_blurb),
    REASONING(R.string.tier_reasoning, R.string.tier_reasoning_blurb);

    companion object {
        fun of(model: ModelDescriptor): ModelTier? = when (model.id) {
            "qwen3_litert" -> FAST
            "gemma4_e2b" -> BALANCED
            "gemma4_e4b" -> BEST
            "deepseek_r1_distill_qwen_1_5b_litert" -> REASONING
            else -> null
        }

        /** The chat tier that fits this phone's memory. Reasoning is never the default pick. */
        fun recommendedFor(context: Context): ModelTier {
            val info = ActivityManager.MemoryInfo()
            (context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager)?.getMemoryInfo(info)
            val gib = info.totalMem / (1024.0 * 1024.0 * 1024.0)
            return when {
                gib >= 10.5 -> BEST
                gib >= 6.5 -> BALANCED
                else -> FAST
            }
        }
    }
}

/** Display name without the engine suffix; the engine moves into the details row. */
fun ModelDescriptor.shortName(): String =
    displayName.removeSuffix(" LiteRT").removeSuffix(" ONNX").removeSuffix(" GGUF")
