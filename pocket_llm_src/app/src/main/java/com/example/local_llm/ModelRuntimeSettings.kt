package com.example.local_llm

import android.content.Context

data class ModelRuntimeSettings(
    val contextLengthTokens: Int
)

object ModelRuntimeSettingsLimits {
    const val ONNX_CONTEXT_LENGTH = 512
    const val LITERT_DEFAULT_CONTEXT_LENGTH = 2_048
    const val LITERT_CONTEXT_LENGTH = 32_000

    fun defaultFor(descriptor: ModelDescriptor): Int = if (descriptor is OnnxQwenSpec) ONNX_CONTEXT_LENGTH else LITERT_DEFAULT_CONTEXT_LENGTH

    fun minFor(descriptor: ModelDescriptor): Int {
        return if (descriptor is OnnxQwenSpec) 256 else 512
    }

    fun maxFor(descriptor: ModelDescriptor): Int {
        return if (descriptor is OnnxQwenSpec) ONNX_CONTEXT_LENGTH else LITERT_CONTEXT_LENGTH
    }

    fun normalize(descriptor: ModelDescriptor, contextLengthTokens: Int): Int {
        return contextLengthTokens.coerceIn(minFor(descriptor), maxFor(descriptor))
    }
}

class ModelRuntimeSettingsStore(context: Context) {
    companion object {
        private const val PREFS_NAME = "model_runtime_settings"
        private const val CONTEXT_LENGTH_SUFFIX = "_context_length"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(descriptor: ModelDescriptor): ModelRuntimeSettings {
        val defaultLength = ModelRuntimeSettingsLimits.defaultFor(descriptor)
        val savedLength = prefs.getInt(contextLengthKey(descriptor), defaultLength)
        return ModelRuntimeSettings(
            contextLengthTokens = ModelRuntimeSettingsLimits.normalize(descriptor, savedLength)
        )
    }

    fun save(descriptor: ModelDescriptor, settings: ModelRuntimeSettings): ModelRuntimeSettings {
        val normalized = settings.copy(
            contextLengthTokens = ModelRuntimeSettingsLimits.normalize(
                descriptor,
                settings.contextLengthTokens
            )
        )
        prefs.edit()
            .putInt(contextLengthKey(descriptor), normalized.contextLengthTokens)
            .apply()
        return normalized
    }

    private fun contextLengthKey(descriptor: ModelDescriptor): String {
        return "${descriptor.id}$CONTEXT_LENGTH_SUFFIX"
    }
}
