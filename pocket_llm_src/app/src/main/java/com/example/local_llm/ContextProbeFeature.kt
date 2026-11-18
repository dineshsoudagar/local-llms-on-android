package com.example.local_llm

import androidx.appcompat.app.AppCompatActivity

interface ContextProbeFeature {
    interface Host {
        val activity: AppCompatActivity
        fun showTransientMessage(message: String)
        fun resolveThemeColor(attrId: Int): Int
        fun updateKeepScreenOn()
        val isModelPreparing: Boolean
        val isModelLoading: Boolean
        fun unloadModelForProbe(message: String): () -> Unit
    }

    fun isRunning(): Boolean
    fun onDestroy()
    fun onContextTestButtonClicked(host: Host, descriptor: ModelDescriptor, dismissDialog: () -> Unit)

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.ContextProbeFeatureImpl"

        val instance: ContextProbeFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as ContextProbeFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null
    }
}
