package com.example.local_llm

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity

interface LlamaFeature {
    interface Host {
        val activity: AppCompatActivity
        fun showModelImportFailure(error: Throwable)
        fun onModelImported(id: String, isChat: Boolean, displayName: String)
    }

    fun createBackend(
        context: Context,
        spec: GgufSpec,
        modelFileResolver: ModelFileResolver,
        runtimeSettings: ModelRuntimeSettings
    ): ChatBackend

    fun loadCustomModels(context: Context): List<ModelDescriptor>

    fun handleLocalImport(host: Host, uri: Uri, fileName: String)

    fun launchHfBrowser(context: Context, launcher: androidx.activity.result.ActivityResultLauncher<Intent>)

    companion object {
        const val LOCAL_REPO = "local"
        private const val IMPLEMENTATION = "com.example.local_llm.LlamaFeatureImpl"

        val instance: LlamaFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as LlamaFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null
    }
}
