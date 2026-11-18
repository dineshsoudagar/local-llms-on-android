package com.example.local_llm

import android.content.Context
import android.content.Intent
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import java.io.File

interface ImageGenFeature {
    interface ChatHost {
        val activity: AppCompatActivity
        val chatController: PersistentChatController?
        val inputEditText: android.widget.EditText
        val sendButton: android.widget.Button
        val stopButton: android.widget.Button
        val attachmentButton: com.google.android.material.button.MaterialButton
        val galleryOcrButton: android.widget.Button
        val cameraOcrButton: android.widget.Button
        val createImageModeButton: com.google.android.material.button.MaterialButton
        val imageModelChip: com.google.android.material.button.MaterialButton
        val refinePromptButton: com.google.android.material.button.MaterialButton
        val webSearchButton: com.google.android.material.button.MaterialButton
        val imageModeStatus: android.widget.TextView
        val imageModePanel: View
        val proGate: ProGate
        val modelFileResolver: ModelFileResolver
        fun requireProOrTrial(feature: ProFeature, action: () -> Unit)
        fun showTransientMessage(message: String)
        fun resolveThemeColor(attrId: Int): Int
        fun updateWebSearchButton()
        fun applyChatState()
        fun setPromptInputText(text: String)
        fun openImageViewer(path: String)
        fun showManageModelsDialog()
        fun startModelDownload(spec: ModelDescriptor)
        fun refreshProBadges()
        fun proLabel(resId: Int, feature: ProFeature): CharSequence
        fun launchImageGen(intent: Intent)
        fun addDownloadableModelOption(descriptor: ModelDescriptor, onUseModel: () -> Unit)
        fun dismissModelDialog()
    }

    fun setupImageMode(host: ChatHost)
    fun applyImageModeComposer(host: ChatHost)
    fun handleCreateImageSend(host: ChatHost)
    fun refineImagePrompt(host: ChatHost)
    fun reportGeneratedImage(host: ChatHost, turn: ChatTurn)
    fun openImageGeneration(host: ChatHost, specId: String? = null, toolName: String? = null, photoPath: String? = null)
    fun showImageToolsSheet(host: ChatHost)
    fun handleImageModelSelection(host: ChatHost, specId: String)
    fun addDownloadableModelOptions(host: ChatHost)
    fun isImageModeActive(): Boolean
    fun setImageMode(host: ChatHost, active: Boolean)
    fun cancel()
    fun isBusy(): Boolean
    fun clearResult()
    fun release()
    fun showImageModelMenu(host: ChatHost, anchor: View)
    fun enterImageMode(host: ChatHost)

    fun isResultManageModels(resultCode: Int): Boolean
    fun isResultUseInChat(resultCode: Int): Boolean
    fun getResultPath(data: Intent?): String?

    fun loadCustomModels(context: Context)
    fun findModelById(id: String?): ModelDescriptor?
    fun allModels(): List<ModelDescriptor>

    suspend fun removeBackground(context: Context, file: File, onProgress: (String) -> Unit): SavedImageEntry
    fun isCpuSupported(): Boolean
    fun showOutOfTries(activity: AppCompatActivity, gate: ProGate, onUnlocked: () -> Unit)

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.ImageGenFeatureImpl"

        val instance: ImageGenFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as ImageGenFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null
    }
}
