package com.example.local_llm

import android.content.Context
import androidx.appcompat.app.AppCompatActivity

/**
 * Optional LAN server / OpenAI-compatible API. The implementation lives in the `src/lan` source
 * folder, which only the Pro build compiles; builds without it get [instance] == null and hide
 * the feature.
 */
interface LanServerFeature {
    enum class DrawerState { OFF, STARTING, RUNNING }

    /** What the LAN dialog needs from the chat screen. */
    interface Host {
        val activity: AppCompatActivity
        val currentModel: ModelDescriptor?
        val chatController: PersistentChatController?
        val proGate: ProGate
        fun showTransientMessage(message: String)
        fun resolveThemeColor(attrId: Int): Int
        fun updateKeepScreenOn()
        val installedModels: List<ModelDescriptor> get() = emptyList()
        fun selectModel(descriptor: ModelDescriptor) = Unit
        fun openManageModels() = Unit
    }

    fun isRunning(): Boolean
    fun drawerState(context: Context): DrawerState
    fun showDialog(host: Host)

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.LanServerFeatureImpl"

        val instance: LanServerFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as LanServerFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null

        fun isRunning(): Boolean = instance?.isRunning() == true
    }
}
