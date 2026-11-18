package com.example.local_llm

import android.content.Context
import androidx.appcompat.app.AppCompatActivity

interface WebSearchFeature {
    interface Host {
        val activity: AppCompatActivity
        val chatController: PersistentChatController?
        fun updateWebSearchButton()
        fun requireProOrTrial(feature: ProFeature, action: () -> Unit)
        fun resolveThemeColor(attrId: Int): Int
    }

    interface SourcesState {
        fun collect(call: ExternalToolCall, result: String)
        fun markdown(answer: String): String
    }

    fun isUsable(context: Context): Boolean
    fun setUsable(context: Context, usable: Boolean)

    val toolDefinitions: List<ExternalToolDefinition>
    val systemInstruction: String
    val incompleteText: String
    val maxRounds: Int

    fun onWebSearchChipClicked(host: Host)
    fun showManageDialog(host: Host)
    fun statusForTool(call: ExternalToolCall): String
    fun executeTool(context: Context, call: ExternalToolCall): String
    fun createSourcesState(): SourcesState

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.WebSearchFeatureImpl"

        val instance: WebSearchFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as WebSearchFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null
    }
}
