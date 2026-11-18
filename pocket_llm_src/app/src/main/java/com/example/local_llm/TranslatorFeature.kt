package com.example.local_llm

import android.content.Context
import android.content.Intent

/** Offline translator (Pro). Lives in src/translator; builds without that folder have no translator. */
interface TranslatorFeature {
    /** Opens the translator, optionally with text to translate or an image to read first. */
    fun createIntent(context: Context, text: String? = null, imagePath: String? = null): Intent

    /** The translation the user wants to continue with in the chat, or null. */
    fun resultChatText(resultCode: Int, data: Intent?): String?

    /**
     * The chat screen owns model loading (checks, crash recovery, the open chat). The translator
     * asks it to load a model instead of copying that logic.
     */
    interface ChatHost {
        /**
         * Starts loading [descriptor] as the chat model and keeps the open chat. Returns null when
         * the load started, otherwise a message for the user saying why not and what to do.
         */
        fun loadModel(descriptor: ModelDescriptor): String?

        /** Why the load started by [loadModel] failed, or null while it runs or after it worked. */
        fun loadFailure(modelId: String): String?
    }

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.TranslatorFeatureImpl"

        val instance: TranslatorFeature? by lazy {
            runCatching {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as TranslatorFeature
            }.getOrNull()
        }

        val isAvailable: Boolean get() = instance != null

        /** Set by the chat screen while it exists. */
        @Volatile
        var chatHost: ChatHost? = null
    }
}

/**
 * Text or an image another app shared with this one. The share screen (src/translator) asks what
 * to do with it and passes the answer to the chat screen in these extras.
 */
object SharedContent {
    const val EXTRA_ACTION = "com.example.local_llm.extra.SHARE_ACTION"
    const val EXTRA_TEXT = "com.example.local_llm.extra.SHARE_TEXT"
    const val EXTRA_IMAGE_PATH = "com.example.local_llm.extra.SHARE_IMAGE_PATH"
    const val ACTION_TRANSLATE = "translate"
    const val ACTION_NEW_CHAT = "new_chat"
    const val ACTION_ADD_TO_CHAT = "add_to_chat"

    /**
     * The translator asks the chat screen to load a model when the chat screen is gone (Android
     * closed it in the background). [EXTRA_TEXT] holds the model id; the chat loads it and reopens
     * the translator with [EXTRA_PENDING_MODEL] so the translator shows the load.
     */
    const val ACTION_LOAD_MODEL = "load_model"
    const val EXTRA_PENDING_MODEL = "com.example.local_llm.extra.PENDING_MODEL"
}
