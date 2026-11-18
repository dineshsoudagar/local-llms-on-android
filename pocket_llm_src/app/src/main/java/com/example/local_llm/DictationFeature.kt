package com.example.local_llm

import android.content.Context

/**
 * Optional dictation with the loaded model's own audio input. The implementation lives in the
 * `src/dictation` source folder, which only the Pro build compiles; builds without it get
 * [instance] == null and always dictate with Whisper.
 */
interface DictationFeature {
    /** Transcribes a normalized recording with the model behind [controller]. */
    suspend fun transcribe(
        context: Context,
        controller: PersistentChatController,
        recording: AttachmentDescriptor,
        segmentMillis: Long
    ): String

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.DictationFeatureImpl"

        val instance: DictationFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as DictationFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null
    }
}
