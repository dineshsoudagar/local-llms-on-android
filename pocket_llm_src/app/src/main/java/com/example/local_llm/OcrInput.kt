package com.example.local_llm

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class OcrInput(
    context: Context,
    private val listener: Listener? = null,
    statusListener: OcrStatusListener? = null
) {
    enum class Source {
        GALLERY,
        CAMERA
    }

    interface Listener {
        fun onOcrStarted(source: Source, requestId: Long)
        fun onOcrTextRecognized(text: String, source: Source, requestId: Long)
        fun onOcrFailed(message: String, source: Source, requestId: Long)
    }

    private val engine: OcrEngine = createOcrEngine(context.applicationContext, statusListener)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    fun recognizeImageUri(
        uri: Uri,
        source: Source = Source.GALLERY,
        requestId: Long = 0L
    ) {
        listener?.onOcrStarted(source, requestId)
        scope.launch {
            try {
                val text = recognizeImageUriText(uri)
                listener?.onOcrTextRecognized(text, source, requestId)
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                listener?.onOcrFailed(error.message ?: "Could not read text from that image.", source, requestId)
            }
        }
    }

    suspend fun recognizeImageUriText(uri: Uri): String {
        return PromptPreprocessor.normalize(engine.recognizeUri(uri))
    }

    fun close() {
        scope.cancel()
        engine.close()
    }
}
