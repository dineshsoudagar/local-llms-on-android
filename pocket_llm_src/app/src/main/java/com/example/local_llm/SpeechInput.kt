package com.example.local_llm

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale

class SpeechInput(
    context: Context,
    private val listener: Listener
) {
    interface Listener {
        fun onSpeechStarted()
        fun onSpeechPartial(text: String)
        fun onSpeechFinal(text: String, confidenceScores: FloatArray?)
        fun onSpeechError(message: String)
        fun onSpeechEnded()
    }

    private val appContext = context.applicationContext
    private var recognizer: SpeechRecognizer? = null
    var isListening: Boolean = false
        private set

    fun start() {
        if (isListening) {
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(appContext)) {
            listener.onSpeechError("Speech recognition is not available on this device.")
            return
        }

        destroyRecognizer()
        val nextRecognizer = SpeechRecognizer.createSpeechRecognizer(appContext)
        recognizer = nextRecognizer
        nextRecognizer.setRecognitionListener(createRecognitionListener())

        runCatching {
            isListening = true
            listener.onSpeechStarted()
            nextRecognizer.startListening(createRecognizerIntent())
        }.onFailure { error ->
            isListening = false
            listener.onSpeechError(error.message ?: "Could not start speech recognition.")
            destroyRecognizer()
        }
    }

    fun stop() {
        recognizer?.stopListening()
    }

    fun cancel() {
        recognizer?.cancel()
        finishListening()
    }

    fun destroy() {
        destroyRecognizer()
        isListening = false
    }

    private fun createRecognizerIntent(): Intent {
        return Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toLanguageTag())

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                putExtra(
                    RecognizerIntent.EXTRA_ENABLE_FORMATTING,
                    RecognizerIntent.FORMATTING_OPTIMIZE_LATENCY
                )
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                putExtra(RecognizerIntent.EXTRA_REQUEST_WORD_CONFIDENCE, true)
                putExtra(RecognizerIntent.EXTRA_REQUEST_WORD_TIMING, true)
            }
        }
    }

    private fun createRecognitionListener(): RecognitionListener {
        return object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) = Unit

            override fun onBeginningOfSpeech() = Unit

            override fun onRmsChanged(rmsdB: Float) = Unit

            override fun onBufferReceived(buffer: ByteArray?) = Unit

            override fun onEndOfSpeech() {
                finishListening()
            }

            override fun onError(error: Int) {
                finishListening()
                listener.onSpeechError(messageForError(error))
            }

            override fun onResults(results: Bundle?) {
                publishResults(results, final = true)
                finishListening()
            }

            override fun onPartialResults(partialResults: Bundle?) {
                publishResults(partialResults, final = false)
            }

            override fun onEvent(eventType: Int, params: Bundle?) = Unit
        }
    }

    private fun publishResults(results: Bundle?, final: Boolean) {
        val text = results
            ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            ?.firstOrNull()
            .orEmpty()

        if (text.isBlank()) {
            return
        }

        if (final) {
            listener.onSpeechFinal(
                text,
                results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)
            )
        } else {
            listener.onSpeechPartial(text)
        }
    }

    private fun finishListening() {
        if (!isListening) {
            return
        }
        isListening = false
        listener.onSpeechEnded()
    }

    private fun destroyRecognizer() {
        recognizer?.setRecognitionListener(null)
        recognizer?.destroy()
        recognizer = null
    }

    private fun messageForError(error: Int): String {
        return when (error) {
            SpeechRecognizer.ERROR_AUDIO -> "Audio capture failed."
            SpeechRecognizer.ERROR_CLIENT -> "Speech recognition client error."
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission is required."
            SpeechRecognizer.ERROR_NETWORK -> "Speech recognition network error."
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Speech recognition timed out."
            SpeechRecognizer.ERROR_NO_MATCH -> "No speech was recognized."
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Speech recognition is already running."
            SpeechRecognizer.ERROR_SERVER -> "Speech recognition service error."
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech was detected."
            else -> "Speech recognition failed."
        }
    }
}
