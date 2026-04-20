package com.example.local_llm

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
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
    private val restartHandler = Handler(Looper.getMainLooper())
    private var recognizer: SpeechRecognizer? = null
    private var suppressNextTerminalError = false
    var isRecording: Boolean = false
        private set
    var isListening: Boolean = false
        private set

    fun start() {
        if (isRecording) {
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(appContext)) {
            listener.onSpeechError("Speech recognition is not available on this device.")
            return
        }

        isRecording = true
        listener.onSpeechStarted()
        startRecognizer()
    }

    fun stop() {
        if (!isRecording) {
            return
        }

        isRecording = false
        restartHandler.removeCallbacksAndMessages(null)
        suppressNextTerminalError = true
        if (isListening) {
            recognizer?.stopListening()
        } else {
            destroyRecognizer()
            suppressNextTerminalError = false
            listener.onSpeechEnded()
        }
    }

    fun cancel() {
        isRecording = false
        restartHandler.removeCallbacksAndMessages(null)
        suppressNextTerminalError = true
        recognizer?.cancel()
        finishListening()
        destroyRecognizer()
        suppressNextTerminalError = false
    }

    fun destroy() {
        isRecording = false
        restartHandler.removeCallbacksAndMessages(null)
        destroyRecognizer()
        isListening = false
    }

    private fun startRecognizer() {
        if (!isRecording) {
            return
        }

        destroyRecognizer()
        suppressNextTerminalError = false
        val nextRecognizer = SpeechRecognizer.createSpeechRecognizer(appContext)
        recognizer = nextRecognizer
        nextRecognizer.setRecognitionListener(createRecognitionListener())

        runCatching {
            isListening = true
            nextRecognizer.startListening(createRecognizerIntent())
        }.onFailure { error ->
            isRecording = false
            isListening = false
            listener.onSpeechError(error.message ?: "Could not start speech recognition.")
            listener.onSpeechEnded()
            destroyRecognizer()
        }
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
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 60_000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 60_000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 60_000L)

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

            override fun onEndOfSpeech() = Unit

            override fun onError(error: Int) {
                finishListening()
                destroyRecognizer()

                if (isRecording && shouldRestartAfterError(error)) {
                    scheduleRestart()
                    return
                }

                val shouldSuppressError = suppressNextTerminalError || !isRecording
                if (!shouldSuppressError) {
                    listener.onSpeechError(messageForError(error))
                }
                isRecording = false
                suppressNextTerminalError = false
                listener.onSpeechEnded()
            }

            override fun onResults(results: Bundle?) {
                publishResults(results, final = true)
                finishListening()
                destroyRecognizer()

                if (isRecording) {
                    scheduleRestart()
                } else {
                    suppressNextTerminalError = false
                    listener.onSpeechEnded()
                }
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

    private fun scheduleRestart() {
        restartHandler.removeCallbacksAndMessages(null)
        restartHandler.postDelayed(
            {
                if (isRecording && !isListening) {
                    startRecognizer()
                }
            },
            250L
        )
    }

    private fun shouldRestartAfterError(error: Int): Boolean {
        return when (error) {
            SpeechRecognizer.ERROR_NO_MATCH,
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT,
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT,
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> true
            else -> false
        }
    }

    private fun finishListening() {
        if (!isListening) {
            return
        }
        isListening = false
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
