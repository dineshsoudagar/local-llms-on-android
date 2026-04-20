package com.example.local_llm

import android.content.Context
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

class OcrInput(
    context: Context,
    private val listener: Listener
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

    private val appContext = context.applicationContext
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    fun recognizeImageUri(
        uri: Uri,
        source: Source = Source.GALLERY,
        requestId: Long = 0L
    ) {
        listener.onOcrStarted(source, requestId)
        val image = runCatching {
            InputImage.fromFilePath(appContext, uri)
        }.getOrElse { error ->
            listener.onOcrFailed(error.message ?: "Could not read that image.", source, requestId)
            return
        }

        recognizer.process(image)
            .addOnSuccessListener { result ->
                listener.onOcrTextRecognized(extractPlainText(result), source, requestId)
            }
            .addOnFailureListener { error ->
                listener.onOcrFailed(error.message ?: "Could not read text from that image.", source, requestId)
            }
    }

    fun close() {
        recognizer.close()
    }

    private fun extractPlainText(text: Text): String {
        val lines = buildList {
            text.textBlocks.forEachIndexed { blockIndex, block ->
                block.lines.forEach { line ->
                    val lineText = line.elements
                        .joinToString(" ") { element -> element.text }
                        .ifBlank { line.text }
                    if (lineText.isNotBlank()) {
                        add(lineText)
                    }
                }

                if (blockIndex < text.textBlocks.lastIndex && isNotEmpty() && last().isNotBlank()) {
                    add("")
                }
            }
        }

        val structuredText = lines.joinToString("\n")
        return PromptPreprocessor.normalize(structuredText.ifBlank { text.text })
    }
}
