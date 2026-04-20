package com.example.local_llm

import android.content.Context
import android.net.Uri
import androidx.camera.core.ImageProxy
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
        fun onOcrStarted(source: Source)
        fun onOcrTextRecognized(text: String, source: Source)
        fun onOcrFailed(message: String, source: Source)
    }

    private val appContext = context.applicationContext
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    @Volatile
    private var cameraFrameInFlight = false

    fun recognizeImageUri(uri: Uri) {
        listener.onOcrStarted(Source.GALLERY)
        val image = runCatching {
            InputImage.fromFilePath(appContext, uri)
        }.getOrElse { error ->
            listener.onOcrFailed(error.message ?: "Could not read that image.", Source.GALLERY)
            return
        }

        recognizer.process(image)
            .addOnSuccessListener { result ->
                listener.onOcrTextRecognized(extractPlainText(result), Source.GALLERY)
            }
            .addOnFailureListener { error ->
                listener.onOcrFailed(error.message ?: "Could not read text from that image.", Source.GALLERY)
            }
    }

    fun recognizeImageProxy(imageProxy: ImageProxy) {
        if (cameraFrameInFlight) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        cameraFrameInFlight = true
        val image = InputImage.fromMediaImage(
            mediaImage,
            imageProxy.imageInfo.rotationDegrees
        )

        recognizer.process(image)
            .addOnSuccessListener { result ->
                listener.onOcrTextRecognized(extractPlainText(result), Source.CAMERA)
            }
            .addOnFailureListener { error ->
                listener.onOcrFailed(error.message ?: "Camera OCR failed.", Source.CAMERA)
            }
            .addOnCompleteListener {
                cameraFrameInFlight = false
                imageProxy.close()
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
