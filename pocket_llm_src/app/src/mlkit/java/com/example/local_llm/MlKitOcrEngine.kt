package com.example.local_llm

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** GitHub and Play builds: Google ML Kit Latin text recognition (model bundled in the APK). */
@Suppress("UNUSED_PARAMETER")
fun createOcrEngine(context: Context, statusListener: OcrStatusListener? = null): OcrEngine =
    MlKitOcrEngine(context)

internal class MlKitOcrEngine(context: Context) : OcrEngine {
    private val appContext = context.applicationContext
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    override suspend fun recognizeUri(uri: Uri): String {
        val image = InputImage.fromFilePath(appContext, uri)
        return extractStructuredText(recognizer.process(image).await())
    }

    override suspend fun recognizeBitmap(bitmap: Bitmap): String {
        return recognizer.process(InputImage.fromBitmap(bitmap, 0)).await().text
    }

    override fun close() {
        recognizer.close()
    }

    private fun extractStructuredText(text: Text): String {
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

        return lines.joinToString("\n").ifBlank { text.text }
    }

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { continuation ->
        addOnSuccessListener { value -> if (continuation.isActive) continuation.resume(value) }
        addOnFailureListener { error -> if (continuation.isActive) continuation.resumeWithException(error) }
        addOnCanceledListener { continuation.cancel() }
    }
}
