package com.example.local_llm

import android.graphics.Bitmap
import android.net.Uri

/**
 * On-device text recognition backend.
 *
 * Each distribution flavor provides a top-level `createOcrEngine(context, statusListener)`:
 * the GitHub and Play builds use Google ML Kit (src/mlkit), the F-Droid build uses
 * Tesseract (src/fdroid), so no proprietary library ends up in the F-Droid APK.
 */
interface OcrEngine {
    /**
     * Reads text from an image URI (file or content). Returns line-structured text with
     * blank lines between text blocks. The caller normalizes it.
     */
    suspend fun recognizeUri(uri: Uri): String

    /** Reads text from an already decoded bitmap, such as a rendered PDF page. */
    suspend fun recognizeBitmap(bitmap: Bitmap): String

    fun close()
}

/** Receives short, user-facing status messages, such as OCR data download progress. */
fun interface OcrStatusListener {
    fun onOcrStatus(message: String)
}
