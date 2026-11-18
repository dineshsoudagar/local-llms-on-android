package com.example.local_llm

import android.graphics.Bitmap
import android.net.Uri

/**
 * On-device text recognition backend.
 *
 * Each distribution flavor provides a top-level `createOcrEngine(context, statusListener)`:
 * Google ML Kit in the Play and GitHub builds, Tesseract in the F-Droid build.
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
