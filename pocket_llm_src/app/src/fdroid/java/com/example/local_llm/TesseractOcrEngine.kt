package com.example.local_llm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import com.googlecode.tesseract.android.TessBaseAPI
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.IOException

/** F-Droid build: open-source Tesseract OCR. English data is downloaded on first use. */
fun createOcrEngine(context: Context, statusListener: OcrStatusListener? = null): OcrEngine =
    TesseractOcrEngine(context, statusListener)

internal class TesseractOcrEngine(
    context: Context,
    private val statusListener: OcrStatusListener?
) : OcrEngine {
    private val appContext = context.applicationContext
    private val recognitionMutex = Mutex()
    private val apiLock = Any()
    private var api: TessBaseAPI? = null
    private var busy = false
    private var closed = false

    override suspend fun recognizeUri(uri: Uri): String {
        val bitmap = withContext(Dispatchers.IO) { decodeUri(uri) }
        return try {
            recognize(bitmap)
        } finally {
            bitmap.recycle()
        }
    }

    override suspend fun recognizeBitmap(bitmap: Bitmap): String = recognize(bitmap)

    override fun close() {
        synchronized(apiLock) {
            closed = true
            if (!busy) releaseApiLocked()
        }
    }

    private suspend fun recognize(bitmap: Bitmap): String {
        val dataPath = TesseractLanguageData.ensureInstalled(appContext, statusListener)
        return recognitionMutex.withLock {
            withContext(Dispatchers.Default) {
                val tess = acquireApi(dataPath)
                try {
                    val opaque = flattenOntoWhite(bitmap)
                    try {
                        tess.setImage(opaque)
                        tess.getUTF8Text().orEmpty()
                    } finally {
                        tess.clear()
                        if (opaque !== bitmap) opaque.recycle()
                    }
                } finally {
                    synchronized(apiLock) {
                        busy = false
                        if (closed) releaseApiLocked()
                    }
                }
            }
        }
    }

    private fun acquireApi(dataPath: String): TessBaseAPI = synchronized(apiLock) {
        check(!closed) { "Text recognition was closed." }
        val existing = api
        val tess = if (existing != null) {
            existing
        } else {
            val created = TessBaseAPI()
            val initialized = runCatching {
                created.init(dataPath, TesseractLanguageData.LANGUAGE, TessBaseAPI.OEM_LSTM_ONLY)
            }.getOrDefault(false)
            if (!initialized) {
                created.recycle()
                TesseractLanguageData.discardInstallation(appContext)
                throw IOException("The OCR language data could not be loaded. Try again to download it.")
            }
            created.setPageSegMode(TessBaseAPI.PageSegMode.PSM_AUTO)
            api = created
            created
        }
        busy = true
        tess
    }

    private fun releaseApiLocked() {
        api?.recycle()
        api = null
    }

    private fun decodeUri(uri: Uri): Bitmap {
        val resolver = appContext.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val boundsStream = resolver.openInputStream(uri) ?: throw IOException("Could not read that image.")
        boundsStream.use { input -> BitmapFactory.decodeStream(input, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IOException("Could not read that image.")
        }

        var sampleSize = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sampleSize * 2) >= MAX_IMAGE_DIMENSION) {
            sampleSize *= 2
        }
        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = resolver.openInputStream(uri)?.use { input ->
            BitmapFactory.decodeStream(input, null, options)
        } ?: throw IOException("Could not read that image.")

        val orientation = runCatching {
            resolver.openInputStream(uri)?.use { input ->
                ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            }
        }.getOrNull() ?: ExifInterface.ORIENTATION_NORMAL
        return applyOrientation(decoded, orientation)
    }

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.preScale(-1f, 1f)
                matrix.postRotate(90f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.preScale(-1f, 1f)
                matrix.postRotate(-90f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(-90f)
            else -> return bitmap
        }
        val rotated = runCatching {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }.getOrElse { bitmap }
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    /**
     * Tesseract ignores alpha, so transparent pixels (for example the background of a
     * rendered PDF page) would read as black. Draw the image onto white first.
     */
    private fun flattenOntoWhite(bitmap: Bitmap): Bitmap {
        if (!bitmap.hasAlpha() && bitmap.config == Bitmap.Config.ARGB_8888) return bitmap
        val output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Canvas(output).apply {
            drawColor(Color.WHITE)
            drawBitmap(bitmap, 0f, 0f, null)
        }
        return output
    }

    private companion object {
        const val MAX_IMAGE_DIMENSION = 3_000
    }
}
