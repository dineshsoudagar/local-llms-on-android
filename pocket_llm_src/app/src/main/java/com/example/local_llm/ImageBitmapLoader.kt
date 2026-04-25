package com.example.local_llm

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import java.io.File

object ImageBitmapLoader {

    fun decodeSampledBitmap(
        imageFile: File,
        requestedWidth: Int,
        requestedHeight: Int
    ): Bitmap? {
        if (!imageFile.exists()) {
            return null
        }

        val bounds = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(imageFile.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            return null
        }

        val options = BitmapFactory.Options().apply {
            inSampleSize = calculateSampleSize(bounds, requestedWidth, requestedHeight)
        }
        val decodedBitmap = BitmapFactory.decodeFile(imageFile.absolutePath, options) ?: return null
        return applyExifOrientation(decodedBitmap, imageFile)
    }

    fun applyExifOrientation(
        bitmap: Bitmap,
        imageFile: File?
    ): Bitmap {
        if (imageFile == null || !imageFile.exists()) {
            return bitmap
        }

        val orientation = runCatching {
            ExifInterface(imageFile.absolutePath)
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.preScale(1f, -1f)
            }
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

        return runCatching {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }.getOrElse {
            bitmap
        }.also { rotated ->
            if (rotated !== bitmap) {
                bitmap.recycle()
            }
        }
    }

    private fun calculateSampleSize(
        bounds: BitmapFactory.Options,
        requestedWidth: Int,
        requestedHeight: Int
    ): Int {
        var sampleSize = 1
        val safeWidth = requestedWidth.coerceAtLeast(1)
        val safeHeight = requestedHeight.coerceAtLeast(1)
        var halfWidth = bounds.outWidth / 2
        var halfHeight = bounds.outHeight / 2

        while (
            halfWidth / sampleSize >= safeWidth &&
            halfHeight / sampleSize >= safeHeight
        ) {
            sampleSize *= 2
        }

        return sampleSize.coerceAtLeast(1)
    }
}
