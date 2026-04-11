package com.example.local_llm

import android.content.Context
import java.io.File
import java.io.FileOutputStream

class ModelFileResolver(private val context: Context) {

    fun resolveAssetToFile(assetPath: String): File {
        val targetDir = File(context.filesDir, "models").apply { mkdirs() }
        val targetFile = File(targetDir, assetPath.substringAfterLast('/'))
        if (targetFile.exists() && targetFile.length() > 0L) {
            return targetFile
        }

        val tempFile = File(targetFile.absolutePath + ".tmp")
        tempFile.parentFile?.mkdirs()

        context.assets.open(assetPath).use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        }

        if (targetFile.exists()) {
            targetFile.delete()
        }
        check(tempFile.renameTo(targetFile)) {
            "Failed to move copied asset into place: ${targetFile.absolutePath}"
        }
        return targetFile
    }
}
