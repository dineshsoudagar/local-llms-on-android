package com.example.local_llm

import android.content.Context
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream

class ModelFileResolver(private val context: Context) {

    fun getModelDirectory(descriptor: ModelDescriptor): File {
        return File(context.filesDir, "models/${descriptor.id}").apply { mkdirs() }
    }

    fun getDownloadedFile(descriptor: ModelDescriptor, fileName: String): File {
        return File(getModelDirectory(descriptor), fileName)
    }

    fun isFileDownloaded(descriptor: ModelDescriptor, fileName: String): Boolean {
        val file = getDownloadedFile(descriptor, fileName)
        return file.exists() && file.length() > 0L
    }

    fun hasBundledAsset(fileName: String): Boolean {
        return runCatching {
            AssetLocator.resolvePath(context, fileName)
            true
        }.getOrDefault(false)
    }

    fun isModelDownloaded(descriptor: ModelDescriptor): Boolean {
        return descriptor.downloadFiles.all { downloadFile ->
            isFileDownloaded(descriptor, downloadFile.localFileName)
        }
    }

    fun isModelAvailable(descriptor: ModelDescriptor): Boolean {
        return descriptor.downloadFiles.all { downloadFile ->
            isFileDownloaded(descriptor, downloadFile.localFileName) ||
                hasBundledAsset(downloadFile.localFileName)
        }
    }

    fun resolveModelFile(descriptor: ModelDescriptor): File {
        return resolveFile(descriptor, descriptor.primaryModelFileName)
    }

    fun resolveFile(descriptor: ModelDescriptor, fileName: String): File {
        val downloadedFile = getDownloadedFile(descriptor, fileName)
        if (downloadedFile.exists() && downloadedFile.length() > 0L) {
            return downloadedFile
        }

        val resolvedAssetPath = runCatching { AssetLocator.resolvePath(context, fileName) }
            .getOrElse {
                throw FileNotFoundException(
                    "Missing '$fileName' for ${descriptor.displayName}. Download the model first."
                )
            }

        val tempFile = File(downloadedFile.absolutePath + ".tmp")
        tempFile.parentFile?.mkdirs()

        context.assets.open(resolvedAssetPath).use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        }

        if (downloadedFile.exists()) {
            downloadedFile.delete()
        }
        check(tempFile.renameTo(downloadedFile)) {
            "Failed to move copied asset into place: ${downloadedFile.absolutePath}"
        }

        return downloadedFile
    }
}
