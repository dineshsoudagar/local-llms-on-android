package com.example.local_llm

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext

data class ModelDownloadProgress(
    val fileName: String,
    val bytesDownloaded: Long,
    val totalBytes: Long?
)

class ModelDownloader(
    private val modelFileResolver: ModelFileResolver
) {

    companion object {
        private const val CONNECT_TIMEOUT_MS = 15_000
        private const val READ_TIMEOUT_MS = 60_000
        private const val BUFFER_SIZE = 128 * 1024
        private const val MAX_REDIRECTS = 6
    }

    suspend fun downloadModel(
        descriptor: ModelDescriptor,
        onProgress: (ModelDownloadProgress) -> Unit
    ) = withContext(Dispatchers.IO) {
        descriptor.downloadFiles.forEach { downloadFile ->
            coroutineContext.ensureActive()
            downloadFile(descriptor, downloadFile, onProgress)
        }
    }

    private suspend fun downloadFile(
        descriptor: ModelDescriptor,
        downloadFile: ModelDownloadFile,
        onProgress: (ModelDownloadProgress) -> Unit
    ) {
        val targetFile = modelFileResolver.getDownloadedFile(descriptor, downloadFile.localFileName)
        if (targetFile.exists() && targetFile.length() > 0L) {
            onProgress(
                ModelDownloadProgress(
                    fileName = downloadFile.localFileName,
                    bytesDownloaded = targetFile.length(),
                    totalBytes = targetFile.length()
                )
            )
            return
        }

        val tempFile = File(targetFile.absolutePath + ".download")
        tempFile.parentFile?.mkdirs()
        if (tempFile.exists()) {
            tempFile.delete()
        }

        val connection = openConnection(downloadFile.downloadUrl)
        try {
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                throw IOException("Download failed with HTTP $responseCode for ${descriptor.displayName}.")
            }

            val totalBytes = connection.contentLengthLong.takeIf { it > 0L }
            connection.inputStream.use { rawInput ->
                BufferedInputStream(rawInput).use { input ->
                    FileOutputStream(tempFile).use { rawOutput ->
                        BufferedOutputStream(rawOutput).use { output ->
                            val buffer = ByteArray(BUFFER_SIZE)
                            var bytesCopied = 0L
                            var lastReportedBytes = Long.MIN_VALUE

                            while (true) {
                                coroutineContext.ensureActive()
                                val read = input.read(buffer)
                                if (read == -1) {
                                    break
                                }

                                output.write(buffer, 0, read)
                                bytesCopied += read

                                if (totalBytes == null || bytesCopied - lastReportedBytes >= 512 * 1024L) {
                                    onProgress(
                                        ModelDownloadProgress(
                                            fileName = downloadFile.localFileName,
                                            bytesDownloaded = bytesCopied,
                                            totalBytes = totalBytes
                                        )
                                    )
                                    lastReportedBytes = bytesCopied
                                }
                            }

                            output.flush()
                            onProgress(
                                ModelDownloadProgress(
                                    fileName = downloadFile.localFileName,
                                    bytesDownloaded = bytesCopied,
                                    totalBytes = totalBytes ?: bytesCopied
                                )
                            )
                        }
                    }
                }
            }

            if (targetFile.exists()) {
                targetFile.delete()
            }
            check(tempFile.renameTo(targetFile)) {
                "Failed to move downloaded file into place: ${targetFile.absolutePath}"
            }
        } catch (cancelled: CancellationException) {
            tempFile.delete()
            throw cancelled
        } catch (error: Exception) {
            tempFile.delete()
            throw error
        } finally {
            connection.disconnect()
        }
    }

    private fun openConnection(url: String, redirectCount: Int = 0): HttpURLConnection {
        require(redirectCount <= MAX_REDIRECTS) {
            "Too many redirects while downloading model files."
        }

        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = false
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            setRequestProperty("User-Agent", "PocketLLM/1.0")
            setRequestProperty("Accept-Encoding", "identity")
        }

        return when (connection.responseCode) {
            HttpURLConnection.HTTP_MOVED_PERM,
            HttpURLConnection.HTTP_MOVED_TEMP,
            HttpURLConnection.HTTP_SEE_OTHER,
            307,
            308 -> {
                val nextUrl = connection.getHeaderField("Location")
                    ?: throw IOException("Redirect response did not include a Location header.")
                connection.disconnect()
                openConnection(URL(URL(url), nextUrl).toString(), redirectCount + 1)
            }
            else -> connection
        }
    }
}
