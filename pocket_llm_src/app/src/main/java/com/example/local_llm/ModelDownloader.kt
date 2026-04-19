package com.example.local_llm

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
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
        private const val PROGRESS_STEP_BYTES = 512 * 1024L
        private const val HTTP_RANGE_NOT_SATISFIABLE = 416
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

        val resumeFromBytes = tempFile.length().takeIf { it > 0L } ?: 0L
        val connection = openConnection(downloadFile.downloadUrl, startByte = resumeFromBytes)
        val cancellationHandle = coroutineContext[Job]?.invokeOnCompletion { cause ->
            if (cause is CancellationException) {
                connection.disconnect()
            }
        }
        try {
            val responseCode = connection.responseCode
            if (responseCode == HTTP_RANGE_NOT_SATISFIABLE && resumeFromBytes > 0L) {
                connection.disconnect()
                tempFile.delete()
                downloadFile(descriptor, downloadFile, onProgress)
                return
            }

            if (responseCode !in 200..299) {
                throw IOException("Download failed with HTTP $responseCode for ${descriptor.displayName}.")
            }

            val isResuming = resumeFromBytes > 0L && responseCode == HttpURLConnection.HTTP_PARTIAL
            if (resumeFromBytes > 0L && !isResuming) {
                tempFile.delete()
            }

            val startingBytes = if (isResuming) resumeFromBytes else 0L
            val responseBytes = connection.contentLengthLong.takeIf { it > 0L }
            val totalBytes = when {
                isResuming -> parseContentRangeTotal(connection.getHeaderField("Content-Range"))
                    ?: responseBytes?.plus(startingBytes)
                else -> responseBytes
            }

            onProgress(
                ModelDownloadProgress(
                    fileName = downloadFile.localFileName,
                    bytesDownloaded = startingBytes,
                    totalBytes = totalBytes
                )
            )

            connection.inputStream.use { rawInput ->
                BufferedInputStream(rawInput).use { input ->
                    FileOutputStream(tempFile, isResuming).use { rawOutput ->
                        BufferedOutputStream(rawOutput).use { output ->
                            val buffer = ByteArray(BUFFER_SIZE)
                            var bytesCopied = startingBytes
                            var lastReportedBytes = startingBytes

                            while (true) {
                                coroutineContext.ensureActive()
                                val read = input.read(buffer)
                                if (read == -1) {
                                    break
                                }

                                output.write(buffer, 0, read)
                                bytesCopied += read

                                if (bytesCopied - lastReportedBytes >= PROGRESS_STEP_BYTES) {
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
                            rawOutput.fd.sync()
                            onProgress(
                                ModelDownloadProgress(
                                    fileName = downloadFile.localFileName,
                                    bytesDownloaded = bytesCopied,
                                    totalBytes = totalBytes ?: bytesCopied
                                )
                            )

                            if (totalBytes != null && bytesCopied != totalBytes) {
                                throw IOException(
                                    "Downloaded ${bytesCopied} bytes for ${downloadFile.localFileName}, expected ${totalBytes} bytes."
                                )
                            }
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
            throw cancelled
        } catch (error: Exception) {
            coroutineContext.ensureActive()
            throw error
        } finally {
            cancellationHandle?.dispose()
            connection.disconnect()
        }
    }

    private fun openConnection(
        url: String,
        redirectCount: Int = 0,
        startByte: Long = 0L
    ): HttpURLConnection {
        require(redirectCount <= MAX_REDIRECTS) {
            "Too many redirects while downloading model files."
        }

        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = false
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            setRequestProperty("User-Agent", "PocketLLM/1.0")
            setRequestProperty("Accept-Encoding", "identity")
            if (startByte > 0L) {
                setRequestProperty("Range", "bytes=$startByte-")
            }
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
                openConnection(URL(URL(url), nextUrl).toString(), redirectCount + 1, startByte)
            }
            else -> connection
        }
    }

    private fun parseContentRangeTotal(contentRange: String?): Long? {
        if (contentRange.isNullOrBlank()) {
            return null
        }

        val totalStart = contentRange.lastIndexOf('/') + 1
        if (totalStart <= 0 || totalStart >= contentRange.length) {
            return null
        }

        return contentRange.substring(totalStart)
            .toLongOrNull()
            ?.takeIf { it > 0L }
    }
}
