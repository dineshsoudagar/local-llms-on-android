package com.example.local_llm

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Tesseract language data for the F-Droid build. Not bundled in the APK: the pinned
 * eng.traineddata from tesseract-ocr/tessdata_fast is downloaded on first OCR use,
 * size- and SHA-256-verified, and installed atomically under filesDir/ocr/tessdata.
 */
internal object TesseractLanguageData {
    const val LANGUAGE = "eng"
    private const val DATA_FILE = "eng.traineddata"
    private const val REVISION = "4.1.0"
    private const val DATA_URL =
        "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/$REVISION/$DATA_FILE"
    private const val DOWNLOAD_SIZE_LABEL = "about 4 MB"
    private val ARTIFACTS = listOf(
        WhisperModelArtifact(
            fileName = DATA_FILE,
            expectedBytes = 4_113_088L,
            sha256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
        )
    )

    private val installMutex = Mutex()
    @Volatile
    private var installer: WhisperModelInstaller? = null

    /** Returns the Tesseract data path (the parent of `tessdata`), downloading the data if needed. */
    suspend fun ensureInstalled(context: Context, statusListener: OcrStatusListener?): String =
        withContext(Dispatchers.IO) {
            installMutex.withLock {
                val dataInstaller = obtainInstaller(context)
                if (!dataInstaller.isInstalled()) {
                    download(dataInstaller, statusListener)
                }
                dataRoot(context).absolutePath
            }
        }

    /** Removes installed data that Tesseract could not load, so the next attempt downloads it again. */
    fun discardInstallation(context: Context) {
        runCatching { File(dataRoot(context), "tessdata").deleteRecursively() }
        installer = null
    }

    private fun dataRoot(context: Context): File = File(context.applicationContext.filesDir, "ocr")

    private fun obtainInstaller(context: Context): WhisperModelInstaller {
        installer?.let { return it }
        return WhisperModelInstaller(
            rootDirectory = dataRoot(context),
            modelDirectoryName = "tessdata",
            revision = "tessdata_fast-$REVISION",
            artifacts = ARTIFACTS
        ).also { installer = it }
    }

    private suspend fun download(dataInstaller: WhisperModelInstaller, statusListener: OcrStatusListener?) {
        val artifact = ARTIFACTS.single()
        statusListener?.onOcrStatus("Downloading OCR language data ($DOWNLOAD_SIZE_LABEL)...")
        val stagingDirectory = dataInstaller.createStagingDirectory()
        try {
            try {
                downloadArtifact(artifact, File(stagingDirectory, artifact.fileName), statusListener)
            } catch (error: IOException) {
                throw IOException(
                    "Text recognition needs its English language data ($DOWNLOAD_SIZE_LABEL) once. " +
                        "Connect to the internet and try again. (${error.message ?: error.javaClass.simpleName})",
                    error
                )
            }
            try {
                dataInstaller.verifyAndPromote(stagingDirectory)
            } catch (error: IllegalArgumentException) {
                throw IOException("The downloaded OCR language data was damaged. Try again.", error)
            }
            check(dataInstaller.isInstalled()) { "OCR language data validation failed after installation." }
            statusListener?.onOcrStatus("OCR language data installed.")
        } catch (error: Throwable) {
            if (stagingDirectory.exists()) {
                runCatching { dataInstaller.discardStagingDirectory(stagingDirectory) }
            }
            throw error
        }
    }

    private suspend fun downloadArtifact(
        artifact: WhisperModelArtifact,
        destination: File,
        statusListener: OcrStatusListener?
    ) {
        val connection = URL(DATA_URL).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 20_000
            connection.readTimeout = 60_000
            connection.instanceFollowRedirects = true
            connection.setRequestProperty("User-Agent", "PocketLLM/1.0")
            connection.setRequestProperty("Accept-Encoding", "identity")
            connection.connect()
            if (connection.responseCode !in 200..299) {
                throw IOException("Download failed with HTTP ${connection.responseCode}.")
            }
            if (connection.contentLengthLong >= 0L && connection.contentLengthLong != artifact.expectedBytes) {
                throw IOException("Unexpected download size ${connection.contentLengthLong}.")
            }
            var lastReportedPercent = -1
            FileOutputStream(destination).use { output ->
                connection.inputStream.buffered().use { input ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var fileBytes = 0L
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) break
                        fileBytes += read
                        if (fileBytes > artifact.expectedBytes) {
                            throw IOException("Download exceeded its expected size.")
                        }
                        output.write(buffer, 0, read)
                        val percent = (fileBytes * 100 / artifact.expectedBytes).toInt()
                        if (percent / 10 != lastReportedPercent / 10) {
                            lastReportedPercent = percent
                            statusListener?.onOcrStatus("Downloading OCR language data... $percent%")
                        }
                    }
                    if (fileBytes != artifact.expectedBytes) {
                        throw IOException("The download was incomplete.")
                    }
                }
                output.fd.sync()
            }
        } finally {
            connection.disconnect()
        }
    }
}
