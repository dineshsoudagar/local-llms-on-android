package com.example.local_llm

import android.content.Context
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

data class WhisperModelFiles(
    val encoder: File,
    val decoder: File,
    val tokens: File
)

class WhisperModelStore(context: Context) {
    private val installer = WhisperModelInstaller(
        rootDirectory = File(context.filesDir, "asr"),
        modelDirectoryName = "whisper-tiny-int8",
        revision = REVISION,
        artifacts = MODEL_ARTIFACTS
    )

    fun files(): WhisperModelFiles = WhisperModelFiles(
        encoder = File(installer.installedDirectory, ENCODER_FILE),
        decoder = File(installer.installedDirectory, DECODER_FILE),
        tokens = File(installer.installedDirectory, TOKENS_FILE)
    )

    fun isAvailable(): Boolean = installer.isInstalled()

    suspend fun download(onProgress: (downloaded: Long, total: Long) -> Unit = { _, _ -> }) =
        withContext(Dispatchers.IO) {
            INSTALLATION_MUTEX.withLock {
                val totalExpected = MODEL_ARTIFACTS.sumOf(WhisperModelArtifact::expectedBytes)
                if (installer.isInstalled()) {
                    onProgress(totalExpected, totalExpected)
                    return@withLock
                }
                val stagingDirectory = installer.createStagingDirectory()
                try {
                    var completed = 0L
                    MODEL_ARTIFACTS.forEach { target ->
                        currentCoroutineContext().ensureActive()
                        downloadArtifact(target, stagingDirectory, completed, totalExpected, onProgress)
                        completed += target.expectedBytes
                        onProgress(completed, totalExpected)
                    }
                    installer.verifyAndPromote(stagingDirectory)
                    check(installer.isInstalled()) { "Whisper model validation failed after installation." }
                } catch (error: Throwable) {
                    if (stagingDirectory.exists()) installer.discardStagingDirectory(stagingDirectory)
                    throw error
                }
            }
        }

    private suspend fun downloadArtifact(
        target: WhisperModelArtifact,
        stagingDirectory: File,
        completed: Long,
        totalExpected: Long,
        onProgress: (downloaded: Long, total: Long) -> Unit
    ) {
        val destination = File(stagingDirectory, target.fileName)
        require(destination.name == target.fileName) { "Unexpected Whisper model file name." }
        val connection = URL("$MODEL_BASE_URL/${target.fileName}?download=true")
            .openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 20_000
            connection.readTimeout = 60_000
            connection.instanceFollowRedirects = true
            connection.connect()
            require(connection.responseCode in 200..299) {
                "Whisper download failed with HTTP ${connection.responseCode}."
            }
            if (connection.contentLengthLong >= 0L) {
                require(connection.contentLengthLong == target.expectedBytes) {
                    "Whisper download size changed for ${target.fileName}."
                }
            }
            FileOutputStream(destination).use { output ->
                connection.inputStream.buffered().use { input ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var fileBytes = 0L
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) break
                        fileBytes += read
                        require(fileBytes <= target.expectedBytes) {
                            "Whisper download exceeded its pinned size for ${target.fileName}."
                        }
                        output.write(buffer, 0, read)
                        onProgress(completed + fileBytes, totalExpected)
                    }
                    require(fileBytes == target.expectedBytes) {
                        "The Whisper download was incomplete for ${target.fileName}."
                    }
                }
                output.fd.sync()
            }
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        const val DOWNLOAD_SIZE_LABEL = "about 100 MB"
        private const val REVISION = "18e7b4cce9526312a66981ec3af7b9446b70db66"
        private const val MODEL_BASE_URL =
            "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/$REVISION"
        private const val ENCODER_FILE = "tiny-encoder.int8.onnx"
        private const val DECODER_FILE = "tiny-decoder.int8.onnx"
        private const val TOKENS_FILE = "tiny-tokens.txt"
        private val INSTALLATION_MUTEX = Mutex()
        internal val MODEL_ARTIFACTS = listOf(
            WhisperModelArtifact(
                fileName = ENCODER_FILE,
                expectedBytes = 12_937_678L,
                sha256 = "dd5531121627e7e184a6b8f6a6e353edf39b40e328032dfdc146b5a359a3e3d0"
            ),
            WhisperModelArtifact(
                fileName = DECODER_FILE,
                expectedBytes = 89_855_289L,
                sha256 = "dd29c14f2104c14c3aca7dc8e13308190bbe4c65be21c82f4b86a27d8326b26b"
            ),
            WhisperModelArtifact(
                fileName = TOKENS_FILE,
                expectedBytes = 816_730L,
                sha256 = "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"
            )
        )
    }
}

class WhisperAttachmentTranscriber(
    private val context: Context,
    private val modelStore: WhisperModelStore = WhisperModelStore(context)
) {
    suspend fun transcribe(
        descriptor: AttachmentDescriptor,
        existingChunks: List<AttachmentChunk> = emptyList(),
        onCheckpoint: (List<AttachmentChunk>) -> Unit = {}
    ): List<AttachmentChunk> = withContext(Dispatchers.IO) {
        require(modelStore.isAvailable()) { "The multilingual Whisper model has not been downloaded." }
        val files = modelStore.files()
        val config = OfflineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
            modelConfig = OfflineModelConfig(
                whisper = OfflineWhisperModelConfig(
                    encoder = files.encoder.absolutePath,
                    decoder = files.decoder.absolutePath,
                    language = "",
                    task = "transcribe",
                    enableTokenTimestamps = true,
                    enableSegmentTimestamps = true
                ),
                tokens = files.tokens.absolutePath,
                numThreads = 4,
                provider = "cpu"
            )
        )
        val recognizer = OfflineRecognizer(context.assets, config)
        val segmented = GemmaAudioSegmenter(context).segment(descriptor)
        try {
            val merged = existingChunks.sortedBy(AttachmentChunk::ordinal).toMutableList()
            segmented.inputs.forEachIndexed { index, input ->
                if (index < merged.size) return@forEachIndexed
                currentCoroutineContext().ensureActive()
                val samples = readCanonicalWav(File(input.filePath))
                val stream = recognizer.createStream()
                try {
                    stream.acceptWaveform(samples, SAMPLE_RATE)
                    recognizer.decode(stream)
                    val result = recognizer.getResult(stream)
                    val chunk = AttachmentChunk(
                        ordinal = index,
                        text = result.text.trim(),
                        source = AttachmentSourceRef(
                            startMillis = input.startMillis,
                            endMillis = input.endMillis
                        ),
                        estimatedTokens = conservativeTokenEstimate(result.text)
                    )
                    val text = merged.lastOrNull()?.let { deduplicateSegmentOverlap(it.text, chunk.text) }
                        ?: chunk.text
                    merged += chunk.copy(text = text, estimatedTokens = conservativeTokenEstimate(text))
                    onCheckpoint(merged.toList())
                } finally {
                    stream.release()
                }
            }
            merged
        } finally {
            recognizer.release()
            segmented.temporaryFiles.forEach { File(it).delete() }
        }
    }

    private fun readCanonicalWav(file: File): FloatArray {
        require(file.length() >= WAV_HEADER_BYTES) { "Normalized WAV is malformed." }
        val sampleCount = ((file.length() - WAV_HEADER_BYTES) / 2L).toInt()
        val samples = FloatArray(sampleCount)
        BufferedInputStream(FileInputStream(file)).use { input ->
            var skipped = 0L
            while (skipped < WAV_HEADER_BYTES) {
                val count = input.skip(WAV_HEADER_BYTES - skipped)
                if (count <= 0L) throw IllegalArgumentException("Normalized WAV header is truncated.")
                skipped += count
            }
            for (index in samples.indices) {
                val low = input.read()
                val high = input.read()
                require(low >= 0 && high >= 0) { "Normalized WAV data is truncated." }
                val sample = (low or (high shl 8)).toShort()
                samples[index] = sample / 32768f
            }
        }
        return samples
    }

    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val WAV_HEADER_BYTES = 44L
    }
}
