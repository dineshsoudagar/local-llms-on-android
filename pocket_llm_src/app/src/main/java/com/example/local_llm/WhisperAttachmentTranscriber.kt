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
    private val directory = File(context.filesDir, "asr/whisper-tiny-int8").apply { mkdirs() }

    fun files(): WhisperModelFiles = WhisperModelFiles(
        encoder = File(directory, ENCODER_FILE),
        decoder = File(directory, DECODER_FILE),
        tokens = File(directory, TOKENS_FILE)
    )

    fun isAvailable(): Boolean {
        val files = files()
        return files.encoder.length() >= MIN_ENCODER_BYTES &&
            files.decoder.length() >= MIN_DECODER_BYTES &&
            files.tokens.length() >= MIN_TOKENS_BYTES
    }

    suspend fun download(onProgress: (downloaded: Long, total: Long) -> Unit = { _, _ -> }) =
        withContext(Dispatchers.IO) {
            val targets = listOf(
                Download(ENCODER_FILE, MIN_ENCODER_BYTES),
                Download(DECODER_FILE, MIN_DECODER_BYTES),
                Download(TOKENS_FILE, MIN_TOKENS_BYTES)
            )
            var completed = 0L
            val totalExpected = targets.sumOf(Download::minimumBytes)
            targets.forEach { target ->
                currentCoroutineContext().ensureActive()
                val destination = File(directory, target.fileName)
                if (destination.length() >= target.minimumBytes) {
                    completed += target.minimumBytes
                    onProgress(completed, totalExpected)
                    return@forEach
                }
                val part = File(directory, target.fileName + ".part")
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
                    FileOutputStream(part).use { output ->
                        connection.inputStream.buffered().use { input ->
                            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                            var fileBytes = 0L
                            while (true) {
                                currentCoroutineContext().ensureActive()
                                val read = input.read(buffer)
                                if (read < 0) break
                                output.write(buffer, 0, read)
                                fileBytes += read
                                onProgress(completed + minOf(fileBytes, target.minimumBytes), totalExpected)
                            }
                        }
                    }
                    require(part.length() >= target.minimumBytes) { "The Whisper download was incomplete." }
                    if (destination.exists()) destination.delete()
                    require(part.renameTo(destination)) { "Could not install ${target.fileName}." }
                    completed += target.minimumBytes
                } finally {
                    connection.disconnect()
                }
            }
            check(isAvailable()) { "Whisper model validation failed after download." }
        }

    private data class Download(val fileName: String, val minimumBytes: Long)

    companion object {
        const val DOWNLOAD_SIZE_LABEL = "about 100 MB"
        private const val REVISION = "18e7b4cce9526312a66981ec3af7b9446b70db66"
        private const val MODEL_BASE_URL =
            "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/$REVISION"
        private const val ENCODER_FILE = "tiny-encoder.int8.onnx"
        private const val DECODER_FILE = "tiny-decoder.int8.onnx"
        private const val TOKENS_FILE = "tiny-tokens.txt"
        private const val MIN_ENCODER_BYTES = 10L * 1024L * 1024L
        private const val MIN_DECODER_BYTES = 80L * 1024L * 1024L
        private const val MIN_TOKENS_BYTES = 700L * 1024L
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
