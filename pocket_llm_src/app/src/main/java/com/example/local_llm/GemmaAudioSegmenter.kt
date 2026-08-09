package com.example.local_llm

import android.content.Context
import java.io.File
import java.io.RandomAccessFile

data class SegmentedNativeAudio(
    val inputs: List<NativeAudioInput>,
    val temporaryFiles: List<String>
)

class GemmaAudioSegmenter(context: Context) {
    private val cacheDirectory = File(context.cacheDir, "gemma_audio_segments").apply { mkdirs() }

    fun segment(descriptor: AttachmentDescriptor): SegmentedNativeAudio {
        val duration = descriptor.durationMillis
            ?: throw IllegalArgumentException("Audio duration is unavailable.")
        require(duration <= AttachmentLimits.MAX_AUDIO_DURATION_MILLIS)
        if (duration <= AttachmentLimits.GEMMA_MAX_AUDIO_INPUT_MILLIS) {
            return SegmentedNativeAudio(
                inputs = listOf(NativeAudioInput(descriptor.sourcePath, 0L, duration)),
                temporaryFiles = emptyList()
            )
        }

        val source = File(descriptor.sourcePath)
        require(source.length() >= WAV_HEADER_BYTES) { "Normalized WAV is malformed." }
        val inputs = mutableListOf<NativeAudioInput>()
        val temporary = mutableListOf<String>()
        var startMillis = 0L
        var ordinal = 0
        while (startMillis < duration) {
            val endMillis = minOf(startMillis + AttachmentLimits.AUDIO_SEGMENT_MILLIS, duration)
            val output = File(cacheDirectory, "${descriptor.id}-${ordinal++}.wav")
            copySegment(source, output, startMillis, endMillis)
            temporary += output.absolutePath
            inputs += NativeAudioInput(output.absolutePath, startMillis, endMillis)
            if (endMillis >= duration) break
            startMillis = endMillis - AttachmentLimits.AUDIO_SEGMENT_OVERLAP_MILLIS
        }
        return SegmentedNativeAudio(inputs, temporary)
    }

    private fun copySegment(source: File, destination: File, startMillis: Long, endMillis: Long) {
        val startByte = WAV_HEADER_BYTES + (startMillis * BYTES_PER_SECOND / 1_000L)
        val dataBytes = ((endMillis - startMillis) * BYTES_PER_SECOND / 1_000L)
            .coerceAtMost(source.length() - startByte)
            .coerceAtLeast(0L)
        require(dataBytes > 0L && dataBytes <= Int.MAX_VALUE) { "Audio segment is empty or too large." }
        RandomAccessFile(source, "r").use { input ->
            RandomAccessFile(destination, "rw").use { output ->
                output.setLength(0L)
                writeHeader(output, dataBytes.toInt())
                input.seek(startByte)
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var remaining = dataBytes
                while (remaining > 0L) {
                    val read = input.read(buffer, 0, minOf(buffer.size.toLong(), remaining).toInt())
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    remaining -= read
                }
                require(remaining == 0L) { "Normalized audio ended before the requested segment." }
            }
        }
    }

    private fun writeHeader(output: RandomAccessFile, dataBytes: Int) {
        output.writeBytes("RIFF")
        output.writeIntLE(dataBytes + 36)
        output.writeBytes("WAVEfmt ")
        output.writeIntLE(16)
        output.writeShortLE(1)
        output.writeShortLE(1)
        output.writeIntLE(SAMPLE_RATE)
        output.writeIntLE(BYTES_PER_SECOND)
        output.writeShortLE(2)
        output.writeShortLE(16)
        output.writeBytes("data")
        output.writeIntLE(dataBytes)
    }

    private fun RandomAccessFile.writeIntLE(value: Int) = writeInt(Integer.reverseBytes(value))
    private fun RandomAccessFile.writeShortLE(value: Int) =
        writeShort(java.lang.Short.reverseBytes(value.toShort()).toInt())

    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val BYTES_PER_SECOND = SAMPLE_RATE * 2
        private const val WAV_HEADER_BYTES = 44L
    }
}
