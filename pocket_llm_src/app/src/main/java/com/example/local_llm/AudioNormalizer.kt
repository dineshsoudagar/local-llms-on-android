package com.example.local_llm

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile

data class NormalizedAudio(
    val file: File,
    val durationMillis: Long
)

class AudioNormalizer {
    suspend fun normalize(source: File, destination: File): NormalizedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        val rawFile = File(destination.parentFile, ".decoded-${destination.name}.pcm")
        try {
            extractor.setDataSource(source.absolutePath)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: throw IllegalArgumentException("This file does not contain a decodable audio track.")
            val inputFormat = extractor.getTrackFormat(trackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalArgumentException("The audio codec could not be identified.")
            val durationMillis = inputFormat.getLong(MediaFormat.KEY_DURATION) / 1_000L
            require(durationMillis in 1..AttachmentLimits.MAX_AUDIO_DURATION_MILLIS) {
                "Audio must be no longer than 2 hours."
            }

            extractor.selectTrack(trackIndex)
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(inputFormat, null, null, 0)
            codec.start()

            var outputSampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var outputChannels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
            var inputEnded = false
            var outputEnded = false
            val bufferInfo = MediaCodec.BufferInfo()
            BufferedOutputStream(FileOutputStream(rawFile)).use { rawOutput ->
                while (!outputEnded) {
                    currentCoroutineContext().ensureActive()
                    if (!inputEnded) {
                        val inputIndex = codec.dequeueInputBuffer(CODEC_TIMEOUT_US)
                        if (inputIndex >= 0) {
                            val inputBuffer = codec.getInputBuffer(inputIndex)
                                ?: throw IllegalStateException("Audio decoder input buffer is unavailable.")
                            val sampleSize = extractor.readSampleData(inputBuffer, 0)
                            if (sampleSize < 0) {
                                codec.queueInputBuffer(
                                    inputIndex,
                                    0,
                                    0,
                                    0L,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                                )
                                inputEnded = true
                            } else {
                                codec.queueInputBuffer(inputIndex, 0, sampleSize, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }

                    when (val outputIndex = codec.dequeueOutputBuffer(bufferInfo, CODEC_TIMEOUT_US)) {
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val format = codec.outputFormat
                            outputSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            outputChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            pcmEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                                format.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            } else {
                                AudioFormat.ENCODING_PCM_16BIT
                            }
                        }

                        MediaCodec.INFO_TRY_AGAIN_LATER, MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                        else -> if (outputIndex >= 0) {
                            val outputBuffer = codec.getOutputBuffer(outputIndex)
                            if (bufferInfo.size > 0 && outputBuffer != null) {
                                require(pcmEncoding == AudioFormat.ENCODING_PCM_16BIT) {
                                    "The decoded audio is not 16-bit PCM and cannot be normalized safely."
                                }
                                val bytes = ByteArray(bufferInfo.size)
                                outputBuffer.position(bufferInfo.offset)
                                outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                                outputBuffer.get(bytes)
                                rawOutput.write(bytes)
                            }
                            outputEnded = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            codec.releaseOutputBuffer(outputIndex, false)
                        }
                    }
                }
            }

            writeMono16kWav(rawFile, destination, outputSampleRate, outputChannels)
            return NormalizedAudio(destination, durationMillis)
        } catch (error: Throwable) {
            destination.delete()
            throw error
        } finally {
            rawFile.delete()
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
    }

    private fun writeMono16kWav(rawFile: File, destination: File, sourceRate: Int, channels: Int) {
        require(sourceRate > 0 && channels > 0) { "The decoded audio format is invalid." }
        destination.parentFile?.mkdirs()
        BufferedInputStream(FileInputStream(rawFile)).use { input ->
            BufferedOutputStream(FileOutputStream(destination)).use { output ->
                output.write(ByteArray(WAV_HEADER_BYTES))
                val frame = ByteArray(channels * 2)
                var sourceFrame = 0L
                var outputFrame = 0L
                var dataBytes = 0L
                while (readFrame(input, frame)) {
                    var mixed = 0
                    for (channel in 0 until channels) {
                        val offset = channel * 2
                        mixed += (frame[offset].toInt() and 0xff) or (frame[offset + 1].toInt() shl 8)
                    }
                    val mono = (mixed / channels).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                    while ((outputFrame * sourceRate) / TARGET_SAMPLE_RATE <= sourceFrame) {
                        output.write(mono and 0xff)
                        output.write((mono ushr 8) and 0xff)
                        dataBytes += 2
                        outputFrame += 1
                    }
                    sourceFrame += 1
                }
                output.flush()
                require(dataBytes > 0L) { "No audible samples could be decoded from this file." }
                writeWavHeader(destination, dataBytes)
            }
        }
    }

    private fun readFrame(input: BufferedInputStream, frame: ByteArray): Boolean {
        var offset = 0
        while (offset < frame.size) {
            val count = input.read(frame, offset, frame.size - offset)
            if (count < 0) return offset == frame.size
            offset += count
        }
        return true
    }

    private fun writeWavHeader(file: File, dataBytes: Long) {
        require(dataBytes <= Int.MAX_VALUE - WAV_HEADER_BYTES) { "Normalized audio is too large for WAV." }
        RandomAccessFile(file, "rw").use { wav ->
            wav.seek(0)
            wav.writeBytes("RIFF")
            wav.writeIntLE((dataBytes + 36).toInt())
            wav.writeBytes("WAVEfmt ")
            wav.writeIntLE(16)
            wav.writeShortLE(1)
            wav.writeShortLE(1)
            wav.writeIntLE(TARGET_SAMPLE_RATE)
            wav.writeIntLE(TARGET_SAMPLE_RATE * 2)
            wav.writeShortLE(2)
            wav.writeShortLE(16)
            wav.writeBytes("data")
            wav.writeIntLE(dataBytes.toInt())
        }
    }

    private fun RandomAccessFile.writeIntLE(value: Int) {
        writeInt(Integer.reverseBytes(value))
    }

    private fun RandomAccessFile.writeShortLE(value: Int) {
        writeShort(java.lang.Short.reverseBytes(value.toShort()).toInt())
    }

    companion object {
        private const val CODEC_TIMEOUT_US = 10_000L
        private const val TARGET_SAMPLE_RATE = 16_000
        private const val WAV_HEADER_BYTES = 44
    }
}
