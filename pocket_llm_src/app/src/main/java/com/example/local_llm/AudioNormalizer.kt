package com.example.local_llm

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.io.RandomAccessFile

data class NormalizedAudio(
    val file: File,
    val durationMillis: Long,
    val normalizedSampleCount: Long
)

internal class DecodedAudioDurationTracker(
    val containerDurationEstimateMicros: Long?,
    private val maximumSamples: Long = AttachmentLimits.MAX_NORMALIZED_AUDIO_SAMPLES
) {
    var normalizedSampleCount: Long = 0L
        private set

    val durationMillis: Long
        get() = normalizedSampleCount * 1_000L / AttachmentLimits.NORMALIZED_AUDIO_SAMPLE_RATE

    fun recordSamples(count: Long) {
        require(count >= 0L) { "The decoded sample count cannot be negative." }
        require(normalizedSampleCount <= maximumSamples - count) {
            "Audio must be no longer than 2 hours of decoded 16 kHz mono samples."
        }
        normalizedSampleCount += count
    }
}

internal class Pcm16MonoResampler(
    private val output: OutputStream,
    private val durationTracker: DecodedAudioDurationTracker
) {
    private var sourceRate: Int? = null
    private var channels: Int? = null
    private var phase = 0L
    private var pendingFrame = ByteArray(0)
    private var pendingSize = 0

    fun consume(bytes: ByteArray, decodedSampleRate: Int, decodedChannels: Int) {
        require(decodedSampleRate > 0 && decodedChannels > 0) { "The decoded audio format is invalid." }
        val configuredRate = sourceRate
        val configuredChannels = channels
        require(configuredRate == null || configuredRate == decodedSampleRate) {
            "The decoded audio sample rate changed during normalization."
        }
        require(configuredChannels == null || configuredChannels == decodedChannels) {
            "The decoded audio channel count changed during normalization."
        }
        sourceRate = decodedSampleRate
        channels = decodedChannels
        val frameBytes = decodedChannels * PCM16_BYTES_PER_SAMPLE
        if (pendingFrame.size != frameBytes) pendingFrame = ByteArray(frameBytes)
        var byteIndex = 0
        while (byteIndex < bytes.size) {
            pendingFrame[pendingSize++] = bytes[byteIndex++]
            if (pendingSize == frameBytes) {
                writeFrame(pendingFrame, decodedSampleRate, decodedChannels)
                pendingSize = 0
            }
        }
    }

    fun finish() {
        require(pendingSize == 0) { "The decoder returned an incomplete PCM audio frame." }
        require(durationTracker.normalizedSampleCount > 0L) {
            "No audible samples could be decoded from this file."
        }
    }

    private fun writeFrame(frame: ByteArray, decodedSampleRate: Int, decodedChannels: Int) {
        var mixed = 0L
        for (channel in 0 until decodedChannels) {
            val offset = channel * PCM16_BYTES_PER_SAMPLE
            val sample = (frame[offset].toInt() and 0xff) or (frame[offset + 1].toInt() shl 8)
            mixed += sample.toShort().toInt()
        }
        val mono = (mixed / decodedChannels).coerceIn(Short.MIN_VALUE.toLong(), Short.MAX_VALUE.toLong()).toInt()
        phase += AttachmentLimits.NORMALIZED_AUDIO_SAMPLE_RATE
        while (phase >= decodedSampleRate) {
            durationTracker.recordSamples(1L)
            output.write(mono and 0xff)
            output.write((mono ushr 8) and 0xff)
            phase -= decodedSampleRate
        }
    }

    private companion object {
        const val PCM16_BYTES_PER_SAMPLE = 2
    }
}

internal fun cleanupPartialAudioOutput(destination: File) {
    if (destination.exists()) destination.delete()
}

class AudioNormalizer {
    suspend fun normalize(source: File, destination: File): NormalizedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(source.absolutePath)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
                extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: throw IllegalArgumentException("This file does not contain a decodable audio track.")
            val inputFormat = extractor.getTrackFormat(trackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalArgumentException("The audio codec could not be identified.")
            val durationTracker = DecodedAudioDurationTracker(
                containerDurationEstimateMicros = inputFormat
                    .takeIf { it.containsKey(MediaFormat.KEY_DURATION) }
                    ?.getLong(MediaFormat.KEY_DURATION)
            )

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
            destination.parentFile?.mkdirs()
            BufferedOutputStream(FileOutputStream(destination)).use { normalizedOutput ->
                normalizedOutput.write(ByteArray(WAV_HEADER_BYTES))
                val resampler = Pcm16MonoResampler(normalizedOutput, durationTracker)
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
                                resampler.consume(bytes, outputSampleRate, outputChannels)
                            }
                            outputEnded = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            codec.releaseOutputBuffer(outputIndex, false)
                        }
                    }
                }
                resampler.finish()
                normalizedOutput.flush()
            }

            writeWavHeader(destination, durationTracker.normalizedSampleCount * PCM16_BYTES_PER_SAMPLE)
            return NormalizedAudio(
                file = destination,
                durationMillis = durationTracker.durationMillis,
                normalizedSampleCount = durationTracker.normalizedSampleCount
            )
        } catch (error: Throwable) {
            cleanupPartialAudioOutput(destination)
            throw error
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
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
            wav.writeIntLE(AttachmentLimits.NORMALIZED_AUDIO_SAMPLE_RATE)
            wav.writeIntLE(AttachmentLimits.NORMALIZED_AUDIO_SAMPLE_RATE * 2)
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
        private const val WAV_HEADER_BYTES = 44
        private const val PCM16_BYTES_PER_SAMPLE = 2L
    }
}
