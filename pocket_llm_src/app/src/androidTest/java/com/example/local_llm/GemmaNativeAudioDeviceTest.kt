package com.example.local_llm

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile

@RunWith(AndroidJUnit4::class)
class GemmaNativeAudioDeviceTest {
    @Test
    fun e2bNativeAudioUnderThirtySeconds() = runModelSmoke(ModelRegistry.gemma4E2B)

    @Test
    fun e4bNativeAudioUnderThirtySeconds() = runModelSmoke(ModelRegistry.gemma4E4B)

    private fun runModelSmoke(spec: GemmaLiteRtSpec) {
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue(arguments.getString("runGemmaAudioDeviceTests") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val resolver = ModelFileResolver(context)
        assumeTrue("Install ${spec.displayName} before running this opt-in test.", resolver.isModelAvailable(spec))
        val audio = File(context.cacheDir, "device-smoke-${spec.id}.wav")
        writeSilentWav(audio, durationMillis = 500)
        val backend = GemmaLiteRtBackend(context, spec, resolver)
        try {
            runBlocking {
                backend.initialize()
                assertTrue("The native audio compatibility gate must pass.", backend.supportsNativeAudioInput)
                backend.streamReply(
                    InferenceRequest(
                        history = listOf(ChatTurn(role = ChatRole.USER, text = "Acknowledge the audio.")),
                        thinkingEnabled = false,
                        modelInstruction = spec.defaultSystemInstruction,
                        nativeAudioInputs = listOf(NativeAudioInput(audio.absolutePath, 0L, 500L))
                    ),
                    onPartial = {}
                )
            }
        } finally {
            backend.close()
            audio.delete()
        }
    }

    private fun writeSilentWav(file: File, durationMillis: Int) {
        val samples = 16_000 * durationMillis / 1_000
        val dataBytes = samples * 2
        RandomAccessFile(file, "rw").use { wav ->
            wav.setLength(0L)
            wav.writeBytes("RIFF")
            wav.writeInt(Integer.reverseBytes(dataBytes + 36))
            wav.writeBytes("WAVEfmt ")
            wav.writeInt(Integer.reverseBytes(16))
            wav.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
            wav.writeShort(java.lang.Short.reverseBytes(1.toShort()).toInt())
            wav.writeInt(Integer.reverseBytes(16_000))
            wav.writeInt(Integer.reverseBytes(32_000))
            wav.writeShort(java.lang.Short.reverseBytes(2.toShort()).toInt())
            wav.writeShort(java.lang.Short.reverseBytes(16.toShort()).toInt())
            wav.writeBytes("data")
            wav.writeInt(Integer.reverseBytes(dataBytes))
            wav.write(ByteArray(dataBytes))
        }
    }
}
