package com.example.local_llm

import com.google.ai.edge.litertlm.Capabilities
import java.io.File

/** null means inspection failed or an older import has not been inspected yet. */
data class CustomModelCapabilities(
    val text: Boolean,
    val vision: Boolean,
    val audio: Boolean,
    val video: Boolean
) {
    fun inputSummary(): String = buildList {
        if (text) add("text")
        if (vision) add("image")
        if (audio) add("audio")
        if (video) add("video (not enabled)")
    }.joinToString().ifEmpty { "none detected" }
}

internal fun inspectCustomModel(file: File): CustomModelCapabilities? = runCatching {
    Capabilities(file.absolutePath).use { reader ->
        val inputs = reader.inputModalities()
        CustomModelCapabilities(inputs.text, inputs.vision, inputs.audio, inputs.video)
    }
}.getOrNull()

val ModelDescriptor.directImageInputAvailable: Boolean
    get() = when (this) {
        is GemmaLiteRtSpec -> directImageInputAvailable
        is CustomLiteRtSpec -> detectedInputs?.vision == true
        else -> false
    }

val ModelDescriptor.directAudioInputAvailable: Boolean
    get() = when (this) {
        is GemmaLiteRtSpec -> directAudioInputAvailable
        is CustomLiteRtSpec -> detectedInputs?.audio == true
        else -> false
    }

/** Custom audio starts with short segments; device feedback can justify larger limits later. */
val ModelDescriptor.nativeAudioSegmentMillis: Long
    get() = if (this is CustomLiteRtSpec) 10_000L else AttachmentLimits.GEMMA_MAX_AUDIO_INPUT_MILLIS

internal fun BackendCapabilities.customReadyStatus(): String =
    "Ready · native image: ${if (supportsNativeImage) "available" else "unavailable (OCR available)"}" +
        " · native audio: ${if (supportsNativeAudio) "available" else "unavailable"}"
