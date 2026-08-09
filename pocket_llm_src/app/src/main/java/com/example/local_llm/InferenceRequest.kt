package com.example.local_llm

data class BackendCapabilities(
    val supportsNativeImage: Boolean = false,
    val supportsNativeAudio: Boolean = false,
    val contextWindowTokens: Int = 512
)

data class InferenceRequest(
    val history: List<ChatTurn>,
    val thinkingEnabled: Boolean,
    val modelInstruction: String,
    val imageFilePaths: List<String> = emptyList(),
    val nativeAudioInputs: List<NativeAudioInput> = emptyList(),
    val attachmentContext: AttachmentContext? = null
)
