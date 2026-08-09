package com.example.local_llm

interface ChatBackend : AutoCloseable {
    val capabilities: BackendCapabilities

    val supportsDirectImageInput: Boolean
        get() = capabilities.supportsNativeImage

    val supportsNativeAudioInput: Boolean
        get() = capabilities.supportsNativeAudio

    fun estimateTokens(text: String): Int = conservativeTokenEstimate(text)

    suspend fun initialize()
    suspend fun resetConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String
    )

    suspend fun streamReply(
        request: InferenceRequest,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse
    fun cancelGeneration()
}
