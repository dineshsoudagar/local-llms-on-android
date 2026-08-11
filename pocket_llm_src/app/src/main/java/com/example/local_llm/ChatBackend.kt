package com.example.local_llm

interface ChatBackend : AutoCloseable {
    val capabilities: BackendCapabilities

    val supportsDirectImageInput: Boolean
        get() = capabilities.supportsNativeImage

    val supportsNativeAudioInput: Boolean
        get() = capabilities.supportsNativeAudio

    fun estimateTokens(text: String): Int = conservativeTokenEstimate(text)

    fun estimateSerializedPromptTokens(request: InferenceRequest): Int {
        val serialized = buildString {
            append("<system>\n").append(request.modelInstruction)
            append(if (request.thinkingEnabled) " /think" else " /no_think")
            append("\n</system>\n")
            request.history.forEach { turn ->
                val role = if (turn.role == ChatRole.USER) "user" else "assistant"
                append('<').append(role).append(">\n")
                append(turn.text).append("\n</").append(role).append(">\n")
            }
            append("<assistant>\n")
        }
        return maxOf(
            estimateTokens(serialized),
            serialized.toByteArray(Charsets.UTF_8).size
        )
    }

    fun promptTokenLimit(request: InferenceRequest): Int {
        require(request.outputTokenReserve >= 0) { "The output token reserve cannot be negative." }
        return capabilities.contextWindowTokens - request.outputTokenReserve
    }

    fun requirePromptFits(request: InferenceRequest): Int {
        val limit = promptTokenLimit(request)
        require(limit > 0) { "The requested output reserve leaves no room for an input prompt." }
        val actual = estimateSerializedPromptTokens(request)
        require(actual <= limit) {
            "The complete prompt needs $actual tokens, but this model allows $limit after reserving " +
                "${request.outputTokenReserve} tokens for the response. Shorten the system instruction, chat history, or request."
        }
        return actual
    }

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
