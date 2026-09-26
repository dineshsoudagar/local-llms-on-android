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
                val role = when {
                    turn.isToolResult -> "tool"
                    turn.role == ChatRole.USER -> "user"
                    else -> "assistant"
                }
                append('<').append(role).append(">\n")
                append(turn.text).append("\n</").append(role).append(">\n")
                turn.toolCalls.forEach { toolCall ->
                    append("<tool_call id=").append(toolCall.id).append(">")
                        .append(toolCall.name).append(':').append(toolCall.argumentsJson)
                        .append("</tool_call>\n")
                }
            }
            request.tools.forEach { tool ->
                append("<tool_definition>").append(tool.name).append(':')
                    .append(tool.parametersJson).append("</tool_definition>\n")
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

    fun fitHistoryWithinContext(request: InferenceRequest): List<ChatTurn> {
        val limit = promptTokenLimit(request)
        require(limit > 0) { "The requested output reserve leaves no room for an input prompt." }
        if (estimateSerializedPromptTokens(request) <= limit) {
            return request.history
        }

        val blocks = mutableListOf<List<ChatTurn>>()
        var end = request.history.size
        while (end > 0) {
            val last = request.history[end - 1]
            if (
                last.role == ChatRole.ASSISTANT &&
                end >= 2 &&
                request.history[end - 2].role == ChatRole.USER
            ) {
                blocks += request.history.subList(end - 2, end)
                end -= 2
            } else {
                blocks += listOf(last)
                end -= 1
            }
        }

        val retained = mutableListOf<ChatTurn>()
        for (block in blocks) {
            val candidate = block + retained
            val candidateRequest = request.copy(history = candidate)
            if (estimateSerializedPromptTokens(candidateRequest) > limit) {
                if (retained.isEmpty()) {
                    throw IllegalArgumentException(
                        "The current user message exceeds the ${limit}-token prompt budget."
                    )
                }
                continue
            }
            retained.addAll(0, block)
        }
        return retained
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
