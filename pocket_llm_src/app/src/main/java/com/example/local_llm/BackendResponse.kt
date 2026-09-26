package com.example.local_llm

data class BackendResponse(
    val text: String,
    val thinkingText: String? = null,
    val toolCalls: List<ExternalToolCall> = emptyList()
)
