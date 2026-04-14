package com.example.local_llm

data class ChatUiState(
    val title: String,
    val transcript: List<ChatTurn> = emptyList(),
    val statusMessage: String = "",
    val isLoading: Boolean = false,
    val isReady: Boolean = false,
    val isGenerating: Boolean = false,
    val supportsThinking: Boolean = false
)
