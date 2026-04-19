package com.example.local_llm

const val MODEL_LOADING_STATUS_MESSAGE = "Please wait model is being loaded"
const val MODEL_READY_STATUS_MESSAGE = "Model is ready"

data class ChatUiState(
    val title: String,
    val transcript: List<ChatTurn> = emptyList(),
    val statusMessage: String = "",
    val isLoading: Boolean = false,
    val isReady: Boolean = false,
    val isGenerating: Boolean = false,
    val supportsThinking: Boolean = false
)
