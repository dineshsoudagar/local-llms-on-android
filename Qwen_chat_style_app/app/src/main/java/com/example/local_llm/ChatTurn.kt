package com.example.local_llm

enum class ChatRole {
    USER,
    ASSISTANT
}

data class ChatTurn(
    val role: ChatRole,
    val text: String,
    val thinkingText: String? = null,
    val stopped: Boolean = false
) {
    val isUser: Boolean
        get() = role == ChatRole.USER
}
