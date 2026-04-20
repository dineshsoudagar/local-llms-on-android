package com.example.local_llm

import java.util.UUID

enum class ChatRole {
    USER,
    ASSISTANT
}

data class ChatTurn(
    val id: String = UUID.randomUUID().toString(),
    val role: ChatRole,
    val text: String,
    val thinkingText: String? = null,
    val thinkingDurationMillis: Long? = null,
    val stopped: Boolean = false,
    val renderAsMarkdown: Boolean = false,
    val isStreaming: Boolean = false
) {
    val isUser: Boolean
        get() = role == ChatRole.USER
}

fun ChatTurn.asModelMemoryTurn(): ChatTurn {
    return copy(
        thinkingText = null,
        thinkingDurationMillis = null,
        stopped = false,
        renderAsMarkdown = false,
        isStreaming = false
    )
}

fun List<ChatTurn>.asModelMemoryTurns(): List<ChatTurn> {
    return map { it.asModelMemoryTurn() }
}
