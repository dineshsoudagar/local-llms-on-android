package com.example.local_llm

import java.util.UUID

enum class ChatRole {
    USER,
    ASSISTANT
}

enum class ChatTurnContentType {
    TEXT,
    IMAGE
}

data class ChatTurn(
    val id: String = UUID.randomUUID().toString(),
    val role: ChatRole,
    val text: String,
    val displayText: String? = null,
    val preResponseStatusText: String? = null,
    val thinkingText: String? = null,
    val thinkingDurationMillis: Long? = null,
    val stopped: Boolean = false,
    val renderAsMarkdown: Boolean = false,
    val isStreaming: Boolean = false,
    val contentType: ChatTurnContentType = ChatTurnContentType.TEXT,
    val imagePath: String? = null
) {
    val isUser: Boolean
        get() = role == ChatRole.USER

    val isImage: Boolean
        get() = contentType == ChatTurnContentType.IMAGE

    val transcriptText: String
        get() = if (isImage) "" else displayText ?: text
}

fun ChatTurn.asModelMemoryTurn(): ChatTurn? {
    if (isImage) {
        return null
    }

    return copy(
        displayText = null,
        preResponseStatusText = null,
        thinkingText = null,
        thinkingDurationMillis = null,
        stopped = false,
        renderAsMarkdown = false,
        isStreaming = false,
        contentType = ChatTurnContentType.TEXT,
        imagePath = null
    )
}

fun List<ChatTurn>.asModelMemoryTurns(): List<ChatTurn> {
    return mapNotNull { it.asModelMemoryTurn() }
}
