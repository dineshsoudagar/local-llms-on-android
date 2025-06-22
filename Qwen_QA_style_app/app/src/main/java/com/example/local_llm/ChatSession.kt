package com.example.local_llm

data class ChatSession(
    val sessionId: String,
    val title: String,
    val messages: MutableList<Message>,
    val tokenHistory: List<Int> = emptyList()
)