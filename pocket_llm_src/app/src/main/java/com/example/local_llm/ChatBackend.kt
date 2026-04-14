package com.example.local_llm

interface ChatBackend : AutoCloseable {
    suspend fun initialize()
    suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean)
    suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse
    fun cancelGeneration()
}
