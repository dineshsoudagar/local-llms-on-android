package com.example.local_llm

import android.content.Context
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

class PersistentChatController(
    context: Context,
    private val modelDescriptor: ModelDescriptor
) {

    companion object {
        // Roughly 10 tokens worth of text before live markdown rendering kicks in.
        private const val MARKDOWN_STREAM_CHAR_THRESHOLD = 40
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val appContext = context.applicationContext
    private val sessionStore = ChatSessionStore(appContext)
    private val backend: ChatBackend
    private val committedTurns = mutableListOf<ChatTurn>()
    private val _state = MutableStateFlow(
        ChatUiState(
            title = "Pocket LLM - ${modelDescriptor.displayName}",
            statusMessage = "Please wait, the model is still loading.",
            isLoading = true,
            supportsThinking = modelDescriptor.supportsThinking
        )
    )

    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var generationJob: Job? = null
    private var streamingAssistantTurn: ChatTurn? = null
    private var thinkingEnabled = false
    private var liveMarkdownEnabled = false
    private var currentGenerationId: Long = 0L
    private var currentSessionId: String? = null
    private var currentSessionCreatedAtMillis: Long = 0L

    init {
        val modelFileResolver = ModelFileResolver(appContext)
        backend = when (modelDescriptor) {
            is OnnxQwenSpec -> OnnxChatBackend(appContext, modelDescriptor, modelFileResolver)
            is GemmaLiteRtSpec -> GemmaLiteRtBackend(appContext, modelDescriptor, modelFileResolver)
            is QwenLiteRtSpec -> QwenLiteRtBackend(appContext, modelDescriptor, modelFileResolver)
        }
    }

    fun initialize() {
        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    backend.initialize()
                    resetConversationForFreshSession()
                }

                publishState(
                    statusMessage = "Model is ready.",
                    isLoading = false,
                    isReady = true
                )
            } catch (e: Exception) {
                publishState(
                    statusMessage = "Error: ${e.message ?: "Unknown error."}",
                    isLoading = false,
                    isReady = false
                )
            }
        }
    }

    fun setThinkingEnabled(enabled: Boolean) {
        thinkingEnabled = enabled
    }

    fun sendPrompt(text: String) {
        val prompt = text.trim()
        if (prompt.isEmpty() || generationJob != null || !_state.value.isReady) {
            return
        }

        ensureActiveSession()
        committedTurns += ChatTurn(role = ChatRole.USER, text = prompt)
        persistCurrentSession()

        val generationId = currentGenerationId + 1L
        currentGenerationId = generationId
        liveMarkdownEnabled = false
        streamingAssistantTurn = ChatTurn(role = ChatRole.ASSISTANT, text = "")
        publishState(statusMessage = "", isGenerating = true)

        generationJob = scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    backend.streamReply(
                        committedTurns.toList(),
                        thinkingEnabled,
                        partialCallback@{ partial ->
                            if (generationId != currentGenerationId) {
                                return@partialCallback
                            }

                            liveMarkdownEnabled = liveMarkdownEnabled || shouldEnableLiveMarkdown(partial)
                            scope.launch {
                                if (generationId != currentGenerationId) {
                                    return@launch
                                }

                                streamingAssistantTurn = (streamingAssistantTurn
                                    ?: ChatTurn(role = ChatRole.ASSISTANT, text = "")).copy(
                                    text = partial.text,
                                    thinkingText = partial.thinkingText.takeIf { partial.text.isBlank() },
                                    renderAsMarkdown = liveMarkdownEnabled
                                )
                                publishState(isGenerating = true)
                            }
                        }
                    )
                }

                if (generationId != currentGenerationId) {
                    return@launch
                }

                val finalAssistantTurn = (streamingAssistantTurn
                    ?: ChatTurn(role = ChatRole.ASSISTANT, text = response.text)).copy(
                    text = response.text,
                    thinkingText = null,
                    stopped = false,
                    renderAsMarkdown = true
                )

                if (finalAssistantTurn.text.isNotBlank()) {
                    committedTurns += finalAssistantTurn
                    persistCurrentSession()
                }

                currentGenerationId = 0L
                streamingAssistantTurn = null
                liveMarkdownEnabled = false
                publishState(isGenerating = false)
            } catch (_: CancellationException) {
                if (generationId == currentGenerationId) {
                    currentGenerationId = 0L
                }
                commitStoppedAssistantTurn()
                liveMarkdownEnabled = false
                publishState(statusMessage = "Generation stopped.", isGenerating = false)
            } catch (e: Exception) {
                if (generationId == currentGenerationId) {
                    currentGenerationId = 0L
                }
                streamingAssistantTurn = null
                liveMarkdownEnabled = false
                publishState(
                    statusMessage = "Error: ${e.message ?: "Unknown error."}",
                    isGenerating = false
                )
            } finally {
                generationJob = null
            }
        }
    }

    fun cancelGeneration() {
        val job = generationJob ?: return
        backend.cancelGeneration()
        job.cancel(CancellationException("Generation stopped by user."))
    }

    fun startNewChat() {
        if (generationJob != null) {
            return
        }

        clearActiveChatState()

        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    backend.resetConversation(emptyList(), thinkingEnabled)
                }
                publishState(statusMessage = "Started a new chat.")
            } catch (e: Exception) {
                publishState(statusMessage = "Error: ${e.message ?: "Unknown error."}")
            }
        }
    }

    fun listSavedSessions(): List<ChatSessionSummary> {
        return sessionStore.list()
    }

    fun deleteSession(sessionId: String): Boolean {
        val deleted = sessionStore.delete(sessionId)
        if (deleted && currentSessionId == sessionId) {
            currentSessionId = null
            currentSessionCreatedAtMillis = 0L
        }
        return deleted
    }

    fun loadSession(sessionId: String) {
        if (generationJob != null) {
            return
        }

        scope.launch {
            val session = withContext(Dispatchers.IO) {
                sessionStore.load(sessionId)
            }

            if (session == null) {
                publishState(statusMessage = "Could not load that chat.")
                return@launch
            }

            currentSessionId = session.sessionId
            currentSessionCreatedAtMillis = session.createdAtMillis
            committedTurns.clear()
            committedTurns += session.turns
            streamingAssistantTurn = null
            liveMarkdownEnabled = false
            currentGenerationId = 0L

            try {
                withContext(Dispatchers.IO) {
                    backend.resetConversation(committedTurns.toList(), thinkingEnabled)
                }
                publishState(statusMessage = "Loaded ${session.title}.")
            } catch (e: Exception) {
                publishState(statusMessage = "Error: ${e.message ?: "Unknown error."}")
            }
        }
    }

    fun close() {
        generationJob?.cancel()
        runCatching { backend.close() }
        scope.cancel()
    }

    private suspend fun resetConversationForFreshSession() {
        clearActiveChatState()
        backend.resetConversation(emptyList(), thinkingEnabled)
    }

    private fun clearActiveChatState() {
        committedTurns.clear()
        streamingAssistantTurn = null
        liveMarkdownEnabled = false
        currentGenerationId = 0L
        currentSessionId = null
        currentSessionCreatedAtMillis = 0L
    }

    private fun ensureActiveSession() {
        if (currentSessionId != null) {
            return
        }

        currentSessionId = UUID.randomUUID().toString()
        currentSessionCreatedAtMillis = System.currentTimeMillis()
    }

    private fun persistCurrentSession() {
        val sessionId = currentSessionId ?: return
        if (committedTurns.isEmpty()) {
            return
        }

        val session = PersistedChatSession(
            sessionId = sessionId,
            title = buildSessionTitle(),
            modelId = modelDescriptor.id,
            modelDisplayName = modelDescriptor.displayName,
            createdAtMillis = currentSessionCreatedAtMillis.takeIf { it > 0 } ?: System.currentTimeMillis(),
            updatedAtMillis = System.currentTimeMillis(),
            turns = committedTurns.toList()
        )
        sessionStore.save(session)
    }

    private fun buildSessionTitle(): String {
        val firstUserPrompt = committedTurns.firstOrNull { it.isUser }?.text.orEmpty()
        val compactPrompt = firstUserPrompt
            .lineSequence()
            .joinToString(" ")
            .trim()
            .replace(Regex("\\s+"), " ")

        return when {
            compactPrompt.isBlank() -> "Untitled chat"
            compactPrompt.length <= 42 -> compactPrompt
            else -> compactPrompt.take(42).trimEnd() + "..."
        }
    }

    private fun commitStoppedAssistantTurn() {
        val partialTurn = streamingAssistantTurn
        if (partialTurn != null && partialTurn.text.isNotBlank()) {
            committedTurns += partialTurn.copy(
                thinkingText = null,
                stopped = true,
                renderAsMarkdown = true
            )
            persistCurrentSession()
        }
        streamingAssistantTurn = null
        liveMarkdownEnabled = false
        currentGenerationId = 0L
    }

    private fun shouldEnableLiveMarkdown(partial: BackendResponse): Boolean {
        return partial.text.length >= MARKDOWN_STREAM_CHAR_THRESHOLD
    }

    private fun publishState(
        statusMessage: String = _state.value.statusMessage,
        isLoading: Boolean = _state.value.isLoading,
        isReady: Boolean = _state.value.isReady,
        isGenerating: Boolean = _state.value.isGenerating
    ) {
        _state.value = ChatUiState(
            title = "Pocket LLM - ${modelDescriptor.displayName}",
            transcript = buildTranscript(),
            statusMessage = statusMessage,
            isLoading = isLoading,
            isReady = isReady,
            isGenerating = isGenerating,
            supportsThinking = modelDescriptor.supportsThinking
        )
    }

    private fun buildTranscript(): List<ChatTurn> {
        return if (streamingAssistantTurn == null) {
            committedTurns.toList()
        } else {
            committedTurns + listOfNotNull(streamingAssistantTurn)
        }
    }
}
