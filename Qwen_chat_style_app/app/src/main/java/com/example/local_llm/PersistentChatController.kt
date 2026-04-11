package com.example.local_llm

import android.content.Context
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
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
        private const val MARKDOWN_STREAM_INTERVAL = 60
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
    private var streamingUpdateCount = 0
    private var currentSessionId: String? = null
    private var currentSessionCreatedAtMillis: Long = 0L

    init {
        val modelFileResolver = ModelFileResolver(appContext)
        backend = when (modelDescriptor) {
            is OnnxQwenSpec -> OnnxChatBackend(appContext, modelDescriptor, modelFileResolver)
            is GemmaLiteRtSpec -> GemmaLiteRtBackend(appContext, modelDescriptor, modelFileResolver)
        }
    }

    fun initialize() {
        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    backend.initialize()
                    restoreLatestSessionIfAvailable()
                }

                publishState(
                    statusMessage = if (committedTurns.isEmpty()) {
                        "Model is ready."
                    } else {
                        "Loaded your latest chat."
                    },
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

        streamingUpdateCount = 0
        streamingAssistantTurn = ChatTurn(role = ChatRole.ASSISTANT, text = "")
        publishState(statusMessage = "", isGenerating = true)

        generationJob = scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    backend.streamReply(committedTurns.toList(), thinkingEnabled) { partial ->
                        streamingUpdateCount += 1
                        val shouldRenderMarkdown = streamingUpdateCount % MARKDOWN_STREAM_INTERVAL == 0
                        scope.launch {
                            streamingAssistantTurn = (streamingAssistantTurn
                                ?: ChatTurn(role = ChatRole.ASSISTANT, text = "")).copy(
                                text = partial.text,
                                thinkingText = partial.thinkingText,
                                renderAsMarkdown = shouldRenderMarkdown
                            )
                            publishState(isGenerating = true)
                        }
                    }
                }

                val finalAssistantTurn = (streamingAssistantTurn
                    ?: ChatTurn(role = ChatRole.ASSISTANT, text = response.text)).copy(
                    text = response.text,
                    thinkingText = response.thinkingText,
                    stopped = false,
                    renderAsMarkdown = true
                )

                if (finalAssistantTurn.text.isNotBlank() || !finalAssistantTurn.thinkingText.isNullOrBlank()) {
                    committedTurns += finalAssistantTurn
                    persistCurrentSession()
                }

                streamingAssistantTurn = null
                publishState(isGenerating = false)
            } catch (_: CancellationException) {
                commitStoppedAssistantTurn()
                withContext(NonCancellable + Dispatchers.IO) {
                    backend.resetConversation(committedTurns.toList(), thinkingEnabled)
                }
                publishState(statusMessage = "Generation stopped.", isGenerating = false)
            } catch (e: Exception) {
                streamingAssistantTurn = null
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

        committedTurns.clear()
        streamingAssistantTurn = null
        streamingUpdateCount = 0
        currentSessionId = null
        currentSessionCreatedAtMillis = 0L

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
            streamingUpdateCount = 0

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

    private suspend fun restoreLatestSessionIfAvailable() {
        val latestSession = sessionStore.list()
            .firstOrNull { it.modelId == modelDescriptor.id }
            ?.let { sessionStore.load(it.sessionId) }

        if (latestSession == null) {
            backend.resetConversation(emptyList(), thinkingEnabled)
            return
        }

        currentSessionId = latestSession.sessionId
        currentSessionCreatedAtMillis = latestSession.createdAtMillis
        committedTurns.clear()
        committedTurns += latestSession.turns
        backend.resetConversation(committedTurns.toList(), thinkingEnabled)
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
        if (partialTurn != null && (partialTurn.text.isNotBlank() || !partialTurn.thinkingText.isNullOrBlank())) {
            committedTurns += partialTurn.copy(
                stopped = true,
                renderAsMarkdown = true
            )
            persistCurrentSession()
        }
        streamingAssistantTurn = null
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
