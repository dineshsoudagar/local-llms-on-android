package com.example.local_llm

import android.content.Context
import android.os.SystemClock
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

data class ActiveChatSnapshot(
    val sessionId: String?,
    val createdAtMillis: Long,
    val turns: List<ChatTurn>
)

class PersistentChatController(
    context: Context,
    private val modelDescriptor: ModelDescriptor
) {

    companion object {
        // Roughly 10 tokens worth of text before live markdown rendering kicks in.
        private const val MARKDOWN_STREAM_CHAR_THRESHOLD = 40
        private const val TABLE_MARKDOWN_UPDATE_WORD_STEP = 50
        private const val TABLE_MARKDOWN_UPDATE_CHAR_STEP = 220
        private val TABLE_SEPARATOR_REGEX = Regex("^\\|?(?:\\s*:?-{3,}:?\\s*\\|)+\\s*:?-{3,}:?\\s*\\|?$")
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val appContext = context.applicationContext
    private val sessionStore = ChatSessionStore(appContext)
    private val backend: ChatBackend
    private val committedTurns = mutableListOf<ChatTurn>()
    private val _state = MutableStateFlow(
        ChatUiState(
            title = "Pocket LLM - ${modelDescriptor.displayName}",
            statusMessage = MODEL_LOADING_STATUS_MESSAGE,
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
    private var lastPublishedMarkdownWordCount: Int = 0
    private var lastPublishedMarkdownTextLength: Int = 0
    private var currentGenerationStartedAtMillis: Long? = null
    private var currentThinkingStartedAtMillis: Long? = null
    private var currentThinkingFinishedAtMillis: Long? = null

    init {
        val modelFileResolver = ModelFileResolver(appContext)
        backend = when (modelDescriptor) {
            is OnnxQwenSpec -> OnnxChatBackend(appContext, modelDescriptor, modelFileResolver)
            is GemmaLiteRtSpec -> GemmaLiteRtBackend(appContext, modelDescriptor, modelFileResolver)
            is QwenLiteRtSpec -> QwenLiteRtBackend(appContext, modelDescriptor, modelFileResolver)
        }
    }

    fun initialize(activeChatSnapshot: ActiveChatSnapshot? = null) {
        val snapshotToRestore = activeChatSnapshot?.takeIf { it.turns.isNotEmpty() }
        if (snapshotToRestore != null) {
            restoreActiveChat(snapshotToRestore)
            publishState(isLoading = true, isReady = false)
        }

        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    backend.initialize()
                    if (snapshotToRestore != null) {
                        backend.resetConversation(snapshotToRestore.turns, thinkingEnabled)
                    } else {
                        resetConversationForFreshSession()
                    }
                }

                publishState(
                    statusMessage = MODEL_READY_STATUS_MESSAGE,
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

    fun isThinkingEnabled(): Boolean {
        return thinkingEnabled
    }

    fun snapshotActiveChat(): ActiveChatSnapshot {
        return ActiveChatSnapshot(
            sessionId = currentSessionId,
            createdAtMillis = currentSessionCreatedAtMillis,
            turns = committedTurns.toList()
        )
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
        startGenerationTimer()
        liveMarkdownEnabled = false
        lastPublishedMarkdownWordCount = 0
        lastPublishedMarkdownTextLength = 0
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
                            if (!shouldPublishStreamingUpdate(partial)) {
                                return@partialCallback
                            }
                            scope.launch {
                                if (generationId != currentGenerationId) {
                                    return@launch
                                }

                                streamingAssistantTurn = (streamingAssistantTurn
                                    ?: ChatTurn(role = ChatRole.ASSISTANT, text = "")).copy(
                                    text = partial.text,
                                    thinkingText = partial.thinkingText,
                                    renderAsMarkdown = liveMarkdownEnabled
                                )
                                updateThinkingTimer(partial)
                                publishState(isGenerating = true)
                            }
                        }
                    )
                }

                if (generationId != currentGenerationId) {
                    return@launch
                }

                updateThinkingTimer(response)
                val finalThinkingText = response.thinkingText
                    ?: streamingAssistantTurn?.thinkingText
                val finalAssistantTurn = (streamingAssistantTurn
                    ?: ChatTurn(role = ChatRole.ASSISTANT, text = response.text)).copy(
                    text = response.text,
                    thinkingText = finalThinkingText,
                    thinkingDurationMillis = thinkingDurationMillis(finalThinkingText),
                    stopped = false,
                    renderAsMarkdown = true
                )

                if (finalAssistantTurn.text.isNotBlank() || !finalAssistantTurn.thinkingText.isNullOrBlank()) {
                    committedTurns += finalAssistantTurn
                    persistCurrentSession()
                }

                currentGenerationId = 0L
                streamingAssistantTurn = null
                liveMarkdownEnabled = false
                lastPublishedMarkdownWordCount = 0
                lastPublishedMarkdownTextLength = 0
                resetGenerationTimer()
                publishState(isGenerating = false)
            } catch (_: CancellationException) {
                if (generationId == currentGenerationId) {
                    currentGenerationId = 0L
                }
                commitStoppedAssistantTurn()
                liveMarkdownEnabled = false
                lastPublishedMarkdownWordCount = 0
                lastPublishedMarkdownTextLength = 0
                resetGenerationTimer()
                publishState(statusMessage = "Generation stopped.", isGenerating = false)
            } catch (e: Exception) {
                if (generationId == currentGenerationId) {
                    currentGenerationId = 0L
                }
                streamingAssistantTurn = null
                liveMarkdownEnabled = false
                lastPublishedMarkdownWordCount = 0
                lastPublishedMarkdownTextLength = 0
                resetGenerationTimer()
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
            resetGenerationTimer()
            lastPublishedMarkdownWordCount = 0
            lastPublishedMarkdownTextLength = 0

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

    private fun restoreActiveChat(snapshot: ActiveChatSnapshot) {
        currentSessionId = snapshot.sessionId
        currentSessionCreatedAtMillis = snapshot.createdAtMillis
        committedTurns.clear()
        committedTurns += snapshot.turns
        streamingAssistantTurn = null
        liveMarkdownEnabled = false
        currentGenerationId = 0L
        resetGenerationTimer()
        lastPublishedMarkdownWordCount = 0
        lastPublishedMarkdownTextLength = 0
    }

    private fun clearActiveChatState() {
        committedTurns.clear()
        streamingAssistantTurn = null
        liveMarkdownEnabled = false
        currentGenerationId = 0L
        currentSessionId = null
        currentSessionCreatedAtMillis = 0L
        resetGenerationTimer()
        lastPublishedMarkdownWordCount = 0
        lastPublishedMarkdownTextLength = 0
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
                thinkingDurationMillis = thinkingDurationMillis(partialTurn.thinkingText),
                stopped = true,
                renderAsMarkdown = true
            )
            persistCurrentSession()
        }
        streamingAssistantTurn = null
        liveMarkdownEnabled = false
        currentGenerationId = 0L
        resetGenerationTimer()
        lastPublishedMarkdownWordCount = 0
        lastPublishedMarkdownTextLength = 0
    }

    private fun startGenerationTimer() {
        val now = SystemClock.elapsedRealtime()
        currentGenerationStartedAtMillis = now
        currentThinkingStartedAtMillis = null
        currentThinkingFinishedAtMillis = null
    }

    private fun updateThinkingTimer(response: BackendResponse) {
        val now = SystemClock.elapsedRealtime()
        if (!response.thinkingText.isNullOrBlank() && currentThinkingStartedAtMillis == null) {
            currentThinkingStartedAtMillis = currentGenerationStartedAtMillis ?: now
        }
        if (response.text.isNotBlank() && currentThinkingStartedAtMillis != null && currentThinkingFinishedAtMillis == null) {
            currentThinkingFinishedAtMillis = now
        }
    }

    private fun thinkingDurationMillis(thinkingText: String?): Long? {
        if (thinkingText.isNullOrBlank()) {
            return null
        }

        val start = currentThinkingStartedAtMillis ?: currentGenerationStartedAtMillis ?: return null
        val end = currentThinkingFinishedAtMillis ?: SystemClock.elapsedRealtime()
        return (end - start).coerceAtLeast(0L)
    }

    private fun resetGenerationTimer() {
        currentGenerationStartedAtMillis = null
        currentThinkingStartedAtMillis = null
        currentThinkingFinishedAtMillis = null
    }

    private fun shouldEnableLiveMarkdown(partial: BackendResponse): Boolean {
        return partial.text.length >= MARKDOWN_STREAM_CHAR_THRESHOLD
    }

    private fun shouldPublishStreamingUpdate(partial: BackendResponse): Boolean {
        if (!liveMarkdownEnabled || !containsMarkdownTable(partial.text)) {
            lastPublishedMarkdownWordCount = countWords(partial.text)
            lastPublishedMarkdownTextLength = partial.text.length
            return true
        }

        val wordCount = countWords(partial.text)
        val wordDelta = wordCount - lastPublishedMarkdownWordCount
        val charDelta = partial.text.length - lastPublishedMarkdownTextLength
        val shouldPublish = lastPublishedMarkdownTextLength == 0 ||
            wordDelta >= TABLE_MARKDOWN_UPDATE_WORD_STEP ||
            charDelta >= TABLE_MARKDOWN_UPDATE_CHAR_STEP

        if (shouldPublish) {
            lastPublishedMarkdownWordCount = wordCount
            lastPublishedMarkdownTextLength = partial.text.length
        }

        return shouldPublish
    }

    private fun containsMarkdownTable(text: String): Boolean {
        val lines = text
            .replace("\r\n", "\n")
            .lineSequence()
            .toList()

        return lines.zipWithNext().any { (current, next) ->
            current.contains('|') && TABLE_SEPARATOR_REGEX.matches(next.trim())
        }
    }

    private fun countWords(text: String): Int {
        return Regex("\\S+").findAll(text).count()
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
