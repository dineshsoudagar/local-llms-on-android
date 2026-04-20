package com.example.local_llm

import android.content.Context
import android.os.SystemClock
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

class ChatController(
    context: Context,
    private val modelDescriptor: ModelDescriptor
) {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val backend: ChatBackend
    private val committedTurns = mutableListOf<ChatTurn>()
    private val _state = MutableStateFlow(
        ChatUiState(
            title = "Pocket LLM — ${modelDescriptor.displayName}",
            statusMessage = MODEL_LOADING_STATUS_MESSAGE,
            isLoading = true,
            supportsThinking = modelDescriptor.supportsThinking
        )
    )

    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var generationJob: Job? = null
    private var streamingAssistantTurn: ChatTurn? = null
    private var thinkingEnabled = false
    private var currentGenerationStartedAtMillis: Long? = null
    private var currentThinkingStartedAtMillis: Long? = null
    private var currentThinkingFinishedAtMillis: Long? = null

    init {
        val appContext = context.applicationContext
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
                    backend.resetConversation(emptyList(), thinkingEnabled)
                }
                publishState(
                    statusMessage = MODEL_READY_STATUS_MESSAGE,
                    isLoading = false,
                    isReady = true
                )
            } catch (e: Exception) {
                publishState(
                    statusMessage = "❌ Error: ${e.message ?: "Unknown error."}",
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

        committedTurns += ChatTurn(role = ChatRole.USER, text = prompt)
        streamingAssistantTurn = ChatTurn(role = ChatRole.ASSISTANT, text = "", isStreaming = true)
        startGenerationTimer()
        publishState(statusMessage = "", isGenerating = true)

        generationJob = scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    backend.streamReply(committedTurns.asModelMemoryTurns(), thinkingEnabled) { partial ->
                        scope.launch {
                            updateThinkingTimer(partial)
                            streamingAssistantTurn = ChatTurn(
                                role = ChatRole.ASSISTANT,
                                text = partial.text,
                                thinkingText = partial.thinkingText,
                                thinkingDurationMillis = thinkingDurationMillis(partial.thinkingText)
                                    .takeIf { partial.text.isNotBlank() },
                                isStreaming = true
                            )
                            publishState(isGenerating = true)
                        }
                    }
                }

                updateThinkingTimer(response)
                val finalThinkingText = response.thinkingText
                    ?: streamingAssistantTurn?.thinkingText
                val finalTurn = ChatTurn(
                    role = ChatRole.ASSISTANT,
                    text = response.text,
                    thinkingText = finalThinkingText,
                    thinkingDurationMillis = thinkingDurationMillis(finalThinkingText),
                    isStreaming = false
                )
                if (finalTurn.text.isNotBlank() || !finalTurn.thinkingText.isNullOrBlank()) {
                    committedTurns += finalTurn
                }
                streamingAssistantTurn = null
                resetGenerationTimer()
                publishState(isGenerating = false)
            } catch (_: CancellationException) {
                commitStoppedAssistantTurn()
                withContext(NonCancellable + Dispatchers.IO) {
                    backend.resetConversation(committedTurns.asModelMemoryTurns(), thinkingEnabled)
                }
                resetGenerationTimer()
                publishState(statusMessage = "⛔ Generation stopped.", isGenerating = false)
            } catch (e: Exception) {
                streamingAssistantTurn = null
                resetGenerationTimer()
                publishState(
                    statusMessage = "❌ Error: ${e.message ?: "Unknown error."}",
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

    fun resetConversation() {
        if (generationJob != null) {
            return
        }

        committedTurns.clear()
        streamingAssistantTurn = null
        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    backend.resetConversation(emptyList(), thinkingEnabled)
                }
                publishState(statusMessage = if (_state.value.isReady) MODEL_READY_STATUS_MESSAGE else "")
            } catch (e: Exception) {
                publishState(statusMessage = "❌ Error: ${e.message ?: "Unknown error."}")
            }
        }
    }

    fun close() {
        generationJob?.cancel()
        runCatching { backend.close() }
        scope.cancel()
    }

    private fun commitStoppedAssistantTurn() {
        val partialTurn = streamingAssistantTurn
        if (partialTurn != null && (partialTurn.text.isNotBlank() || !partialTurn.thinkingText.isNullOrBlank())) {
            committedTurns += partialTurn.copy(
                thinkingDurationMillis = thinkingDurationMillis(partialTurn.thinkingText),
                stopped = true,
                isStreaming = false
            )
        }
        streamingAssistantTurn = null
        resetGenerationTimer()
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

    private fun publishState(
        statusMessage: String = _state.value.statusMessage,
        isLoading: Boolean = _state.value.isLoading,
        isReady: Boolean = _state.value.isReady,
        isGenerating: Boolean = _state.value.isGenerating
    ) {
        _state.value = ChatUiState(
            title = _state.value.title,
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
