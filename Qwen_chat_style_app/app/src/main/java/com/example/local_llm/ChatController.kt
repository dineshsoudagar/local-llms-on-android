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
            statusMessage = "⏳ Please wait, the model is still loading.",
            isLoading = true,
            supportsThinking = modelDescriptor.supportsThinking
        )
    )

    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var generationJob: Job? = null
    private var streamingAssistantTurn: ChatTurn? = null
    private var thinkingEnabled = false

    init {
        val appContext = context.applicationContext
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
                    backend.resetConversation(emptyList(), thinkingEnabled)
                }
                publishState(
                    statusMessage = "✅ Model is ready.",
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
        streamingAssistantTurn = ChatTurn(role = ChatRole.ASSISTANT, text = "")
        publishState(statusMessage = "", isGenerating = true)

        generationJob = scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    backend.streamReply(committedTurns.toList(), thinkingEnabled) { partial ->
                        scope.launch {
                            streamingAssistantTurn = ChatTurn(
                                role = ChatRole.ASSISTANT,
                                text = partial.text,
                                thinkingText = partial.thinkingText
                            )
                            publishState(isGenerating = true)
                        }
                    }
                }

                val finalTurn = ChatTurn(
                    role = ChatRole.ASSISTANT,
                    text = response.text,
                    thinkingText = response.thinkingText
                )
                if (finalTurn.text.isNotBlank() || !finalTurn.thinkingText.isNullOrBlank()) {
                    committedTurns += finalTurn
                }
                streamingAssistantTurn = null
                publishState(isGenerating = false)
            } catch (_: CancellationException) {
                commitStoppedAssistantTurn()
                withContext(NonCancellable + Dispatchers.IO) {
                    backend.resetConversation(committedTurns.toList(), thinkingEnabled)
                }
                publishState(statusMessage = "⛔ Generation stopped.", isGenerating = false)
            } catch (e: Exception) {
                streamingAssistantTurn = null
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
                publishState(statusMessage = if (_state.value.isReady) "✅ Model is ready." else "")
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
            committedTurns += partialTurn.copy(stopped = true)
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
