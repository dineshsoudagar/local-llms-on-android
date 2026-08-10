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
import java.io.File
import java.util.UUID

data class ActiveChatSnapshot(
    val sessionId: String?,
    val createdAtMillis: Long,
    val turns: List<ChatTurn>,
    val activeAttachmentId: String? = null
)

class PersistentChatController(
    context: Context,
    private val modelDescriptor: ModelDescriptor,
    private val initializationPolicy: BackendInitializationPolicy = BackendInitializationPolicy()
) {

    companion object {
        // Roughly 10 tokens worth of text before live markdown rendering kicks in.
        private const val MARKDOWN_STREAM_CHAR_THRESHOLD = 40
        private const val TABLE_MARKDOWN_UPDATE_WORD_STEP = 50
        private const val TABLE_MARKDOWN_UPDATE_CHAR_STEP = 220
        private const val MATH_MARKDOWN_UPDATE_WORD_STEP = 24
        private const val MATH_MARKDOWN_UPDATE_CHAR_STEP = 180
        private const val MATH_MARKDOWN_UPDATE_MIN_INTERVAL_MS = 650L
        private val TABLE_SEPARATOR_REGEX = Regex("^\\|?(?:\\s*:?-{3,}:?\\s*\\|)+\\s*:?-{3,}:?\\s*\\|?$")
        private val LATEX_DELIMITER_REGEX = Regex(
            """\\\[|\\\]|\\\(|\\\)|\${'$'}\${'$'}|(?<!\\)\${'$'}(?=\S*[A-Za-z\\_^{}=+\-*/<>])"""
        )
        private val LATEX_COMMAND_REGEX = Regex(
            """\\(?:begin|end|frac|sqrt|sum|prod|int|lim|left|right|cdot|times|div|pm|mp|leq|geq|neq|approx|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|omega|infty)\b"""
        )
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val appContext = context.applicationContext
    private val sessionStore = ChatSessionStore(appContext)
    private val modelInstructionStore = ModelInstructionStore(appContext)
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
    private var initializationJob: Job? = null
    private var streamingAssistantTurn: ChatTurn? = null
    private var thinkingEnabled = false
    private var liveMarkdownEnabled = false
    private var currentGenerationId: Long = 0L
    private var currentSessionId: String? = null
    private var currentSessionCreatedAtMillis: Long = 0L
    private var preparedPromptUserTurnId: String? = null
    private var preparedPromptTurnIds: Set<String> = emptySet()
    private var lastPublishedMarkdownWordCount: Int = 0
    private var lastPublishedMarkdownTextLength: Int = 0
    private var lastPublishedMarkdownAtMillis: Long = 0L
    private var currentGenerationStartedAtMillis: Long? = null
    private var currentThinkingStartedAtMillis: Long? = null
    private var currentThinkingFinishedAtMillis: Long? = null
    private var currentGenerationImageFilePaths: List<String> = emptyList()
    private var currentNativeAudioInputs: List<NativeAudioInput> = emptyList()
    private var currentAttachmentContext: AttachmentContext? = null
    private var activeAttachmentId: String? = null

    init {
        val modelFileResolver = ModelFileResolver(appContext)
        backend = when (modelDescriptor) {
            is OnnxQwenSpec -> OnnxChatBackend(appContext, modelDescriptor, modelFileResolver)
            is GemmaLiteRtSpec -> GemmaLiteRtBackend(appContext, modelDescriptor, modelFileResolver, initializationPolicy)
            is QwenLiteRtSpec -> QwenLiteRtBackend(appContext, modelDescriptor, modelFileResolver, initializationPolicy)
        }
    }

    fun initialize(
        activeChatSnapshot: ActiveChatSnapshot? = null,
        onComplete: (Result<Unit>) -> Unit = {}
    ): Job {
        val snapshotToRestore = activeChatSnapshot?.takeIf {
            it.turns.isNotEmpty() || it.sessionId != null || it.activeAttachmentId != null
        }
        if (snapshotToRestore != null) {
            restoreActiveChat(snapshotToRestore)
            publishState(isLoading = true, isReady = false)
        }

        initializationJob?.cancel()
        return scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    backend.initialize()
                    if (snapshotToRestore != null) {
                        backend.resetConversation(
                            snapshotToRestore.turns.asModelMemoryTurns(),
                            thinkingEnabled,
                            currentModelInstruction()
                        )
                    } else {
                        resetConversationForFreshSession()
                    }
                }

                publishState(
                    statusMessage = MODEL_READY_STATUS_MESSAGE,
                    isLoading = false,
                    isReady = true
                )
                onComplete(Result.success(Unit))
            } catch (_: CancellationException) {
                publishState(statusMessage = "Model loading cancelled.", isLoading = false, isReady = false)
            } catch (error: OutOfMemoryError) {
                runCatching { backend.close() }
                val message = "Not enough memory to initialize ${modelDescriptor.displayName}."
                publishState(statusMessage = "Error: $message", isLoading = false, isReady = false)
                onComplete(Result.failure(IllegalStateException(message, error)))
            } catch (e: Exception) {
                runCatching { backend.close() }
                publishState(
                    statusMessage = "Error: ${e.message ?: "Unknown error."}",
                    isLoading = false,
                    isReady = false
                )
                onComplete(Result.failure(e))
            } finally {
                initializationJob = null
            }
        }.also { initializationJob = it }
    }

    fun cancelInitialization() {
        initializationJob?.cancel(CancellationException("Model loading cancelled."))
        initializationJob = null
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
            turns = committedTurns.toList(),
            activeAttachmentId = activeAttachmentId
        )
    }

    fun ensureSessionId(): String {
        ensureActiveSession()
        return requireNotNull(currentSessionId)
    }

    fun setActiveAttachment(attachmentId: String?) {
        activeAttachmentId = attachmentId
        persistCurrentSession()
        publishState()
    }

    fun restoreAttachmentSession(sessionId: String) {
        if (currentSessionId == null && committedTurns.isEmpty()) {
            currentSessionId = sessionId
            currentSessionCreatedAtMillis = System.currentTimeMillis()
        }
    }

    fun buildAttachmentContext(
        descriptor: AttachmentDescriptor,
        chunks: List<AttachmentChunk>,
        userPrompt: String
    ): AttachmentContext {
        val reservedText = buildString {
            append(currentModelInstruction())
            committedTurns.asModelMemoryTurns().forEach { append('\n').append(it.text) }
            append('\n').append(userPrompt)
            append("\nAttachment: ").append(descriptor.displayName)
        }
        val outputReserve = (backend.capabilities.contextWindowTokens / 4).coerceIn(128, 512)
        val inputBudget = (
            backend.capabilities.contextWindowTokens - backend.estimateTokens(reservedText) - outputReserve
        ).coerceAtLeast(64)
        val measuredChunks = chunks.map { chunk ->
            chunk.copy(estimatedTokens = backend.estimateTokens(chunk.text))
        }
        val task = AttachmentTaskRouter.route(userPrompt)
        val sourceTokens = measuredChunks.sumOf(AttachmentChunk::estimatedTokens)
        require(task != AttachmentTask.SUMMARY || sourceTokens <= AttachmentLimits.MAX_SUMMARY_SOURCE_TOKENS) {
            "Whole-document summaries are limited to 100,000 source tokens."
        }
        require(task != AttachmentTask.TRANSFORMATION || sourceTokens <= AttachmentLimits.MAX_TRANSFORMATION_SOURCE_TOKENS) {
            "Whole-document transformations are limited to 25,000 source tokens."
        }
        val plan = AttachmentPromptPlanner().plan(measuredChunks, userPrompt, inputBudget)
        return AttachmentContext(
            attachmentId = descriptor.id,
            displayName = descriptor.displayName,
            route = descriptor.processingRoute,
            contextText = plan.batches.firstOrNull()?.prompt.orEmpty(),
            sourceRefs = plan.batches.flatMap { it.chunks }.map(AttachmentChunk::source).distinct(),
            task = plan.task,
            promptBatches = plan.batches.map(AttachmentPromptBatch::prompt),
            requiresFinalSynthesis = plan.requiresFinalSynthesis
        )
    }

    suspend fun <T> withRuntimeLease(work: suspend () -> T): T {
        check(generationJob == null && initializationJob == null) {
            "Attachment processing cannot start while model work is active."
        }
        publishState(
            statusMessage = "Releasing the chat model for attachment processing…",
            isLoading = true,
            isReady = false
        )
        withContext(Dispatchers.IO) { backend.close() }
        var workFailure: Throwable? = null
        try {
            return work()
        } catch (error: Throwable) {
            workFailure = error
            throw error
        } finally {
            try {
                withContext(Dispatchers.IO) {
                    backend.initialize()
                    backend.resetConversation(
                        committedTurns.asModelMemoryTurns(),
                        thinkingEnabled,
                        currentModelInstruction()
                    )
                }
                publishState(
                    statusMessage = MODEL_READY_STATUS_MESSAGE,
                    isLoading = false,
                    isReady = true
                )
            } catch (restoreError: Throwable) {
                if (workFailure != null) workFailure.addSuppressed(restoreError) else throw restoreError
                publishState(
                    statusMessage = "Error: the chat model could not be restored after attachment processing.",
                    isLoading = false,
                    isReady = false
                )
            }
        }
    }

    fun beginPromptPreparation(
        displayText: String,
        statusText: String,
        leadingTurns: List<ChatTurn> = emptyList()
    ): Boolean {
        val displayPrompt = PromptPreprocessor.normalize(displayText)
        if (
            (displayPrompt.isEmpty() && leadingTurns.isEmpty()) ||
            generationJob != null ||
            preparedPromptUserTurnId != null ||
            !_state.value.isReady
        ) {
            return false
        }

        ensureActiveSession()
        committedTurns += leadingTurns
        val userTurn = ChatTurn(
            role = ChatRole.USER,
            text = displayPrompt,
            displayText = displayPrompt
        )
        committedTurns += userTurn
        preparedPromptUserTurnId = userTurn.id
        preparedPromptTurnIds = (leadingTurns.map { it.id } + userTurn.id).toSet()
        streamingAssistantTurn = ChatTurn(
            role = ChatRole.ASSISTANT,
            text = "",
            preResponseStatusText = statusText,
            isStreaming = true
        )
        publishState(statusMessage = "")
        return true
    }

    fun cancelPromptPreparation() {
        if (preparedPromptTurnIds.isEmpty()) {
            return
        }
        committedTurns.removeAll { it.id in preparedPromptTurnIds }
        preparedPromptUserTurnId = null
        preparedPromptTurnIds = emptySet()
        streamingAssistantTurn = null
        publishState()
    }

    fun sendPreparedPrompt(
        text: String,
        displayText: String = text,
        imageFilePaths: List<String> = emptyList(),
        nativeAudioInputs: List<NativeAudioInput> = emptyList(),
        attachmentContext: AttachmentContext? = null,
        leadingTurns: List<ChatTurn> = emptyList()
    ): Boolean {
        val preparedUserTurnId = preparedPromptUserTurnId
            ?: return sendPrompt(
                text,
                displayText,
                imageFilePaths,
                nativeAudioInputs,
                attachmentContext,
                leadingTurns
            )
        val prompt = text.trim()
        if (prompt.isEmpty() || generationJob != null || !_state.value.isReady) {
            return false
        }

        val preparedUserTurnIndex = committedTurns.indexOfFirst { it.id == preparedUserTurnId }
        if (preparedUserTurnIndex == -1) {
            committedTurns.removeAll { it.id in preparedPromptTurnIds }
            preparedPromptUserTurnId = null
            preparedPromptTurnIds = emptySet()
            streamingAssistantTurn = null
            return sendPrompt(
                text,
                displayText,
                imageFilePaths,
                nativeAudioInputs,
                attachmentContext,
                leadingTurns
            )
        }

        val displayPrompt = normalizeDisplayPrompt(displayText, prompt)
        committedTurns[preparedUserTurnIndex] = committedTurns[preparedUserTurnIndex].copy(
            text = prompt,
            displayText = displayPrompt
        )
        preparedPromptUserTurnId = null
        preparedPromptTurnIds = emptySet()
        persistCurrentSession()

        return startGeneration(imageFilePaths, nativeAudioInputs, attachmentContext)
    }

    fun sendPrompt(
        text: String,
        displayText: String = text,
        imageFilePaths: List<String> = emptyList(),
        nativeAudioInputs: List<NativeAudioInput> = emptyList(),
        attachmentContext: AttachmentContext? = null,
        leadingTurns: List<ChatTurn> = emptyList()
    ): Boolean {
        val prompt = text.trim()
        if (prompt.isEmpty() || generationJob != null || !_state.value.isReady) {
            return false
        }

        ensureActiveSession()
        committedTurns += leadingTurns
        val displayPrompt = normalizeDisplayPrompt(displayText, prompt)
        committedTurns += ChatTurn(
            role = ChatRole.USER,
            text = prompt,
            displayText = displayPrompt
        )
        preparedPromptUserTurnId = null
        preparedPromptTurnIds = emptySet()
        persistCurrentSession()

        return startGeneration(imageFilePaths, nativeAudioInputs, attachmentContext)
    }

    private fun startGeneration(
        imageFilePaths: List<String> = emptyList(),
        nativeAudioInputs: List<NativeAudioInput> = emptyList(),
        attachmentContext: AttachmentContext? = null
    ): Boolean {
        if (generationJob != null || !_state.value.isReady) {
            return false
        }

        val generationId = currentGenerationId + 1L
        currentGenerationId = generationId
        startGenerationTimer()
        resetLiveMarkdownState()
        streamingAssistantTurn = ChatTurn(role = ChatRole.ASSISTANT, text = "", isStreaming = true)
        publishState(statusMessage = "", isGenerating = true)
        currentGenerationImageFilePaths = imageFilePaths
        currentNativeAudioInputs = nativeAudioInputs
        currentAttachmentContext = attachmentContext

        generationJob = scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    executeInferenceRequest(
                        request = InferenceRequest(
                            history = committedTurns.asModelMemoryTurns(),
                            thinkingEnabled = thinkingEnabled,
                            modelInstruction = currentModelInstruction(),
                            imageFilePaths = currentGenerationImageFilePaths,
                            nativeAudioInputs = currentNativeAudioInputs,
                            attachmentContext = currentAttachmentContext
                        ),
                        onPartial = partialCallback@{ partial ->
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

                                updateThinkingTimer(partial)
                                streamingAssistantTurn = (streamingAssistantTurn
                                    ?: ChatTurn(role = ChatRole.ASSISTANT, text = "", isStreaming = true)).copy(
                                    text = partial.text,
                                    thinkingText = partial.thinkingText,
                                    thinkingDurationMillis = thinkingDurationMillis(partial.thinkingText)
                                        .takeIf { partial.text.isNotBlank() },
                                    renderAsMarkdown = liveMarkdownEnabled,
                                    isStreaming = true
                                )
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
                    renderAsMarkdown = true,
                    isStreaming = false
                )

                if (finalAssistantTurn.text.isNotBlank() || !finalAssistantTurn.thinkingText.isNullOrBlank()) {
                    committedTurns += finalAssistantTurn
                    persistCurrentSession()
                }

                currentGenerationId = 0L
                streamingAssistantTurn = null
                resetLiveMarkdownState()
                resetGenerationTimer()
                publishState(isGenerating = false)
            } catch (_: CancellationException) {
                if (generationId == currentGenerationId) {
                    currentGenerationId = 0L
                }
                commitStoppedAssistantTurn()
                resetLiveMarkdownState()
                resetGenerationTimer()
                publishState(statusMessage = "Generation stopped.", isGenerating = false)
            } catch (e: Exception) {
                if (generationId == currentGenerationId) {
                    currentGenerationId = 0L
                }
                streamingAssistantTurn = null
                resetLiveMarkdownState()
                resetGenerationTimer()
                publishState(
                    statusMessage = "Error: ${e.message ?: "Unknown error."}",
                    isGenerating = false
                )
            } finally {
                deleteTransientImageFiles(currentGenerationImageFilePaths)
                currentAttachmentContext?.temporaryFiles?.forEach { path -> runCatching { File(path).delete() } }
                currentGenerationImageFilePaths = emptyList()
                currentNativeAudioInputs = emptyList()
                currentAttachmentContext = null
                generationJob = null
            }
        }
        return true
    }

    private suspend fun executeInferenceRequest(
        request: InferenceRequest,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse {
        val context = request.attachmentContext
        if (request.nativeAudioInputs.isNotEmpty()) {
            return executeNativeAudioRequest(request, context, onPartial)
        }
        val batches = context?.promptBatches.orEmpty()
        if (batches.size <= 1) {
            val prompt = batches.firstOrNull() ?: context?.contextText
            return backend.streamReply(request.withUserPrompt(prompt), onPartial)
        }

        val results = batches.map { batch ->
            backend.streamReply(
                request.copy(
                    history = request.history.replaceLastUserText(batch),
                    attachmentContext = null
                ),
                onPartial = {}
            ).text
        }
        if (context?.task == AttachmentTask.TRANSFORMATION) {
            val merged = BackendResponse(text = results.joinToString("\n\n"))
            onPartial(merged)
            return merged
        }
        return synthesizeAttachmentResults(request, results, context, onPartial)
    }

    private suspend fun executeNativeAudioRequest(
        request: InferenceRequest,
        context: AttachmentContext?,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse {
        require(backend.supportsNativeAudioInput) {
            "Gemma native audio is unavailable on this device. Audio was not sent to Whisper."
        }
        val originalPrompt = request.history.lastOrNull { it.role == ChatRole.USER }?.text.orEmpty()
        if (request.nativeAudioInputs.size == 1) {
            val input = request.nativeAudioInputs.single()
            require((input.endMillis ?: 0L) - input.startMillis <= AttachmentLimits.GEMMA_MAX_AUDIO_INPUT_MILLIS) {
                "A Gemma native-audio segment may not exceed 30 seconds."
            }
            return backend.streamReply(request, onPartial)
        }

        val notes = request.nativeAudioInputs.mapIndexed { index, input ->
            val segmentPrompt = buildString {
                append("Analyze audio segment ${input.startMillis}-${input.endMillis ?: input.startMillis} ms. ")
                append("Produce a timestamped transcript plus audible-event notes. Preserve the spoken language. ")
                append("This is segment ${index + 1} of ${request.nativeAudioInputs.size}. User request: ")
                append(originalPrompt)
            }
            backend.streamReply(
                request.copy(
                    history = request.history.replaceLastUserText(segmentPrompt),
                    nativeAudioInputs = listOf(input),
                    attachmentContext = null
                ),
                onPartial = {}
            ).text
        }.fold(mutableListOf<String>()) { merged, next ->
            val deduplicated = merged.lastOrNull()?.let { deduplicateSegmentOverlap(it, next) } ?: next
            merged += deduplicated
            merged
        }

        if (context?.task == AttachmentTask.TRANSFORMATION) {
            val merged = BackendResponse(text = notes.joinToString("\n\n"))
            onPartial(merged)
            return merged
        }

        val synthesisInputs = if (context?.task == AttachmentTask.QUESTION) {
            val noteChunks = notes.mapIndexed { index, note ->
                AttachmentChunk(
                    ordinal = index,
                    text = note,
                    source = AttachmentSourceRef(
                        startMillis = request.nativeAudioInputs[index].startMillis,
                        endMillis = request.nativeAudioInputs[index].endMillis
                    ),
                    estimatedTokens = backend.estimateTokens(note)
                )
            }
            val budget = (backend.capabilities.contextWindowTokens / 2).coerceAtLeast(128)
            val relevant = Bm25AttachmentRetriever(noteChunks).retrieve(originalPrompt, budget)
            relevant.map { chunk ->
                val input = request.nativeAudioInputs[chunk.ordinal]
                backend.streamReply(
                    request.copy(
                        history = request.history.replaceLastUserText(
                            "Answer this request from the attached source audio segment ${chunk.source.label()}: $originalPrompt"
                        ),
                        nativeAudioInputs = listOf(input),
                        attachmentContext = null
                    ),
                    onPartial = {}
                ).text
            }
        } else {
            notes
        }
        return synthesizeAttachmentResults(request, synthesisInputs, context, onPartial)
    }

    private suspend fun synthesizeAttachmentResults(
        request: InferenceRequest,
        results: List<String>,
        context: AttachmentContext?,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse {
        val originalPrompt = request.history.lastOrNull { it.role == ChatRole.USER }?.text.orEmpty()
        var level = results.filter(String::isNotBlank)
        while (level.size > 1) {
            val groups = packTextsForSynthesis(level)
            level = groups.mapIndexed { index, group ->
                val synthesisPrompt = buildString {
                    append("Combine these ordered intermediate results faithfully. Preserve source markers and do not invent facts.\n")
                    append("Original user request: ").append(originalPrompt).append("\n")
                    append("Group ${index + 1} of ${groups.size}:\n\n")
                    append(group.joinToString("\n\n"))
                }
                backend.streamReply(
                    request.copy(
                        history = request.history.replaceLastUserText(synthesisPrompt),
                        nativeAudioInputs = emptyList(),
                        attachmentContext = null
                    ),
                    onPartial = if (groups.size == 1) onPartial else { _: BackendResponse -> }
                ).text
            }
        }
        if (level.isEmpty()) throw IllegalStateException("Attachment processing produced no result.")
        val response = BackendResponse(text = level.single())
        onPartial(response)
        return response
    }

    private fun packTextsForSynthesis(texts: List<String>): List<List<String>> {
        val budget = (backend.capabilities.contextWindowTokens / 2).coerceAtLeast(128)
        val groups = mutableListOf<MutableList<String>>()
        var used = 0
        texts.forEach { text ->
            val cost = backend.estimateTokens(text)
            if (groups.isEmpty() || used + cost > budget) {
                groups += mutableListOf<String>()
                used = 0
            }
            groups.last() += text
            used += cost
        }
        if (groups.size == texts.size && texts.size > 1) {
            return texts.chunked(2)
        }
        return groups
    }

    private fun InferenceRequest.withUserPrompt(prompt: String?): InferenceRequest {
        if (prompt.isNullOrBlank()) return this
        return copy(history = history.replaceLastUserText(prompt), attachmentContext = null)
    }

    private fun List<ChatTurn>.replaceLastUserText(text: String): List<ChatTurn> {
        val index = indexOfLast { it.role == ChatRole.USER }
        require(index >= 0) { "Inference history must contain a user turn." }
        return mapIndexed { turnIndex, turn -> if (turnIndex == index) turn.copy(text = text) else turn }
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
                    backend.resetConversation(emptyList(), thinkingEnabled, currentModelInstruction())
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
            activeAttachmentId = session.activeAttachmentId
            committedTurns.clear()
            committedTurns += session.turns
            streamingAssistantTurn = null
            currentGenerationId = 0L
            preparedPromptUserTurnId = null
            preparedPromptTurnIds = emptySet()
            resetGenerationTimer()
            resetLiveMarkdownState()

            try {
                withContext(Dispatchers.IO) {
                    backend.resetConversation(
                        committedTurns.asModelMemoryTurns(),
                        thinkingEnabled,
                        currentModelInstruction()
                    )
                }
                publishState(statusMessage = "Loaded ${session.title}.")
            } catch (e: Exception) {
                publishState(statusMessage = "Error: ${e.message ?: "Unknown error."}")
            }
        }
    }

    fun close() {
        cancelInitialization()
        generationJob?.cancel()
        runCatching { backend.close() }
        scope.cancel()
    }

    private suspend fun resetConversationForFreshSession() {
        clearActiveChatState()
        backend.resetConversation(emptyList(), thinkingEnabled, currentModelInstruction())
    }

    private fun restoreActiveChat(snapshot: ActiveChatSnapshot) {
        currentSessionId = snapshot.sessionId
        currentSessionCreatedAtMillis = snapshot.createdAtMillis
        activeAttachmentId = snapshot.activeAttachmentId
        committedTurns.clear()
        committedTurns += snapshot.turns
        streamingAssistantTurn = null
        currentGenerationId = 0L
        preparedPromptUserTurnId = null
        preparedPromptTurnIds = emptySet()
        deleteTransientImageFiles(currentGenerationImageFilePaths)
        currentGenerationImageFilePaths = emptyList()
        currentNativeAudioInputs = emptyList()
        currentAttachmentContext = null
        resetGenerationTimer()
        resetLiveMarkdownState()
    }

    private fun clearActiveChatState() {
        committedTurns.clear()
        streamingAssistantTurn = null
        currentGenerationId = 0L
        preparedPromptUserTurnId = null
        preparedPromptTurnIds = emptySet()
        currentSessionId = null
        currentSessionCreatedAtMillis = 0L
        activeAttachmentId = null
        resetGenerationTimer()
        resetLiveMarkdownState()
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
            title = buildChatSessionTitle(committedTurns),
            modelId = modelDescriptor.id,
            modelDisplayName = modelDescriptor.displayName,
            createdAtMillis = currentSessionCreatedAtMillis.takeIf { it > 0 } ?: System.currentTimeMillis(),
            updatedAtMillis = System.currentTimeMillis(),
            turns = committedTurns.toList(),
            activeAttachmentId = activeAttachmentId
        )
        sessionStore.save(session)
    }

    private fun commitStoppedAssistantTurn() {
        val partialTurn = streamingAssistantTurn
        if (partialTurn != null && (partialTurn.text.isNotBlank() || !partialTurn.thinkingText.isNullOrBlank())) {
            committedTurns += partialTurn.copy(
                thinkingDurationMillis = thinkingDurationMillis(partialTurn.thinkingText),
                stopped = true,
                renderAsMarkdown = true,
                isStreaming = false
            )
            persistCurrentSession()
        }
        streamingAssistantTurn = null
        currentGenerationId = 0L
        preparedPromptUserTurnId = null
        preparedPromptTurnIds = emptySet()
        resetGenerationTimer()
        resetLiveMarkdownState()
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

    private fun currentModelInstruction(): String {
        return modelInstructionStore.loadInstruction(modelDescriptor)
    }

    private fun normalizeDisplayPrompt(
        displayText: String,
        prompt: String
    ): String? {
        val normalizedDisplay = PromptPreprocessor.normalize(displayText)
        return if (normalizedDisplay == prompt && normalizedDisplay.isNotBlank()) {
            null
        } else {
            normalizedDisplay
        }
    }

    private fun shouldEnableLiveMarkdown(partial: BackendResponse): Boolean {
        return partial.text.length >= MARKDOWN_STREAM_CHAR_THRESHOLD
    }

    private fun shouldPublishStreamingUpdate(partial: BackendResponse): Boolean {
        val hasTable = containsMarkdownTable(partial.text)
        val hasMath = containsLatexMath(partial.text)
        if (!liveMarkdownEnabled || (!hasTable && !hasMath)) {
            lastPublishedMarkdownWordCount = countWords(partial.text)
            lastPublishedMarkdownTextLength = partial.text.length
            lastPublishedMarkdownAtMillis = SystemClock.elapsedRealtime()
            return true
        }

        val now = SystemClock.elapsedRealtime()
        val wordCount = countWords(partial.text)
        val wordDelta = wordCount - lastPublishedMarkdownWordCount
        val charDelta = partial.text.length - lastPublishedMarkdownTextLength
        val elapsedMillis = now - lastPublishedMarkdownAtMillis
        val shouldPublish = if (hasMath) {
            lastPublishedMarkdownTextLength == 0 ||
                wordDelta >= MATH_MARKDOWN_UPDATE_WORD_STEP ||
                charDelta >= MATH_MARKDOWN_UPDATE_CHAR_STEP ||
                elapsedMillis >= MATH_MARKDOWN_UPDATE_MIN_INTERVAL_MS
        } else {
            lastPublishedMarkdownTextLength == 0 ||
                wordDelta >= TABLE_MARKDOWN_UPDATE_WORD_STEP ||
                charDelta >= TABLE_MARKDOWN_UPDATE_CHAR_STEP
        }

        if (shouldPublish) {
            lastPublishedMarkdownWordCount = wordCount
            lastPublishedMarkdownTextLength = partial.text.length
            lastPublishedMarkdownAtMillis = now
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

    private fun containsLatexMath(text: String): Boolean {
        return LATEX_DELIMITER_REGEX.containsMatchIn(text) ||
            LATEX_COMMAND_REGEX.containsMatchIn(text)
    }

    private fun countWords(text: String): Int {
        return Regex("\\S+").findAll(text).count()
    }

    private fun resetLiveMarkdownState() {
        liveMarkdownEnabled = false
        lastPublishedMarkdownWordCount = 0
        lastPublishedMarkdownTextLength = 0
        lastPublishedMarkdownAtMillis = 0L
    }

    private fun deleteTransientImageFiles(paths: List<String>) {
        paths.forEach { path ->
            runCatching { File(path).delete() }
        }
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
            supportsThinking = modelDescriptor.supportsThinking,
            supportsDirectImageInput = backend.supportsDirectImageInput,
            supportsNativeAudioInput = backend.supportsNativeAudioInput,
            activeAttachmentId = activeAttachmentId
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
