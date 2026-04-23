package com.example.local_llm

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext

class GemmaLiteRtBackend(
    private val context: Context,
    private val spec: GemmaLiteRtSpec,
    private val modelFileResolver: ModelFileResolver
) : ChatBackend {

    companion object {
        private const val THOUGHT_CHANNEL_NAME = "thought"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)
        val modelPath = modelFile.absolutePath

        engine = createInitializedEngine(modelPath, Backend.GPU(), Backend.GPU())
            ?: createInitializedEngine(modelPath, Backend.GPU(), Backend.CPU())
            ?: createInitializedEngine(modelPath, Backend.CPU(), Backend.CPU())
            ?: throw IllegalStateException("Failed to initialize Gemma LiteRT-LM with GPU or CPU backends.")
    }

    override suspend fun resetConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String
    ) {
        recreateConversation(history, thinkingEnabled, modelInstruction)
    }

    override suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String,
        imageFilePaths: List<String>,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(imageFilePaths.isEmpty() || spec.directImageInputAvailable) {
            "This Gemma model does not support direct image input."
        }
        require(history.isNotEmpty() && history.last().role == ChatRole.USER) {
            "Gemma backend expects the final history turn to be the user's prompt."
        }

        val initialHistory = history.dropLast(1)
        val userTurn = history.last()
        recreateConversation(initialHistory, thinkingEnabled, modelInstruction)

        val activeConversation = conversation
            ?: throw IllegalStateException("Conversation was not created.")

        val textBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()

        val messageForModel = buildUserMessage(userTurn.text, imageFilePaths)
        activeConversation.sendMessageAsync(messageForModel).collect { message ->
            val chunkText = extractTextContent(message)
            if (chunkText.isNotEmpty()) {
                textBuilder.append(chunkText)
            }

            val thoughtChunk = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
            if (thoughtChunk.isNotEmpty()) {
                thinkingBuilder.append(thoughtChunk)
            }

            onPartial(
                BackendResponse(
                    text = textBuilder.toString(),
                    thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() }
                )
            )
        }

        BackendResponse(
            text = textBuilder.toString(),
            thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() }
        )
    }

    private fun createInitializedEngine(
        modelPath: String,
        backend: Backend,
        visionBackend: Backend
    ): Engine? {
        return runCatching {
            Engine(
                EngineConfig(
                    modelPath = modelPath,
                    backend = backend,
                    visionBackend = visionBackend,
                    cacheDir = context.cacheDir.absolutePath
                )
            ).apply { initialize() }
        }.getOrNull()
    }

    override fun cancelGeneration() {
        conversation?.cancelProcess()
    }

    override fun close() {
        closeConversation()
        if (::engine.isInitialized && engine.isInitialized()) {
            engine.close()
        }
    }

    private fun recreateConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String
    ) {
        closeConversation()
        conversation = engine.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(buildSystemInstruction(thinkingEnabled, modelInstruction)),
                initialMessages = history.map { turn ->
                    when (turn.role) {
                        ChatRole.USER -> Message.user(turn.text)
                        ChatRole.ASSISTANT -> Message.model(turn.text)
                    }
                },
                channels = if (thinkingEnabled) null else emptyList()
            )
        )
    }

    private fun buildSystemInstruction(thinkingEnabled: Boolean, modelInstruction: String): String {
        return if (thinkingEnabled) {
            "<|think|>\n$modelInstruction"
        } else {
            modelInstruction
        }
    }

    private fun buildUserMessage(text: String, imageFilePaths: List<String>): Message {
        if (imageFilePaths.isEmpty()) {
            return Message.user(text)
        }

        val textContent = text.ifBlank { "Describe the attached image." }
        val contents = buildList<Content> {
            imageFilePaths.forEach { path -> add(Content.ImageFile(path)) }
            add(Content.Text(textContent))
        }
        return Message.user(Contents.of(*contents.toTypedArray()))
    }

    private fun extractTextContent(message: Message): String {
        val text = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString(separator = "") { content -> content.text }
        return text.ifBlank { message.contents.toString() }
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }
}
