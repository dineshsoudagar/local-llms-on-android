package com.example.local_llm

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

class FastVlmImageInput(
    private val context: Context,
    private val spec: FastVlmLiteRtSpec,
    private val modelFileResolver: ModelFileResolver
) {

    companion object {
        private const val IMAGE_DESCRIPTION_PROMPT =
            "Describe this image as context for another assistant. Include visible text, key objects, people, layout, and any details useful for answering the user's message. Keep the description concise and factual."
    }

    private val appContext = context.applicationContext
    private val inferenceMutex = Mutex()
    private var engine: Engine? = null

    suspend fun describeImageFile(imageFile: File): String = withContext(Dispatchers.IO) {
        require(imageFile.exists() && imageFile.length() > 0L) {
            "Could not read that image."
        }

        inferenceMutex.withLock {
            coroutineContext.ensureActive()
            val activeEngine = ensureEngine()
            val conversation = activeEngine.createConversation(ConversationConfig())
            val outputBuilder = StringBuilder()

            try {
                val message = Message.user(
                    Contents.of(
                        Content.ImageFile(imageFile.absolutePath),
                        Content.Text(IMAGE_DESCRIPTION_PROMPT)
                    )
                )

                conversation.sendMessageAsync(message).collect { partial ->
                    coroutineContext.ensureActive()
                    val chunkText = partial.contents.toString()
                    if (chunkText.isNotEmpty()) {
                        outputBuilder.append(chunkText)
                    }
                }
            } finally {
                runCatching { conversation.cancelProcess() }
                runCatching { conversation.close() }
            }

            PromptPreprocessor.normalize(outputBuilder.toString())
        }
    }

    fun close() {
        val currentEngine = engine ?: return
        runCatching {
            if (currentEngine.isInitialized()) {
                currentEngine.close()
            }
        }
        engine = null
    }

    private fun ensureEngine(): Engine {
        engine?.takeIf { it.isInitialized() }?.let { return it }

        val modelFile = modelFileResolver.resolveModelFile(spec)
        val gpuResult = runCatching {
            createInitializedEngine(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU()
            )
        }

        engine = gpuResult.getOrElse { gpuError ->
            runCatching {
                createInitializedEngine(
                    modelPath = modelFile.absolutePath,
                    backend = Backend.CPU(),
                    visionBackend = Backend.CPU()
                )
            }.getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize FastVLM GPU (${gpuError.message}) and CPU (${cpuError.message}).",
                    cpuError
                )
            }
        }

        return engine ?: error("FastVLM engine was not created.")
    }

    private fun createInitializedEngine(
        modelPath: String,
        backend: Backend,
        visionBackend: Backend
    ): Engine {
        val candidate = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = backend,
                visionBackend = visionBackend,
                maxNumImages = 1,
                cacheDir = appContext.cacheDir.absolutePath
            )
        )

        try {
            candidate.initialize()
            return candidate
        } catch (error: Exception) {
            runCatching { candidate.close() }
            throw error
        }
    }
}
