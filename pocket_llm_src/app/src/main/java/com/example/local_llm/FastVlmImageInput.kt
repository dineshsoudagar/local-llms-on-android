package com.example.local_llm

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import kotlin.coroutines.coroutineContext

class FastVlmImageInput(
    private val context: Context,
    private val spec: FastVlmLiteRtSpec,
    private val modelFileResolver: ModelFileResolver
) {

    companion object {
        private const val TAG = "FastVlmImageInput"
        private const val IMAGE_DESCRIPTION_TIMEOUT_MS = 180_000L
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
            try {
                withTimeout(IMAGE_DESCRIPTION_TIMEOUT_MS) {
                    coroutineContext.ensureActive()
                    val activeEngine = ensureEngine()
                    val conversation = activeEngine.createConversation(ConversationConfig())
                    val outputBuilder = StringBuilder()

                    try {
                        @Suppress("DEPRECATION")
                        val message = Message.of(
                            Content.ImageFile(imageFile.absolutePath),
                            Content.Text(IMAGE_DESCRIPTION_PROMPT)
                        )

                        conversation.sendMessageAsync(message).collect { partial ->
                            coroutineContext.ensureActive()
                            val chunkText = partial.contents.toString()
                            if (chunkText.isNotEmpty()) {
                                appendStreamingText(outputBuilder, chunkText)
                            }
                        }
                    } finally {
                        runCatching { conversation.cancelProcess() }
                        runCatching { conversation.close() }
                    }

                    val rawOutput = outputBuilder.toString()
                    val sanitizedOutput = sanitizeFastVlmResponse(rawOutput)
                    Log.d(TAG, "Raw FastVLM output: ${rawOutput.ifBlank { "<empty>" }}")
                    Log.d(TAG, "Sanitized FastVLM output: ${sanitizedOutput.ifBlank { "<empty>" }}")
                    if (sanitizedOutput.isBlank()) {
                        throw FastVlmNoUsableOutputException(rawOutput)
                    }
                    sanitizedOutput
                }
            } catch (error: TimeoutCancellationException) {
                throw FastVlmTimedOutException(IMAGE_DESCRIPTION_TIMEOUT_MS / 1000L, error)
            }
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
        val recommendedResult = runCatching {
            createInitializedEngine(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                visionBackend = Backend.GPU()
            )
        }

        engine = recommendedResult.getOrElse { recommendedError ->
            runCatching {
                createInitializedEngine(
                    modelPath = modelFile.absolutePath,
                    backend = Backend.CPU(),
                    visionBackend = Backend.CPU()
                )
            }.getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize FastVLM CPU+GPU (${recommendedError.message}) and CPU (${cpuError.message}).",
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

    private fun appendStreamingText(builder: StringBuilder, chunkText: String) {
        val currentText = builder.toString()
        if (chunkText.startsWith(currentText) && chunkText.length > currentText.length) {
            builder.clear()
            builder.append(chunkText)
        } else {
            builder.append(chunkText)
        }
    }

    private fun sanitizeFastVlmResponse(rawOutput: String): String {
        val withoutTurnMarkers = rawOutput
            .replace(Regex("<start_of_turn>\\s*(user|model|assistant)?", RegexOption.IGNORE_CASE), "\n")
            .replace(Regex("<end_of_turn>", RegexOption.IGNORE_CASE), "\n")
            .replace(Regex("<\\|im_start\\|>\\s*(user|model|assistant)?", RegexOption.IGNORE_CASE), "\n")
            .replace(Regex("<\\|im_end\\|>", RegexOption.IGNORE_CASE), "\n")
            .replace(Regex("</?s>|<bos>|<eos>", RegexOption.IGNORE_CASE), "\n")
            .replace(IMAGE_DESCRIPTION_PROMPT, "")

        val cleaned = withoutTurnMarkers
            .replace("\r\n", "\n")
            .lineSequence()
            .map { line -> line.trim() }
            .filterNot { line ->
                line.equals("user", ignoreCase = true) ||
                    line.equals("model", ignoreCase = true) ||
                    line.equals("assistant", ignoreCase = true)
            }
            .joinToString("\n")

        return PromptPreprocessor.normalize(cleaned)
            .takeIf { text -> text.any { it.isLetterOrDigit() } }
            .orEmpty()
    }

    class FastVlmNoUsableOutputException(rawOutput: String) : IllegalStateException(buildMessage(rawOutput)) {
        val rawOutput: String = rawOutput

        companion object {
            private const val RAW_OUTPUT_PREVIEW_LENGTH = 700

            private fun buildMessage(rawOutput: String): String {
                return "FastVLM returned no usable description. Raw output: ${rawOutput.toRawPreview()}"
            }

            private fun String.toRawPreview(): String {
                val normalized = replace("\r\n", "\n")
                    .replace("\n", " ")
                    .trim()
                    .ifBlank { "<empty>" }
                return normalized.take(RAW_OUTPUT_PREVIEW_LENGTH)
            }
        }
    }

    class FastVlmTimedOutException(
        timeoutSeconds: Long,
        cause: Throwable
    ) : IllegalStateException(
        "FastVLM did not finish image description within ${timeoutSeconds}s.",
        cause
    )
}
