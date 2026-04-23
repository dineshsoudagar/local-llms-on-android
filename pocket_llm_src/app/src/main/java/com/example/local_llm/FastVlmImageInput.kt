package com.example.local_llm

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
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
        private const val RAW_OUTPUT_PREVIEW_LENGTH = 700
        private const val IMAGE_DESCRIPTION_PROMPT =
            "Describe the attached image in plain text. Mention visible text, key objects, people, layout, and details useful for answering the user's message. Do not output chat template markers."
        private val IMAGE_DESCRIPTION_SAMPLER_CONFIG = SamplerConfig(
            topK = 40,
            topP = 0.95,
            temperature = 0.2,
            seed = 17
        )
        private val TURN_MARKER_REGEX = Regex(
            "<start_of_turn>\\s*(user|model|assistant)?|<end_of_turn>|" +
                "<\\|im_start\\|>\\s*(user|model|assistant)?|<\\|im_end\\|>|</?s>|<bos>|<eos>",
            RegexOption.IGNORE_CASE
        )

        private fun rawPreview(rawOutput: String): String {
            val normalized = rawOutput
                .replace("\r\n", "\n")
                .replace("\n", " ")
                .trim()
                .ifBlank { "<empty>" }
            return normalized.take(RAW_OUTPUT_PREVIEW_LENGTH)
        }
    }

    private enum class EngineMode(val label: String) {
        CPU_WITH_GPU_VISION("CPU+GPU vision"),
        CPU_ONLY("CPU only")
    }

    private val appContext = context.applicationContext
    private val inferenceMutex = Mutex()
    private var engine: Engine? = null
    private var engineMode: EngineMode? = null

    suspend fun describeImageFile(imageFile: File): String = withContext(Dispatchers.IO) {
        require(imageFile.exists() && imageFile.length() > 0L) {
            "Could not read that image."
        }

        inferenceMutex.withLock {
            try {
                withTimeout(IMAGE_DESCRIPTION_TIMEOUT_MS) {
                    coroutineContext.ensureActive()
                    val primaryResult = runDescriptionAttempt(
                        imageFile = imageFile,
                        preferredMode = EngineMode.CPU_WITH_GPU_VISION
                    )
                    if (primaryResult.hasUsableDescription()) {
                        return@withTimeout primaryResult.sanitizedOutput
                    }

                    if (
                        primaryResult.engineMode != EngineMode.CPU_ONLY &&
                        primaryResult.shouldRetryWithCpuOnly()
                    ) {
                        Log.w(
                            TAG,
                            "FastVLM produced degenerate output with ${primaryResult.engineMode.label}; retrying CPU-only vision. " +
                                "reason=${primaryResult.degenerationReason()}"
                        )
                        closeEngine()
                        val fallbackResult = runDescriptionAttempt(
                            imageFile = imageFile,
                            preferredMode = EngineMode.CPU_ONLY
                        )
                        if (fallbackResult.hasUsableDescription()) {
                            return@withTimeout fallbackResult.sanitizedOutput
                        }

                        throw FastVlmNoUsableOutputException(
                            "Primary (${primaryResult.engineMode.label}): ${primaryResult.rawOutput.toRawPreview()}\n" +
                                "Primary reason: ${primaryResult.degenerationReason()}\n" +
                                "Fallback (${fallbackResult.engineMode.label}): ${fallbackResult.rawOutput.toRawPreview()}\n" +
                                "Fallback reason: ${fallbackResult.degenerationReason()}"
                        )
                    }

                    throw FastVlmNoUsableOutputException(primaryResult.rawOutput)
                }
            } catch (error: TimeoutCancellationException) {
                throw FastVlmTimedOutException(IMAGE_DESCRIPTION_TIMEOUT_MS / 1000L, error)
            }
        }
    }

    fun close() {
        closeEngine()
    }

    private suspend fun runDescriptionAttempt(
        imageFile: File,
        preferredMode: EngineMode
    ): FastVlmAttemptResult {
        val activeEngine = ensureEngine(preferredMode)
        val activeMode = engineMode ?: preferredMode
        val conversation = activeEngine.createConversation(
            ConversationConfig(
                samplerConfig = IMAGE_DESCRIPTION_SAMPLER_CONFIG,
                automaticToolCalling = false
            )
        )
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
                val chunkText = extractTextContent(partial)
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
        Log.d(TAG, "Raw FastVLM output (${activeMode.label}): ${rawOutput.toRawPreview()}")
        Log.d(TAG, "Sanitized FastVLM output (${activeMode.label}): ${sanitizedOutput.ifBlank { "<empty>" }}")
        return FastVlmAttemptResult(
            engineMode = activeMode,
            rawOutput = rawOutput,
            sanitizedOutput = sanitizedOutput
        )
    }

    private fun closeEngine() {
        val currentEngine = engine ?: return
        runCatching {
            if (currentEngine.isInitialized()) {
                currentEngine.close()
            }
        }
        engine = null
        engineMode = null
    }

    private fun ensureEngine(preferredMode: EngineMode): Engine {
        engine?.takeIf { it.isInitialized() }?.let { currentEngine ->
            val currentMode = engineMode
            if (currentMode == preferredMode || currentMode == EngineMode.CPU_ONLY) {
                return currentEngine
            }

            closeEngine()
        }

        val modelFile = modelFileResolver.resolveModelFile(spec)
        if (preferredMode == EngineMode.CPU_ONLY) {
            engine = createInitializedEngine(
                modelPath = modelFile.absolutePath,
                mode = EngineMode.CPU_ONLY
            )
            engineMode = EngineMode.CPU_ONLY
            return engine ?: error("FastVLM engine was not created.")
        }

        val recommendedResult = runCatching {
            createInitializedEngine(
                modelPath = modelFile.absolutePath,
                mode = EngineMode.CPU_WITH_GPU_VISION
            )
        }
        engine = recommendedResult.getOrElse { recommendedError ->
            runCatching {
                createInitializedEngine(
                    modelPath = modelFile.absolutePath,
                    mode = EngineMode.CPU_ONLY
                )
            }.getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize FastVLM CPU+GPU (${recommendedError.message}) and CPU (${cpuError.message}).",
                    cpuError
                )
            }.also {
                engineMode = EngineMode.CPU_ONLY
            }
        }.also {
            if (engineMode == null) {
                engineMode = EngineMode.CPU_WITH_GPU_VISION
            }
        }

        return engine ?: error("FastVLM engine was not created.")
    }

    private fun createInitializedEngine(
        modelPath: String,
        mode: EngineMode
    ): Engine {
        val candidate = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = Backend.CPU(),
                visionBackend = when (mode) {
                    EngineMode.CPU_WITH_GPU_VISION -> Backend.GPU()
                    EngineMode.CPU_ONLY -> Backend.CPU()
                },
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

    private fun extractTextContent(message: Message): String {
        val text = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString(separator = "") { content -> content.text }
        return text.ifBlank { message.contents.toString() }
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
            .replace(TURN_MARKER_REGEX, "\n")
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

    private fun String.isOnlyTurnMarkers(): Boolean {
        return replace(TURN_MARKER_REGEX, "")
            .replace(IMAGE_DESCRIPTION_PROMPT, "")
            .trim()
            .isBlank()
    }

    private fun FastVlmAttemptResult.hasUsableDescription(): Boolean {
        return sanitizedOutput.isNotBlank() && degenerationReason() == null
    }

    private fun FastVlmAttemptResult.shouldRetryWithCpuOnly(): Boolean {
        return degenerationReason() != null
    }

    private fun FastVlmAttemptResult.degenerationReason(): String? {
        if (rawOutput.isOnlyTurnMarkers()) {
            return "turn markers only"
        }

        val markerCount = TURN_MARKER_REGEX.findAll(rawOutput).count()
        val words = sanitizedOutput
            .lowercase()
            .split(Regex("[^\\p{L}\\p{N}]+"))
            .filter { it.length >= 2 }

        if (words.isEmpty()) {
            return "empty text after cleaning"
        }

        val wordCounts = words.groupingBy { it }.eachCount()
        val mostRepeated = wordCounts.maxByOrNull { it.value }
        val repeatedWordCount = mostRepeated?.value ?: 0
        val uniqueWordCount = wordCounts.size
        val repeatedShare = repeatedWordCount.toDouble() / words.size.toDouble()

        if (
            markerCount >= 4 &&
            words.size >= 8 &&
            uniqueWordCount <= 3 &&
            repeatedShare >= 0.7
        ) {
            return "turn-marker loop with repeated '${mostRepeated?.key}'"
        }

        if (
            words.size >= 12 &&
            uniqueWordCount <= 2 &&
            repeatedShare >= 0.85
        ) {
            return "low-diversity repeated '${mostRepeated?.key}'"
        }

        return null
    }

    private fun String.toRawPreview(): String {
        return rawPreview(this)
    }

    private data class FastVlmAttemptResult(
        val engineMode: EngineMode,
        val rawOutput: String,
        val sanitizedOutput: String
    )

    class FastVlmNoUsableOutputException(rawOutput: String) : IllegalStateException(buildMessage(rawOutput)) {
        val rawOutput: String = rawOutput

        companion object {
            private fun buildMessage(rawOutput: String): String {
                return "FastVLM returned no usable description. Raw output: ${rawPreview(rawOutput)}"
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
