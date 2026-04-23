package com.example.local_llm

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

class OnnxChatBackend(
    private val context: Context,
    private val spec: OnnxQwenSpec,
    private val modelFileResolver: ModelFileResolver
) : ChatBackend {

    private lateinit var tokenizer: BpeTokenizer
    private lateinit var config: ModelConfig
    private lateinit var promptBuilder: PromptBuilder
    private lateinit var onnxModel: OnnxModel
    private val cancelRequested = AtomicBoolean(false)

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        tokenizer = BpeTokenizer(context, spec, modelFileResolver)
        config = spec.toModelConfig(tokenizer)
        promptBuilder = PromptBuilder(tokenizer, config)

        val modelFile = modelFileResolver.resolveModelFile(spec)
        onnxModel = OnnxModel(modelFile, config)
    }

    override suspend fun resetConversation(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String
    ) {
        cancelRequested.set(false)
    }

    override suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        modelInstruction: String,
        imageFilePaths: List<String>,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(imageFilePaths.isEmpty()) {
            "ONNX chat models do not support direct image input."
        }
        cancelRequested.set(false)
        val coroutineIsActive = { isActive }
        val isQwen3 = spec.modelName.equals("qwen3", ignoreCase = true)

        val systemPrompt = buildSystemPrompt(thinkingEnabled, modelInstruction)
        val promptTokens = promptBuilder.buildPromptTokens(history, PromptIntent.QA(systemPrompt))
        val responseBuilder = StringBuilder()
        val streamDecoder = tokenizer.createStreamDecoder()
        var tokenCounter = 0

        onnxModel.runInferenceStreamingWithPastKV(
            inputIds = promptTokens,
            endTokenIds = config.eosTokenIds,
            shouldStop = { cancelRequested.get() || !coroutineIsActive() },
            onTokenGenerated = { tokenId ->
                val tokenText = streamDecoder.append(tokenId)

                val shouldSkip = isQwen3 && tokenCounter < 4
                if (!shouldSkip) {
                    responseBuilder.append(tokenText)
                    onPartial(parseBackendResponse(responseBuilder.toString(), isQwen3))
                }
                tokenCounter += 1
            }
        )

        val trailingText = streamDecoder.flush()
        if (trailingText.isNotEmpty()) {
            responseBuilder.append(trailingText)
            onPartial(parseBackendResponse(responseBuilder.toString(), isQwen3))
        }

        parseBackendResponse(responseBuilder.toString(), isQwen3)
    }

    override fun cancelGeneration() {
        cancelRequested.set(true)
    }

    override fun close() {
        if (::onnxModel.isInitialized) {
            onnxModel.close()
        }
    }

    private fun buildSystemPrompt(thinkingEnabled: Boolean, modelInstruction: String): String {
        if (!spec.modelName.equals("qwen3", ignoreCase = true) || !spec.thinkingModeAvailable) {
            return modelInstruction
        }

        val thinkingDirective = if (thinkingEnabled) "/think" else "/no_think"
        return "$modelInstruction $thinkingDirective".trim()
    }

    private fun parseBackendResponse(rawOutput: String, isQwen3: Boolean): BackendResponse {
        return if (isQwen3) {
            QwenResponseParser.parseVisibleResponse(rawOutput)
        } else {
            BackendResponse(text = rawOutput)
        }
    }

    private fun OnnxQwenSpec.toModelConfig(tokenizer: BpeTokenizer): ModelConfig {
        val roleTokens = RoleTokenIds(
            systemStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("system"),
                tokenizer.getTokenId("Ċ")
            ),
            userStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("user"),
                tokenizer.getTokenId("Ċ")
            ),
            assistantStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("assistant"),
                tokenizer.getTokenId("Ċ")
            ),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )

        return ModelConfig(
            modelName = modelName,
            modelPath = modelAssetName,
            promptStyle = promptStyle,
            eosTokenIds = eosTokenIds,
            numLayers = numLayers,
            numKvHeads = numKvHeads,
            headDim = headDim,
            batchSize = batchSize,
            defaultSystemPrompt = defaultSystemPrompt,
            roleTokenIds = roleTokens,
            scalarPosId = scalarPosId,
            dtype = dtype,
            IsThinkingModeAvailable = thinkingModeAvailable
        )
    }
}
