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
    private lateinit var tokenDisplayMapper: TokenDisplayMapper
    private lateinit var onnxModel: OnnxModel
    private val cancelRequested = AtomicBoolean(false)

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        tokenizer = BpeTokenizer(context, spec.tokenizerAssetName)
        config = spec.toModelConfig(tokenizer)
        promptBuilder = PromptBuilder(tokenizer, config)
        tokenDisplayMapper = TokenDisplayMapper(
            context = context,
            modelName = spec.modelName,
            assetFilename = spec.tokenDisplayMappingAssetName
        )

        val modelFile = modelFileResolver.resolveAssetToFile(spec.modelAssetName)
        onnxModel = OnnxModel(modelFile, config)
    }

    override suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) {
        cancelRequested.set(false)
    }

    override suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        cancelRequested.set(false)
        val coroutineIsActive = { isActive }

        val systemPrompt = buildSystemPrompt(thinkingEnabled)
        val promptTokens = promptBuilder.buildPromptTokens(history, PromptIntent.QA(systemPrompt))
        val responseBuilder = StringBuilder()
        var tokenCounter = 0

        onnxModel.runInferenceStreamingWithPastKV(
            inputIds = promptTokens,
            endTokenIds = config.eosTokenIds,
            shouldStop = { cancelRequested.get() || !coroutineIsActive() },
            onTokenGenerated = { tokenId ->
                val tokenText = if (spec.modelName.startsWith("Qwen", ignoreCase = true)) {
                    tokenDisplayMapper.map(tokenId)
                } else {
                    tokenizer.decodeSingleToken(tokenId)
                }

                val shouldSkip = spec.modelName.equals("qwen3", ignoreCase = true) && tokenCounter < 4
                if (!shouldSkip) {
                    responseBuilder.append(tokenText)
                    onPartial(BackendResponse(responseBuilder.toString()))
                }
                tokenCounter += 1
            }
        )

        BackendResponse(text = responseBuilder.toString())
    }

    override fun cancelGeneration() {
        cancelRequested.set(true)
    }

    override fun close() {
        if (::onnxModel.isInitialized) {
            onnxModel.close()
        }
    }

    private fun buildSystemPrompt(thinkingEnabled: Boolean): String {
        if (!spec.modelName.equals("qwen3", ignoreCase = true) || !spec.thinkingModeAvailable) {
            return spec.defaultSystemPrompt
        }

        val thinkingDirective = if (thinkingEnabled) "/think" else "/no_think"
        return "${spec.defaultSystemPrompt} $thinkingDirective".trim()
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
