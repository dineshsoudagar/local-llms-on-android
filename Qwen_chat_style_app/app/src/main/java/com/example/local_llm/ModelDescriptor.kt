package com.example.local_llm

sealed class ModelDescriptor(
    val id: String,
    val displayName: String,
    val supportsThinking: Boolean
)

data class OnnxQwenSpec(
    val modelName: String,
    val displayNameOverride: String = modelName,
    val promptStyle: PromptStyle,
    val modelAssetName: String,
    val tokenizerAssetName: String,
    val tokenDisplayMappingAssetName: String?,
    val eosTokenIds: Set<Int>,
    val numLayers: Int,
    val numKvHeads: Int,
    val headDim: Int,
    val batchSize: Int,
    val defaultSystemPrompt: String,
    val scalarPosId: Boolean = false,
    val dtype: String = "float32",
    val thinkingModeAvailable: Boolean = false
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable
)

data class GemmaLiteRtSpec(
    val modelName: String,
    val modelAssetName: String,
    val defaultSystemInstruction: String,
    val displayNameOverride: String = modelName,
    val thinkingModeAvailable: Boolean = false
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable
)

data class QwenLiteRtSpec(
    val modelName: String,
    val modelAssetName: String,
    val defaultSystemInstruction: String,
    val displayNameOverride: String = modelName,
    val thinkingModeAvailable: Boolean = false
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable
)

object ModelRegistry {
    private const val TOKENIZER_ASSET = "tokenizer.json"
    private const val QWEN_MODEL_ASSET = "model.onnx"
    private const val QWEN_DISPLAY_MAPPING_ASSET = "qwen_token_display_mapping.json"
    private const val QWEN_LITERT_MODEL_ASSET = "qwen3.litertlm"
    private const val GEMMA_MODEL_ASSET = "gemma-4-E2B-it.litertlm"
    private const val GEMMA_E4B_MODEL_ASSET = "gemma-4-E4B-it.litertlm"

    val qwen25 = OnnxQwenSpec(
        modelName = "Qwen2_5",
        promptStyle = PromptStyle.QWEN2_5,
        modelAssetName = QWEN_MODEL_ASSET,
        tokenizerAssetName = TOKENIZER_ASSET,
        tokenDisplayMappingAssetName = QWEN_DISPLAY_MAPPING_ASSET,
        eosTokenIds = setOf(151643, 151645),
        numLayers = 24,
        numKvHeads = 2,
        headDim = 64,
        batchSize = 1,
        defaultSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )

    val qwen3 = OnnxQwenSpec(
        modelName = "Qwen3",
        promptStyle = PromptStyle.QWEN3,
        modelAssetName = QWEN_MODEL_ASSET,
        tokenizerAssetName = TOKENIZER_ASSET,
        tokenDisplayMappingAssetName = QWEN_DISPLAY_MAPPING_ASSET,
        eosTokenIds = setOf(151643, 151645),
        numLayers = 28,
        numKvHeads = 8,
        headDim = 128,
        batchSize = 1,
        defaultSystemPrompt = "You are Qwen. a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        scalarPosId = true,
        dtype = "float16",
        thinkingModeAvailable = true
    )

    val qwen3LiteRt = QwenLiteRtSpec(
        modelName = "Qwen3_LiteRT",
        modelAssetName = QWEN_LITERT_MODEL_ASSET,
        defaultSystemInstruction = "You are Qwen. a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        displayNameOverride = "Qwen3 LiteRT",
        thinkingModeAvailable = true
    )

    val gemma4E2B = GemmaLiteRtSpec(
        modelName = "Gemma4_E2B",
        modelAssetName = GEMMA_MODEL_ASSET,
        defaultSystemInstruction = "You are Gemma, a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        displayNameOverride = "Gemma4",
        thinkingModeAvailable = true
    )

    val gemma4E4B = GemmaLiteRtSpec(
        modelName = "Gemma4_E4B",
        modelAssetName = GEMMA_E4B_MODEL_ASSET,
        defaultSystemInstruction = "You are Gemma, a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        displayNameOverride = "Gemma4",
        thinkingModeAvailable = true
    )

    private val models = listOf(qwen25, qwen3, qwen3LiteRt, gemma4E2B, gemma4E4B)

    private const val SELECTED_MODEL_ID = "gemma4_e2b"

    val selected: ModelDescriptor = models.first { it.id == SELECTED_MODEL_ID }
}
