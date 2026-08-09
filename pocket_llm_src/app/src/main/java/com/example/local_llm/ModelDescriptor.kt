package com.example.local_llm

data class ModelDownloadFile(
    val localFileName: String,
    val downloadUrl: String,
    val expectedBytes: Long? = null
)

sealed class ModelDescriptor(
    val id: String,
    val displayName: String,
    val supportsThinking: Boolean,
    val backendLabel: String,
    val sizeLabel: String,
    val deviceRecommendation: String,
    val approxDownloadBytes: Long,
    val downloadFiles: List<ModelDownloadFile>
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
    val thinkingModeAvailable: Boolean = false,
    val downloadSizeLabel: String,
    val recommendationLabel: String,
    val estimatedDownloadBytes: Long,
    val downloadArtifacts: List<ModelDownloadFile>
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable,
    backendLabel = "ONNX",
    sizeLabel = downloadSizeLabel,
    deviceRecommendation = recommendationLabel,
    approxDownloadBytes = estimatedDownloadBytes,
    downloadFiles = downloadArtifacts
)

data class GemmaLiteRtSpec(
    val modelName: String,
    val modelAssetName: String,
    val defaultSystemInstruction: String,
    val displayNameOverride: String = modelName,
    val thinkingModeAvailable: Boolean = false,
    val directImageInputAvailable: Boolean = true,
    val directAudioInputAvailable: Boolean = true,
    val downloadSizeLabel: String,
    val recommendationLabel: String,
    val estimatedDownloadBytes: Long,
    val downloadArtifacts: List<ModelDownloadFile>
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable,
    backendLabel = "LiteRT",
    sizeLabel = downloadSizeLabel,
    deviceRecommendation = recommendationLabel,
    approxDownloadBytes = estimatedDownloadBytes,
    downloadFiles = downloadArtifacts
)

data class QwenLiteRtSpec(
    val modelName: String,
    val modelAssetName: String,
    val defaultSystemInstruction: String,
    val displayNameOverride: String = modelName,
    val thinkingModeAvailable: Boolean = false,
    val downloadSizeLabel: String,
    val recommendationLabel: String,
    val estimatedDownloadBytes: Long,
    val downloadArtifacts: List<ModelDownloadFile>
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable,
    backendLabel = "LiteRT",
    sizeLabel = downloadSizeLabel,
    deviceRecommendation = recommendationLabel,
    approxDownloadBytes = estimatedDownloadBytes,
    downloadFiles = downloadArtifacts
)

object ModelRegistry {
    private const val TOKENIZER_ASSET = "tokenizer.json"
    private const val QWEN_MODEL_ASSET = "model.onnx"
    private const val QWEN_DISPLAY_MAPPING_ASSET = "qwen_token_display_mapping.json"
    private const val QWEN_LITERT_MODEL_ASSET = "Qwen3-0.6B.litertlm"
    private const val DEEPSEEK_R1_QWEN_1_5B_LITERT_MODEL_ASSET =
        "DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm"
    private const val GEMMA_MODEL_ASSET = "gemma-4-E2B-it.litertlm"
    private const val GEMMA_E4B_MODEL_ASSET = "gemma-4-E4B-it.litertlm"
    private const val HF_BASE = "https://huggingface.co"
    private const val DEEPSEEK_R1_QWEN_1_5B_REVISION = "2f8b8ee90d8f93b15305b699e8772b277d074a9a"
    private const val DEEPSEEK_R1_QWEN_1_5B_BYTES = 1_833_451_520L
    private const val GEMMA_E2B_REVISION = "7fa1d78473894f7e736a21d920c3aa80f950c0db"
    private const val GEMMA_E4B_REVISION = "9695417f248178c63a9f318c6e0c56cb917cb837"
    private const val GEMMA_E2B_BYTES = 2_583_085_056L
    private const val GEMMA_E4B_BYTES = 3_654_467_584L

    val qwen25 = OnnxQwenSpec(
        modelName = "Qwen2_5",
        displayNameOverride = "Qwen2.5 0.5B ONNX",
        promptStyle = PromptStyle.QWEN2_5,
        modelAssetName = QWEN_MODEL_ASSET,
        tokenizerAssetName = TOKENIZER_ASSET,
        tokenDisplayMappingAssetName = QWEN_DISPLAY_MAPPING_ASSET,
        eosTokenIds = setOf(151643, 151645),
        numLayers = 24,
        numKvHeads = 2,
        headDim = 64,
        batchSize = 1,
        defaultSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        downloadSizeLabel = "2.0 GB",
        recommendationLabel = "Best for mid to high-end mobiles, full precision",
        estimatedDownloadBytes = 1_997_000_000L,
        downloadArtifacts = listOf(
            ModelDownloadFile(
                localFileName = QWEN_MODEL_ASSET,
                downloadUrl = "$HF_BASE/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/onnx/model.onnx?download=true"
            ),
            ModelDownloadFile(
                localFileName = TOKENIZER_ASSET,
                downloadUrl = "$HF_BASE/onnx-community/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json?download=true"
            )
        )
    )

    val qwen3 = OnnxQwenSpec(
        modelName = "Qwen3",
        displayNameOverride = "Qwen3 0.6B Q4F16 ONNX",
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
        thinkingModeAvailable = true,
        downloadSizeLabel = "592 MB",
        recommendationLabel = "Good for low to mid-range mobiles",
        estimatedDownloadBytes = 592_000_000L,
        downloadArtifacts = listOf(
            ModelDownloadFile(
                localFileName = QWEN_MODEL_ASSET,
                downloadUrl = "$HF_BASE/onnx-community/Qwen3-0.6B-ONNX/resolve/main/onnx/model_q4f16.onnx?download=true"
            ),
            ModelDownloadFile(
                localFileName = TOKENIZER_ASSET,
                downloadUrl = "$HF_BASE/onnx-community/Qwen3-0.6B-ONNX/resolve/main/tokenizer.json?download=true"
            )
        )
    )

    val qwen3LiteRt = QwenLiteRtSpec(
        modelName = "Qwen3_LiteRT",
        modelAssetName = QWEN_LITERT_MODEL_ASSET,
        defaultSystemInstruction = "You are Qwen. a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        displayNameOverride = "Qwen3 0.6B LiteRT",
        thinkingModeAvailable = true,
        downloadSizeLabel = "614 MB",
        recommendationLabel = "Best for low-end mobiles",
        estimatedDownloadBytes = 614_000_000L,
        downloadArtifacts = listOf(
            ModelDownloadFile(
                localFileName = QWEN_LITERT_MODEL_ASSET,
                downloadUrl = "$HF_BASE/litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm?download=true"
            )
        )
    )

    val deepSeekR1DistillQwen15BLiteRt = QwenLiteRtSpec(
        modelName = "DeepSeek_R1_Distill_Qwen_1_5B_LiteRT",
        modelAssetName = DEEPSEEK_R1_QWEN_1_5B_LITERT_MODEL_ASSET,
        defaultSystemInstruction = "You are DeepSeek-R1-Distill-Qwen, a careful reasoning assistant. Think through hard problems before answering, then give the user a clear, direct final answer. Keep simple answers concise.",
        displayNameOverride = "DeepSeek R1 Distill Qwen 1.5B LiteRT",
        thinkingModeAvailable = false,
        downloadSizeLabel = "1.83 GB",
        recommendationLabel = "Optional reasoning model for high-RAM mobiles; slower than Qwen3 0.6B",
        estimatedDownloadBytes = DEEPSEEK_R1_QWEN_1_5B_BYTES,
        downloadArtifacts = listOf(
            ModelDownloadFile(
                localFileName = DEEPSEEK_R1_QWEN_1_5B_LITERT_MODEL_ASSET,
                downloadUrl = "$HF_BASE/litert-community/DeepSeek-R1-Distill-Qwen-1.5B/resolve/$DEEPSEEK_R1_QWEN_1_5B_REVISION/$DEEPSEEK_R1_QWEN_1_5B_LITERT_MODEL_ASSET?download=true",
                expectedBytes = DEEPSEEK_R1_QWEN_1_5B_BYTES
            )
        )
    )

    val gemma4E2B = GemmaLiteRtSpec(
        modelName = "Gemma4_E2B",
        modelAssetName = GEMMA_MODEL_ASSET,
        defaultSystemInstruction = "You are Gemma, a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        displayNameOverride = "Gemma 4 E2B LiteRT",
        thinkingModeAvailable = true,
        downloadSizeLabel = "2.58 GB",
        recommendationLabel = "Best for decent to mid-range mobiles",
        estimatedDownloadBytes = GEMMA_E2B_BYTES,
        downloadArtifacts = listOf(
            ModelDownloadFile(
                localFileName = GEMMA_MODEL_ASSET,
                downloadUrl = "$HF_BASE/litert-community/gemma-4-E2B-it-litert-lm/resolve/$GEMMA_E2B_REVISION/gemma-4-E2B-it.litertlm?download=true",
                expectedBytes = GEMMA_E2B_BYTES
            )
        )
    )

    val gemma4E4B = GemmaLiteRtSpec(
        modelName = "Gemma4_E4B",
        modelAssetName = GEMMA_E4B_MODEL_ASSET,
        defaultSystemInstruction = "You are Gemma, a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful. Be conversational when appropriate, and ask a follow-up question only when needed.",
        displayNameOverride = "Gemma 4 E4B LiteRT",
        thinkingModeAvailable = true,
        downloadSizeLabel = "3.65 GB",
        recommendationLabel = "Best for flagship mobiles",
        estimatedDownloadBytes = GEMMA_E4B_BYTES,
        downloadArtifacts = listOf(
            ModelDownloadFile(
                localFileName = GEMMA_E4B_MODEL_ASSET,
                downloadUrl = "$HF_BASE/litert-community/gemma-4-E4B-it-litert-lm/resolve/$GEMMA_E4B_REVISION/gemma-4-E4B-it.litertlm?download=true",
                expectedBytes = GEMMA_E4B_BYTES
            )
        )
    )

    val all = listOf(gemma4E4B, gemma4E2B, deepSeekR1DistillQwen15BLiteRt, qwen3LiteRt, qwen3, qwen25)

    fun findById(id: String?): ModelDescriptor? {
        if (id.isNullOrBlank()) {
            return null
        }

        return all.firstOrNull { it.id == id }
    }
}

object DownloadableModelRegistry {
    fun findById(id: String?): ModelDescriptor? {
        return ModelRegistry.findById(id)
    }
}

val ModelDescriptor.primaryModelFileName: String
    get() = when (this) {
        is OnnxQwenSpec -> modelAssetName
        is GemmaLiteRtSpec -> modelAssetName
        is QwenLiteRtSpec -> modelAssetName
    }

val ModelDescriptor.defaultInstruction: String
    get() = when (this) {
        is OnnxQwenSpec -> defaultSystemPrompt
        is GemmaLiteRtSpec -> defaultSystemInstruction
        is QwenLiteRtSpec -> defaultSystemInstruction
    }
