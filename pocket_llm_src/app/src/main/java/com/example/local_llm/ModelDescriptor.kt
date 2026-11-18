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

data class CustomLiteRtSpec(
    val modelAssetName: String,
    val fileBytes: Long,
    val modelId: String,
    val modelDisplayName: String,
    val detectedInputs: CustomModelCapabilities? = null,
    /** Set for models added from Hugging Face; empty for files imported from the phone. */
    val downloadUrl: String = ""
) : ModelDescriptor(
    id = modelId,
    displayName = modelDisplayName,
    supportsThinking = false,
    backendLabel = "LiteRT (local beta)",
    sizeLabel = "%.2f GB".format(java.util.Locale.US, fileBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)),
    deviceRecommendation = "${detectedInputs?.let { "Detected inputs: ${it.inputSummary()}." } ?: "Input capabilities unknown; text chat will be attempted."} Native image/audio availability depends on this device. Beta compatibility is not guaranteed.",
    approxDownloadBytes = fileBytes,
    downloadFiles = listOf(ModelDownloadFile(modelAssetName, downloadUrl, fileBytes))
)

/** A GGUF model run by the llama.cpp engine (Play edition, Pocket LLM Pro). */
data class GgufSpec(
    val modelName: String,
    val displayNameOverride: String,
    val repoId: String,
    val revision: String,
    val fileName: String,
    val fileBytes: Long,
    val licenseLabel: String,
    val defaultSystemInstruction: String,
    val thinkingModeAvailable: Boolean = false,
    val downloadSizeLabel: String,
    val recommendationLabel: String,
    /** Further parts of a split GGUF (name-00002-of-0000N.gguf...); llama.cpp loads them from the first part. */
    val extraFiles: List<ModelDownloadFile> = emptyList(),
    /** Exact URL of the first file when it sits in a subfolder of the repository. */
    val downloadUrlOverride: String? = null
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = thinkingModeAvailable,
    backendLabel = "GGUF (Pro)",
    sizeLabel = downloadSizeLabel,
    deviceRecommendation = recommendationLabel,
    approxDownloadBytes = fileBytes + extraFiles.sumOf { it.expectedBytes ?: 0L },
    downloadFiles = listOf(
        ModelDownloadFile(
            localFileName = fileName,
            downloadUrl = downloadUrlOverride ?: "https://huggingface.co/$repoId/resolve/$revision/$fileName?download=true",
            expectedBytes = fileBytes
        )
    ) + extraFiles
)

/** Samplers used by stable-diffusion.cpp; [native] is its sample_method_t value. */
enum class ImageSampler(val native: Int) {
    EULER_A(1),
    DPMPP_2M(5),
    LCM(9)
}

/** An image generation model run by stable-diffusion.cpp (Play edition, Pocket LLM Pro). */
data class ImageGenSpec(
    val modelName: String,
    val displayNameOverride: String,
    val modelFile: ModelDownloadFile,
    val decoderFile: ModelDownloadFile? = null,
    val steps: Int,
    /** Highest step count offered in the quality settings; one-step models stay at 1. */
    val maxSteps: Int = steps,
    val cfgScale: Float,
    val licenseLabel: String,
    val minimumRamGb: Int,
    /** First-run guess for a whole generation on a 12 GB / 8 GB / smaller phone, before any measurement. */
    val guessSecondsByTier: Triple<Int, Int, Int>,
    val downloadSizeLabel: String,
    val recommendationLabel: String,
    val modelCardUrl: String? = null,
    val sampler: ImageSampler = ImageSampler.EULER_A
) : ModelDescriptor(
    id = modelName.lowercase(),
    displayName = displayNameOverride,
    supportsThinking = false,
    backendLabel = "Image (Pro)",
    sizeLabel = downloadSizeLabel,
    deviceRecommendation = recommendationLabel,
    approxDownloadBytes = (modelFile.expectedBytes ?: 0L) + (decoderFile?.expectedBytes ?: 0L),
    downloadFiles = listOfNotNull(modelFile, decoderFile)
)

/** Per-generation quality choices from the Create image screen. */
data class ImageGenOptions(
    val steps: Int,
    /** Decode with the model's own VAE instead of the fast TAESD decoder, when the model has one. */
    val fullDecoder: Boolean = false,
    val seed: Long? = null,
    /** Edit mode: RGB bytes (width x height x 3) of the photo to change, and how much to change it (0..1). */
    val initImage: ByteArray? = null,
    val strength: Float = 0.45f,
    val width: Int = 512,
    val height: Int = 512
) {
    fun timingKey(spec: ImageGenSpec): String = if (fullDecoder && spec.decoderFile != null) "${spec.id}_full" else spec.id
}

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

    val qwen35TwoBGguf = GgufSpec(
        modelName = "Qwen3_5_2B_GGUF",
        displayNameOverride = "Qwen3.5 2B GGUF",
        repoId = "unsloth/Qwen3.5-2B-GGUF",
        revision = "f6d5376be1edb4d416d56da11e5397a961aca8ae",
        fileName = "Qwen3.5-2B-Q4_K_M.gguf",
        fileBytes = 1_280_835_840L,
        licenseLabel = "Apache 2.0",
        defaultSystemInstruction = "You are Qwen, a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful.",
        thinkingModeAvailable = true,
        downloadSizeLabel = "1.28 GB",
        recommendationLabel = "Pro. Runs on CPU with llama.cpp; good for mid-range mobiles with 6 GB RAM or more"
    )

    private const val GGUF_ASSISTANT_INSTRUCTION = "You are a helpful personal assistant. Answer clearly, naturally, and in a friendly way. Stay focused on the user's question and avoid unnecessary details. Keep replies concise but useful."

    val qwen35SmallGguf = GgufSpec(
        modelName = "Qwen3_5_0_8B_GGUF",
        displayNameOverride = "Qwen3.5 0.8B GGUF",
        repoId = "unsloth/Qwen3.5-0.8B-GGUF",
        revision = "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
        fileName = "Qwen3.5-0.8B-Q4_K_M.gguf",
        fileBytes = 532_517_120L,
        licenseLabel = "Apache 2.0",
        defaultSystemInstruction = GGUF_ASSISTANT_INSTRUCTION,
        thinkingModeAvailable = true,
        downloadSizeLabel = "533 MB",
        recommendationLabel = "Pro. Small and fast on CPU; good for phones with 4 to 6 GB RAM"
    )

    val qwen35FourBGguf = GgufSpec(
        modelName = "Qwen3_5_4B_GGUF",
        displayNameOverride = "Qwen3.5 4B GGUF",
        repoId = "unsloth/Qwen3.5-4B-GGUF",
        revision = "e87f176479d0855a907a41277aca2f8ee7a09523",
        fileName = "Qwen3.5-4B-Q4_K_M.gguf",
        fileBytes = 2_740_937_888L,
        licenseLabel = "Apache 2.0",
        defaultSystemInstruction = GGUF_ASSISTANT_INSTRUCTION,
        thinkingModeAvailable = true,
        downloadSizeLabel = "2.74 GB",
        recommendationLabel = "Pro. Smarter answers, slower; for phones with 12 GB RAM or more"
    )

    val llama32ThreeBGguf = GgufSpec(
        modelName = "Llama_3_2_3B_GGUF",
        displayNameOverride = "Llama 3.2 3B GGUF",
        repoId = "bartowski/Llama-3.2-3B-Instruct-GGUF",
        revision = "5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
        fileName = "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        fileBytes = 2_019_377_696L,
        licenseLabel = "Llama 3.2 Community License",
        defaultSystemInstruction = GGUF_ASSISTANT_INSTRUCTION,
        downloadSizeLabel = "2.02 GB",
        recommendationLabel = "Pro. Meta's 3B chat model; for phones with 8 GB RAM or more"
    )

    val phi4MiniGguf = GgufSpec(
        modelName = "Phi_4_mini_GGUF",
        displayNameOverride = "Phi-4-mini GGUF",
        repoId = "unsloth/Phi-4-mini-instruct-GGUF",
        revision = "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80",
        fileName = "Phi-4-mini-instruct-Q4_K_M.gguf",
        fileBytes = 2_491_874_272L,
        licenseLabel = "MIT",
        defaultSystemInstruction = GGUF_ASSISTANT_INSTRUCTION,
        downloadSizeLabel = "2.49 GB",
        recommendationLabel = "Pro. Microsoft's 3.8B model, strong at reasoning and maths; 8 GB RAM or more"
    )

    // Translation models (Jira PLP-16): suggested by the translator as lighter options next to Gemma 4.
    val indexTranslate2BGguf = GgufSpec(
        modelName = "Index_Translate_2B_GGUF",
        displayNameOverride = "Index-Translate 2B GGUF",
        repoId = "IndexTeam/Index-Translate-2B-GGUF",
        revision = "3a0eca6ce931d88d00614d427ab3ed8ff0743902",
        fileName = "Index-Translate-2B.Q4_K_M.gguf",
        fileBytes = 1_312_164_352L,
        licenseLabel = "Apache 2.0",
        defaultSystemInstruction = "",
        downloadSizeLabel = "1.31 GB",
        recommendationLabel = "Pro. For the translator only: a light model that translates most languages well; weak at reading Hindi, Bengali and Thai"
    )

    val milmmt1BGguf = GgufSpec(
        modelName = "MiLMMT_46_1B_GGUF",
        displayNameOverride = "MiLMMT 1B GGUF",
        repoId = "mradermacher/MiLMMT-46-1B-v1.0-GGUF",
        revision = "34df5efbe6592773ec168cc7b307728c08623472",
        fileName = "MiLMMT-46-1B-v1.0.Q4_K_M.gguf",
        fileBytes = 806_057_408L,
        licenseLabel = "Gemma Terms of Use",
        defaultSystemInstruction = "",
        downloadSizeLabel = "0.81 GB",
        recommendationLabel = "Pro. For the translator only: the smallest model, best at translating into English; no Ukrainian"
    )

    // PLaMo 3 Translate (Preferred Networks, PLaMo Community License), translator only. See Jira PLP-19.
    val plamoTranslate08BGguf = GgufSpec(
        modelName = "PLaMo_3_Translate_0_8B_GGUF",
        displayNameOverride = "PLaMo Translate 0.8B GGUF",
        repoId = "pfnet/plamo-3-translate-0.8b-GGUF",
        revision = "1cf96ae02133958a20074bfdfab76517bb737d00",
        fileName = "plamo-3-translate-0.8b-Q4_K_M.gguf",
        fileBytes = 573_126_848L,
        licenseLabel = "PLaMo Community License",
        defaultSystemInstruction = "",
        downloadSizeLabel = "0.57 GB",
        recommendationLabel = "Pro. For the translator only: the smallest, fast model for 50 languages; no Portuguese, Turkish, Ukrainian or Kannada"
    )

    val plamoTranslate2BGguf = GgufSpec(
        modelName = "PLaMo_3_Translate_2B_GGUF",
        displayNameOverride = "PLaMo Translate 2B GGUF",
        repoId = "pfnet/plamo-3-translate-2b-GGUF",
        revision = "4f672106b847200bfdc0f6af6c321cd94eba1aed",
        fileName = "plamo-3-translate-2b-Q3_K_M.gguf",
        fileBytes = 1_495_886_560L,
        licenseLabel = "PLaMo Community License",
        defaultSystemInstruction = "",
        downloadSizeLabel = "1.50 GB",
        recommendationLabel = "Pro. For the translator only: a mid-size model for 50 languages; no Portuguese, Turkish, Ukrainian or Kannada"
    )

    private val builtInModels = listOf(
        gemma4E4B, gemma4E2B, deepSeekR1DistillQwen15BLiteRt,
        qwen35FourBGguf, phi4MiniGguf, llama32ThreeBGguf, qwen35TwoBGguf, qwen35SmallGguf,
        indexTranslate2BGguf, milmmt1BGguf, plamoTranslate08BGguf, plamoTranslate2BGguf,
        qwen3LiteRt, qwen3, qwen25
    )
    @Volatile private var customModels: List<ModelDescriptor> = emptyList()

    /** GGUF models need the llama.cpp engine: only in builds that include it, and only packaged for arm64-v8a. */
    private val ggufEngineAbi: Boolean by lazy {
        LlamaFeature.isAvailable &&
            runCatching { android.os.Build.SUPPORTED_64_BIT_ABIS.firstOrNull() == "arm64-v8a" }.getOrDefault(false)
    }

    val all: List<ModelDescriptor>
        get() = (builtInModels + customModels).filter { it !is GgufSpec || ggufEngineAbi }

    fun loadCustomModels(context: android.content.Context) {
        customModels = CustomLiteRtModelStore(context).loadAll() +
            (LlamaFeature.instance?.loadCustomModels(context) ?: emptyList())
    }

    fun findById(id: String?): ModelDescriptor? {
        if (id.isNullOrBlank()) {
            return null
        }

        return all.firstOrNull { it.id == id }
    }
}

object DownloadableModelRegistry {
    fun findById(id: String?): ModelDescriptor? {
        // Files imported from the phone have nothing to download.
        return ModelRegistry.findById(id)?.takeUnless { it is CustomLiteRtSpec && it.downloadUrl.isEmpty() }
            ?: ImageGenFeature.instance?.findModelById(id)
    }
}

val ModelDescriptor.primaryModelFileName: String
    get() = when (this) {
        is OnnxQwenSpec -> modelAssetName
        is GemmaLiteRtSpec -> modelAssetName
        is QwenLiteRtSpec -> modelAssetName
        is CustomLiteRtSpec -> modelAssetName
        is ImageGenSpec -> modelFile.localFileName
        is GgufSpec -> fileName
    }

val ModelDescriptor.defaultInstruction: String
    get() = when (this) {
        is OnnxQwenSpec -> defaultSystemPrompt
        is GemmaLiteRtSpec -> defaultSystemInstruction
        is QwenLiteRtSpec -> defaultSystemInstruction
        is CustomLiteRtSpec -> InstructionPreset.default.instruction
        is ImageGenSpec -> ""
        is GgufSpec -> defaultSystemInstruction
    }
