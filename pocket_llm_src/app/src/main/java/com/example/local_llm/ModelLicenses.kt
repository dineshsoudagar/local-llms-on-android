package com.example.local_llm

/** License shown next to a model in Manage models, with a link to its model card. */
data class ModelLicense(val label: String, val url: String?, val attribution: String? = null)

private const val HF = "https://huggingface.co"

val ModelDescriptor.license: ModelLicense?
    get() = when (this) {
        is OnnxQwenSpec -> ModelLicense("Apache 2.0", if (this == ModelRegistry.qwen25) "$HF/Qwen/Qwen2.5-0.5B-Instruct" else "$HF/Qwen/Qwen3-0.6B")
        is GemmaLiteRtSpec -> ModelLicense("Apache 2.0", "$HF/google/${modelAssetName.substringBefore(".litertlm")}")
        is QwenLiteRtSpec -> if (this == ModelRegistry.deepSeekR1DistillQwen15BLiteRt) {
            ModelLicense("MIT", "$HF/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        } else {
            ModelLicense("Apache 2.0", "$HF/Qwen/Qwen3-0.6B")
        }
        is GgufSpec -> if (repoId == LlamaFeature.LOCAL_REPO) null else ModelLicense(
            label = licenseLabel,
            url = "$HF/$repoId",
            attribution = if (licenseLabel.contains("Llama", ignoreCase = true)) "Built with Llama" else null
        )
        is ImageGenSpec -> if (modelCardUrl == null && licenseLabel == "Your own file") null else ModelLicense(licenseLabel, modelCardUrl)
        // Imported files: the user supplied them and knows their license.
        is CustomLiteRtSpec -> if (downloadUrl.isNotEmpty()) {
            ModelLicense("See the model card", downloadUrl.substringBefore("/resolve/"))
        } else {
            null
        }
    }

/** One line for the model list, e.g. "License: Apache 2.0 · Built with Llama". */
fun ModelLicense.summary(): String = buildString {
    append("License: ").append(label)
    attribution?.let { append(" · ").append(it) }
}