package com.example.local_llm

enum class PromptStyle {
    QWEN
}

data class RoleTokenIds(
    val systemStart: List<Int>,
    val userStart: List<Int>,
    val assistantStart: List<Int>,
    val endToken: Int
)

data class ModelConfig(
    val modelName: String,
    val promptStyle: PromptStyle,
    val eosTokenIds: Set<Int>,
    val numLayers: Int,
    val numKvHeads: Int,
    val headDim: Int,
    val batchSize: Int,
    val defaultSystemPrompt: String,
    val roleTokenIds: RoleTokenIds
)
