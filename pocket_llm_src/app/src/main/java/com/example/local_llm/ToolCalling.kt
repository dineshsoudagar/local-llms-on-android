package com.example.local_llm

enum class ExternalToolChoiceMode {
    AUTO,
    NONE,
    REQUIRED,
    FUNCTION
}

data class ExternalToolChoice(
    val mode: ExternalToolChoiceMode = ExternalToolChoiceMode.AUTO,
    val functionName: String? = null
)

data class ExternalToolDefinition(
    val name: String,
    val description: String?,
    val parametersJson: String
)

data class ExternalToolCall(
    val id: String,
    val name: String,
    val argumentsJson: String
)
