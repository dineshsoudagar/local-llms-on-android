package com.example.local_llm

import org.json.JSONArray
import org.json.JSONObject

internal data class ParsedChatRequest(
    val model: String?,
    val history: List<ChatTurn>,
    val systemInstruction: String?,
    val stream: Boolean,
    val outputTokenReserve: Int?,
    val tools: List<ExternalToolDefinition>,
    val toolChoice: ExternalToolChoice,
    val parallelToolCalls: Boolean
) {
    val hasToolMessages: Boolean
        get() = history.any { it.toolCalls.isNotEmpty() || it.isToolResult }

    fun toolsForInference(): List<ExternalToolDefinition> {
        return when (toolChoice.mode) {
            ExternalToolChoiceMode.NONE -> emptyList()
            ExternalToolChoiceMode.FUNCTION -> tools.filter { it.name == toolChoice.functionName }
            else -> tools
        }
    }
}

internal object LanChatProtocol {
    fun parseChatRequest(body: String): ParsedChatRequest {
        val request = JSONObject(body)
        val messages = request.optJSONArray("messages")
        val history = mutableListOf<ChatTurn>()
        val systemInstructionParts = mutableListOf<String>()

        if (messages != null) {
            for (index in 0 until messages.length()) {
                val message = messages.optJSONObject(index) ?: continue
                val role = message.optString("role").lowercase()
                when (role) {
                    "system", "developer" -> {
                        extractTextContent(message.opt("content"))
                            .takeIf(String::isNotBlank)
                            ?.let(systemInstructionParts::add)
                    }

                    "user" -> {
                        val content = extractTextContent(message.opt("content"))
                        if (content.isNotBlank()) {
                            history += ChatTurn(role = ChatRole.USER, text = content)
                        }
                    }

                    "assistant" -> {
                        val content = extractTextContent(message.opt("content"))
                        val toolCalls = parseAssistantToolCalls(message.optJSONArray("tool_calls"))
                        if (content.isNotBlank() || toolCalls.isNotEmpty()) {
                            history += ChatTurn(
                                role = ChatRole.ASSISTANT,
                                text = content,
                                toolCalls = toolCalls
                            )
                        }
                    }

                    "tool" -> {
                        val toolCallId = message.optString("tool_call_id").trim()
                        require(toolCallId.isNotEmpty()) {
                            "Tool messages must include tool_call_id."
                        }
                        val toolCall = history.asReversed()
                            .flatMap { it.toolCalls }
                            .firstOrNull { it.id == toolCallId }
                        require(toolCall != null) {
                            "tool_call_id '$toolCallId' does not match an earlier assistant tool call."
                        }
                        history += ChatTurn(
                            role = ChatRole.ASSISTANT,
                            text = extractTextContent(message.opt("content")),
                            toolCallId = toolCallId,
                            toolName = toolCall.name,
                            isToolResult = true
                        )
                    }
                }
            }
        }

        if (history.isEmpty()) {
            request.optString("prompt", "").trim()
                .takeIf(String::isNotEmpty)
                ?.let { history += ChatTurn(role = ChatRole.USER, text = it) }
        }

        require(
            history.isNotEmpty() &&
                (history.last().role == ChatRole.USER || history.last().isToolResult)
        ) {
            "Provide at least one user message or tool result."
        }

        val tools = parseTools(request.optJSONArray("tools"))
        val toolChoice = parseToolChoice(request.opt("tool_choice"), tools)
        return ParsedChatRequest(
            model = request.optString("model", "").trim().takeIf(String::isNotEmpty),
            history = history,
            systemInstruction = systemInstructionParts.joinToString("\n\n")
                .takeIf(String::isNotBlank),
            stream = request.optBoolean("stream", false),
            outputTokenReserve = request.optInt("max_tokens", 0).takeIf { it > 0 },
            tools = tools,
            toolChoice = toolChoice,
            parallelToolCalls = request.optBoolean("parallel_tool_calls", true)
        )
    }

    fun extractTextContent(value: Any?): String {
        return when (value) {
            is String -> value.trim()
            is JSONArray -> buildString {
                for (index in 0 until value.length()) {
                    val part = value.optJSONObject(index) ?: continue
                    if (part.optString("type") == "text") {
                        append(part.optString("text"))
                    }
                }
            }.trim()
            else -> ""
        }
    }

    fun isBearerCredentialAccepted(
        authorization: String,
        passwordVerifier: (String) -> Boolean,
        apiKeyVerifier: (String) -> Boolean
    ): Boolean {
        if (!authorization.startsWith("Bearer ")) return false
        val token = authorization.removePrefix("Bearer ").trim()
        return passwordVerifier(token) || apiKeyVerifier(token)
    }

    fun completionResponse(
        completionId: String,
        created: Long,
        modelId: String,
        response: BackendResponse
    ): JSONObject {
        val message = JSONObject()
            .put("role", "assistant")
            .put("content", response.text.takeIf(String::isNotEmpty) ?: JSONObject.NULL)
        if (response.toolCalls.isNotEmpty()) {
            message.put("tool_calls", toolCallsJson(response.toolCalls))
        }
        return JSONObject()
            .put("id", completionId)
            .put("object", "chat.completion")
            .put("created", created)
            .put("model", modelId)
            .put(
                "choices",
                JSONArray().put(
                    JSONObject()
                        .put("index", 0)
                        .put("message", message)
                        .put("finish_reason", if (response.toolCalls.isEmpty()) "stop" else "tool_calls")
                )
            )
    }

    fun toolCallsJson(
        toolCalls: List<ExternalToolCall>,
        includeIndex: Boolean = false,
        startIndex: Int = 0
    ): JSONArray {
        return JSONArray().apply {
            toolCalls.forEachIndexed { index, call ->
                val json = JSONObject()
                        .put("id", call.id)
                        .put("type", "function")
                        .put(
                            "function",
                            JSONObject()
                                .put("name", call.name)
                                .put("arguments", call.argumentsJson)
                        )
                if (includeIndex) json.put("index", startIndex + index)
                put(json)
            }
        }
    }

    private fun parseTools(value: JSONArray?): List<ExternalToolDefinition> {
        if (value == null) return emptyList()
        return (0 until value.length()).map { index ->
            val tool = value.optJSONObject(index)
                ?: throw IllegalArgumentException("Each tool must be an object.")
            require(tool.optString("type") == "function") {
                "Only function tools are supported."
            }
            val function = tool.optJSONObject("function")
                ?: throw IllegalArgumentException("Function tools must include function metadata.")
            val name = function.optString("name").trim()
            require(name.isNotEmpty()) { "Function tools must include a name." }
            val parameters = function.optJSONObject("parameters") ?: JSONObject()
                .put("type", "object")
                .put("properties", JSONObject())
            ExternalToolDefinition(
                name = name,
                description = function.optString("description", "")
                    .takeIf(String::isNotBlank),
                parametersJson = parameters.toString()
            )
        }
    }

    private fun parseToolChoice(value: Any?, tools: List<ExternalToolDefinition>): ExternalToolChoice {
        if (value == null || value == JSONObject.NULL) return ExternalToolChoice()
        if (value is String) {
            return when (value.lowercase()) {
                "auto" -> ExternalToolChoice()
                "none" -> ExternalToolChoice(ExternalToolChoiceMode.NONE)
                "required" -> ExternalToolChoice(ExternalToolChoiceMode.REQUIRED)
                else -> throw IllegalArgumentException("Unsupported tool_choice '$value'.")
            }
        }
        val choice = value as? JSONObject
            ?: throw IllegalArgumentException("tool_choice must be auto, none, required, or a function object.")
        require(choice.optString("type") == "function") {
            "Only function tool_choice objects are supported."
        }
        val name = choice.optJSONObject("function")?.optString("name")?.trim().orEmpty()
        require(name.isNotEmpty()) { "Function tool_choice must include a function name." }
        require(tools.any { it.name == name }) {
            "tool_choice refers to an undeclared function '$name'."
        }
        return ExternalToolChoice(ExternalToolChoiceMode.FUNCTION, name)
    }

    private fun parseAssistantToolCalls(value: JSONArray?): List<ExternalToolCall> {
        if (value == null) return emptyList()
        return (0 until value.length()).mapIndexed { index, itemIndex ->
            val call = value.optJSONObject(itemIndex)
                ?: throw IllegalArgumentException("Each assistant tool call must be an object.")
            require(call.optString("type", "function") == "function") {
                "Only function assistant tool calls are supported."
            }
            val function = call.optJSONObject("function")
                ?: throw IllegalArgumentException("Assistant tool calls must include function metadata.")
            val name = function.optString("name").trim()
            require(name.isNotEmpty()) { "Assistant tool calls must include a function name." }
            val arguments = function.opt("arguments")
            ExternalToolCall(
                id = call.optString("id").trim().ifEmpty { "call_input_$index" },
                name = name,
                argumentsJson = when (arguments) {
                    is JSONObject, is JSONArray -> arguments.toString()
                    is String -> arguments
                    else -> "{}"
                }
            )
        }
    }
}
