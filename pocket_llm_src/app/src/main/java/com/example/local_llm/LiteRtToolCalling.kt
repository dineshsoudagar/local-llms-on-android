package com.example.local_llm

import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

internal fun externalToolProviders(
    definitions: List<ExternalToolDefinition>
): List<ToolProvider> {
    return definitions.map { definition ->
        tool(object : OpenApiTool {
            override fun getToolDescriptionJsonString(): String {
                val description = JSONObject()
                    .put("name", definition.name)
                    .put("parameters", JSONObject(definition.parametersJson))
                definition.description?.let { description.put("description", it) }
                return description.toString()
            }

            override fun execute(paramsJsonString: String): String {
                return JSONObject()
                    .put("error", "Pocket LLM exposes tool calls to the client; it does not execute client tools.")
                    .toString()
            }
        })
    }
}

internal fun ChatTurn.toLiteRtMessage(): Message {
    if (isToolResult) {
        val name = toolName ?: error("A tool result is missing its tool name.")
        return Message.tool(
            Contents.of(Content.ToolResponse(name, jsonValueOrString(text)))
        )
    }
    if (toolCalls.isNotEmpty()) {
        val contents = if (text.isBlank()) {
            Contents.emptyForToolCall()
        } else {
            Contents.of(text)
        }
        return Message.model(
            contents = contents,
            toolCalls = toolCalls.map { it.toLiteRtToolCall() }
        )
    }
    return when (role) {
        ChatRole.USER -> Message.user(text)
        ChatRole.ASSISTANT -> Message.model(text)
    }
}

private fun Contents.Companion.emptyForToolCall(): Contents {
    return Contents.of(*emptyArray<Content>())
}

private fun ExternalToolCall.toLiteRtToolCall(): ToolCall {
    return ToolCall(name, JSONObject(argumentsJson).toMapValues())
}

internal fun nativeToolCallsToExternal(
    nativeToolCalls: List<ToolCall>,
    previous: List<ExternalToolCall> = emptyList(),
    parallelToolCalls: Boolean
): List<ExternalToolCall> {
    if (!parallelToolCalls && nativeToolCalls.size > 1) {
        error("The model returned parallel tool calls while parallel_tool_calls=false.")
    }
    return nativeToolCalls.mapIndexed { index, nativeCall ->
        val existing = previous.getOrNull(index)
        ExternalToolCall(
            id = existing?.id ?: "call_${UUID.randomUUID().toString().replace("-", "")}",
            name = nativeCall.name,
            argumentsJson = JSONObject(nativeCall.arguments).toString()
        )
    }
}

internal fun Message.visibleText(): String {
    val text = contents.contents
        .filterIsInstance<Content.Text>()
        .joinToString(separator = "") { it.text }
    return if (toolCalls.isNotEmpty()) text else text.ifBlank { contents.toString() }
}

private fun jsonValueOrString(text: String): Any? {
    return runCatching { JSONObject(text).toMapValues() }
        .recoverCatching { JSONArray(text).toListValues() }
        .getOrElse { text }
}

private fun JSONObject.toMapValues(): Map<String, Any?> {
    val result = linkedMapOf<String, Any?>()
    keys().forEach { key -> result[key] = valueOrNull(opt(key)) }
    return result
}

private fun JSONArray.toListValues(): List<Any?> {
    return (0 until length()).map { index -> valueOrNull(opt(index)) }
}

private fun valueOrNull(value: Any?): Any? {
    return when (value) {
        JSONObject.NULL -> null
        is JSONObject -> value.toMapValues()
        is JSONArray -> value.toListValues()
        else -> value
    }
}
