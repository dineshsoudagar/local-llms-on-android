package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LanChatProtocolTest {
    @Test
    fun normalTextCompletionIsOpenAiShaped() {
        val response = LanChatProtocol.completionResponse(
            completionId = "chatcmpl-test",
            created = 123L,
            modelId = "qwen3_litert",
            response = BackendResponse(text = "hello")
        )

        assertEquals("chat.completion", response.getString("object"))
        assertEquals("hello", response.getJSONArray("choices")
            .getJSONObject(0).getJSONObject("message").getString("content"))
        assertEquals("stop", response.getJSONArray("choices")
            .getJSONObject(0).getString("finish_reason"))
    }

    @Test
    fun toolDeclarationAssistantCallAndToolFollowUpAreParsed() {
        val request = LanChatProtocol.parseChatRequest(
            """
            {
              "model": "qwen3_litert",
              "tools": [{
                "type": "function",
                "function": {
                  "name": "read_file",
                  "description": "Read a file",
                  "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                  }
                }
              }],
              "tool_choice": "auto",
              "parallel_tool_calls": false,
              "messages": [
                {"role": "user", "content": "Read README.md"},
                {"role": "assistant", "content": null, "tool_calls": [{
                  "id": "call_1",
                  "type": "function",
                  "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"}
              ]
            }
            """.trimIndent()
        )

        assertEquals("read_file", request.tools.single().name)
        assertFalse(request.parallelToolCalls)
        assertEquals(3, request.history.size)
        assertEquals("call_1", request.history[1].toolCalls.single().id)
        assertTrue(request.history[2].isToolResult)
        assertEquals("read_file", request.history[2].toolName)
        assertEquals("call_1", request.history[2].toolCallId)
    }

    @Test
    fun assistantToolCallResponseContainsStructuredCall() {
        val response = LanChatProtocol.completionResponse(
            completionId = "chatcmpl-tool",
            created = 123L,
            modelId = "qwen3_litert",
            response = BackendResponse(
                text = "",
                toolCalls = listOf(
                    ExternalToolCall("call_1", "read_file", "{\"path\":\"README.md\"}")
                )
            )
        )
        val choice = response.getJSONArray("choices").getJSONObject(0)
        val call = choice.getJSONObject("message").getJSONArray("tool_calls").getJSONObject(0)

        assertEquals("call_1", call.getString("id"))
        assertEquals("function", call.getString("type"))
        assertEquals("read_file", call.getJSONObject("function").getString("name"))
        assertEquals("{\"path\":\"README.md\"}", call.getJSONObject("function").getString("arguments"))
        assertEquals("tool_calls", choice.getString("finish_reason"))

        val streamCall = LanChatProtocol.toolCallsJson(
            listOf(ExternalToolCall("call_1", "read_file", "{\"path\":\"README.md\"}")),
            includeIndex = true
        ).getJSONObject(0)
        assertEquals(0, streamCall.getInt("index"))
    }

    @Test
    fun wrongBearerCredentialIsRejected() {
        assertFalse(
            LanChatProtocol.isBearerCredentialAccepted(
                authorization = "Bearer wrong",
                passwordVerifier = { it == "correct" },
                apiKeyVerifier = { false }
            )
        )
        assertTrue(
            LanChatProtocol.isBearerCredentialAccepted(
                authorization = "Bearer correct",
                passwordVerifier = { it == "correct" },
                apiKeyVerifier = { false }
            )
        )
    }

    @Test
    fun maxCompletionTokensWinsOverMaxTokens() {
        val request = LanChatProtocol.parseChatRequest(
            """{"max_tokens": 64, "max_completion_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}"""
        )
        assertEquals(16, request.maxOutputTokens)

        val legacy = LanChatProtocol.parseChatRequest(
            """{"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}"""
        )
        assertEquals(64, legacy.maxOutputTokens)
    }

    @Test
    fun streamOptionsIncludeUsageIsParsed() {
        val request = LanChatProtocol.parseChatRequest(
            """{"stream": true, "stream_options": {"include_usage": true}, "messages": [{"role": "user", "content": "hi"}]}"""
        )
        assertTrue(request.includeUsageInStream)

        val plain = LanChatProtocol.parseChatRequest(
            """{"stream": true, "messages": [{"role": "user", "content": "hi"}]}"""
        )
        assertFalse(plain.includeUsageInStream)
    }

    @Test
    fun lengthLimitedCompletionReportsLengthAndUsage() {
        val response = LanChatProtocol.completionResponse(
            completionId = "chatcmpl-test",
            created = 123L,
            modelId = "gemma4_e4b",
            response = BackendResponse(text = "partial answer"),
            finishReason = LanChatProtocol.finishReason(BackendResponse(text = "partial answer"), lengthLimited = true),
            usage = LanChatProtocol.usageJson(promptTokens = 10, completionTokens = 4)
        )

        assertEquals("length", response.getJSONArray("choices")
            .getJSONObject(0).getString("finish_reason"))
        val usage = response.getJSONObject("usage")
        assertEquals(10, usage.getInt("prompt_tokens"))
        assertEquals(4, usage.getInt("completion_tokens"))
        assertEquals(14, usage.getInt("total_tokens"))
    }

    @Test
    fun toolCallsKeepToolCallsFinishReasonEvenWhenLengthLimited() {
        val response = BackendResponse(
            text = "",
            toolCalls = listOf(ExternalToolCall(id = "call_1", name = "read_file", argumentsJson = "{}"))
        )
        assertEquals("tool_calls", LanChatProtocol.finishReason(response, lengthLimited = true))
    }

    @Test
    fun completionTokenEstimateGrowsWithText() {
        assertEquals(0, LanChatProtocol.estimateCompletionTokens(BackendResponse(text = "")))
        val short = LanChatProtocol.estimateCompletionTokens(BackendResponse(text = "hello"))
        val long = LanChatProtocol.estimateCompletionTokens(BackendResponse(text = "hello ".repeat(200)))
        assertTrue(short > 0)
        assertTrue(long > short)
    }
}
