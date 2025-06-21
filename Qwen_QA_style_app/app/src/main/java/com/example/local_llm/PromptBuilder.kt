package com.example.local_llm

class PromptBuilder(
    private val tokenizer: BpeTokenizer,
    private val config: ModelConfig
) {
    fun buildPromptTokens(messages: List<Message>, intent: PromptIntent, maxTokens: Int = 500): IntArray {
        return when (config.promptStyle) {
            PromptStyle.QWEN2_5, PromptStyle.QWEN3 -> when (intent) {
                is PromptIntent.QA -> buildQwenChatPrompt(messages, intent.systemPrompt, maxTokens)
            }
        }
    }

    private fun buildQwenQA(userInput: String, systemPromptOverride: String?): IntArray {
        val systemPrompt = systemPromptOverride ?: config.defaultSystemPrompt
        val userPrompt = "Q: $userInput\nA:"

        val systemTokens = tokenizer.tokenize(systemPrompt)
        val userTokens = tokenizer.tokenize(userPrompt)

        return buildList {
            addAll(config.roleTokenIds.systemStart)
            addAll(systemTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.userStart)
            addAll(userTokens.toList())
            add(config.roleTokenIds.endToken)

            addAll(config.roleTokenIds.assistantStart)
        }.toIntArray()
    }

    fun buildQwenChatPrompt(messages: List<Message>, systemPrompt: String? = null, maxTokens: Int = 500): IntArray {
        val systemTokens = tokenizer.tokenize(systemPrompt ?: config.defaultSystemPrompt)
        val assistantStart = config.roleTokenIds.assistantStart
        val end = config.roleTokenIds.endToken

        val conversationTokens = mutableListOf<Int>()
        conversationTokens.addAll(config.roleTokenIds.systemStart)
        conversationTokens.addAll(systemTokens.toList())
        conversationTokens.add(end)

        val turns = mutableListOf<Int>()
        for (msg in messages) {
            val roleTokens = if (msg.isUser) config.roleTokenIds.userStart else assistantStart
            val msgTokens = tokenizer.tokenize(msg.text)
            turns.addAll(roleTokens)
            turns.addAll(msgTokens.toList())
            turns.add(end)
        }

        val finalTurns = if (turns.size > maxTokens) {
            turns.takeLast(maxTokens)
        } else turns

        val result = mutableListOf<Int>()
        result.addAll(conversationTokens)
        result.addAll(finalTurns)
        result.addAll(assistantStart)

        return result.toIntArray()
    }
}
