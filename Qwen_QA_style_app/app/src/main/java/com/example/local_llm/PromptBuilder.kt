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
        val userStart = config.roleTokenIds.userStart
        val end = config.roleTokenIds.endToken

        val result = mutableListOf<Int>()

        // Add system prompt
        result.addAll(config.roleTokenIds.systemStart)
        result.addAll(systemTokens.toList())
        result.add(end)

        // Build context block: all full user-bot pairs except the last user prompt
        val contextPairs = StringBuilder()
        val numMessages = messages.size
        for (i in 0 until numMessages - 1 step 2) {
            val user = messages.getOrNull(i)
            val bot = messages.getOrNull(i + 1)
            if (user?.isUser == true && bot?.isUser == false) {
                contextPairs.append("Q: ${user.text.trim()}\nA: ${bot.text.trim()}\n")
            }
        }

        if (contextPairs.isNotEmpty()) {
            val contextTokens = tokenizer.tokenize(contextPairs.toString()).toList()
            val trimmedContext = if (contextTokens.size > maxTokens) contextTokens.takeLast(maxTokens) else contextTokens

            // Wrap context block
            val contextStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("context"),
                tokenizer.getTokenId("Ċ")
            )
            result.addAll(contextStart)
            result.addAll(trimmedContext)
            result.add(end)
        }

        // Add last user prompt
        val lastUser = messages.lastOrNull { it.isUser } ?: return result.toIntArray()
        result.addAll(userStart)
        result.addAll(tokenizer.tokenize(lastUser.text).toList())
        result.add(end)

        // Assistant start (generation begins here)
        result.addAll(assistantStart)

        return result.toIntArray()
    }

}
