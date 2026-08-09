package com.example.local_llm

class PromptBuilder(
    private val tokenizer: BpeTokenizer,
    private val config: ModelConfig
) {
    fun buildPromptTokens(messages: List<ChatTurn>, intent: PromptIntent, maxTokens: Int = OnnxModel.MAX_INPUT_TOKENS): IntArray {
        return when (config.promptStyle) {
            PromptStyle.QWEN2_5, PromptStyle.QWEN3 -> when (intent) {
                is PromptIntent.QA -> buildQwenChatPrompt(messages, intent.systemPrompt, maxTokens)
            }
        }
    }

    fun buildQwenChatPrompt(
        messages: List<ChatTurn>,
        systemPrompt: String? = null,
        maxTokens: Int = OnnxModel.MAX_INPUT_TOKENS
    ): IntArray {
        val systemTokens = tokenizer.tokenize(systemPrompt ?: config.defaultSystemPrompt)
        val assistantStart = config.roleTokenIds.assistantStart
        val end = config.roleTokenIds.endToken

        val systemBlock = buildList {
            addAll(config.roleTokenIds.systemStart)
            addAll(systemTokens.toList())
            add(end)
        }
        require(systemBlock.size + assistantStart.size < maxTokens) {
            "The system instruction exceeds the ONNX prompt budget. Shorten it before sending."
        }
        val turnBlocks = messages.map { msg ->
            val roleTokens = if (msg.role == ChatRole.USER) config.roleTokenIds.userStart else assistantStart
            val msgTokens = tokenizer.tokenize(msg.text)
            buildList {
                addAll(roleTokens)
                addAll(msgTokens.toList())
                add(end)
            }
        }
        val turnBudget = maxTokens - systemBlock.size - assistantStart.size
        require((turnBlocks.lastOrNull()?.size ?: 0) <= turnBudget) {
            "The current user message exceeds the ONNX prompt budget. Shorten the request or attachment context."
        }
        val retainedReversed = mutableListOf<List<Int>>()
        var used = 0
        for (block in turnBlocks.asReversed()) {
            if (used + block.size > turnBudget) continue
            retainedReversed += block
            used += block.size
        }

        val result = mutableListOf<Int>()
        result.addAll(systemBlock)
        retainedReversed.asReversed().forEach(result::addAll)
        result.addAll(assistantStart)

        return result.toIntArray()
    }
}
