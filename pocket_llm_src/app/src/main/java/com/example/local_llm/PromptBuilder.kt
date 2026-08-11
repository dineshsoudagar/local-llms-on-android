package com.example.local_llm

class PromptBuilder(
    private val tokenizer: BpeTokenizer,
    private val config: ModelConfig
) {
    private val qwenSerializer = QwenChatPromptSerializer(tokenizer::tokenize, config.roleTokenIds)

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
        return qwenSerializer.serializeWithinLimit(
            messages = messages,
            systemPrompt = systemPrompt ?: config.defaultSystemPrompt,
            maxTokens = maxTokens,
            allowHistoryTruncation = true
        )
    }

    fun buildPromptTokensStrict(
        messages: List<ChatTurn>,
        intent: PromptIntent,
        maxTokens: Int
    ): IntArray = when (config.promptStyle) {
        PromptStyle.QWEN2_5, PromptStyle.QWEN3 -> when (intent) {
            is PromptIntent.QA -> qwenSerializer.serializeWithinLimit(
                messages = messages,
                systemPrompt = intent.systemPrompt ?: config.defaultSystemPrompt,
                maxTokens = maxTokens,
                allowHistoryTruncation = false
            )
        }
    }

    fun countPromptTokens(messages: List<ChatTurn>, intent: PromptIntent): Int = when (config.promptStyle) {
        PromptStyle.QWEN2_5, PromptStyle.QWEN3 -> when (intent) {
            is PromptIntent.QA -> qwenSerializer.serializeAll(
                messages,
                intent.systemPrompt ?: config.defaultSystemPrompt
            ).size
        }
    }
}

internal class QwenChatPromptSerializer(
    private val tokenize: (String) -> IntArray,
    private val roleTokenIds: RoleTokenIds
) {
    fun serializeAll(messages: List<ChatTurn>, systemPrompt: String): IntArray {
        val blocks = promptBlocks(messages, systemPrompt)
        return buildList {
            blocks.forEach(::addAll)
            addAll(roleTokenIds.assistantStart)
        }.toIntArray()
    }

    fun serializeWithinLimit(
        messages: List<ChatTurn>,
        systemPrompt: String,
        maxTokens: Int,
        allowHistoryTruncation: Boolean
    ): IntArray {
        require(maxTokens > 0) { "The ONNX prompt limit must be positive." }
        val blocks = promptBlocks(messages, systemPrompt)
        val systemBlock = blocks.first()
        val assistantStart = roleTokenIds.assistantStart
        require(systemBlock.size + assistantStart.size <= maxTokens) {
            "The system instruction exceeds the ONNX prompt budget. Shorten it before sending."
        }
        val turnBlocks = blocks.drop(1)
        val completeSize = systemBlock.size + turnBlocks.sumOf(List<Int>::size) + assistantStart.size
        if (!allowHistoryTruncation) {
            require(completeSize <= maxTokens) {
                "The complete serialized prompt needs $completeSize tokens, but the ONNX prompt limit is $maxTokens."
            }
            return serializeAll(messages, systemPrompt)
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
        return buildList {
            addAll(systemBlock)
            retainedReversed.asReversed().forEach(::addAll)
            addAll(assistantStart)
        }.toIntArray()
    }

    private fun promptBlocks(messages: List<ChatTurn>, systemPrompt: String): List<List<Int>> = buildList {
        add(buildList {
            addAll(roleTokenIds.systemStart)
            addAll(tokenize(systemPrompt).toList())
            add(roleTokenIds.endToken)
        })
        messages.forEach { message ->
            add(buildList {
                addAll(if (message.role == ChatRole.USER) roleTokenIds.userStart else roleTokenIds.assistantStart)
                addAll(tokenize(message.text).toList())
                add(roleTokenIds.endToken)
            })
        }
    }
}
