package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PromptBudgetingTest {
    private val roleTokens = RoleTokenIds(
        systemStart = listOf(10),
        userStart = listOf(20),
        assistantStart = listOf(30),
        endToken = 40
    )
    private val serializer = QwenChatPromptSerializer(
        tokenize = { text -> IntArray(text.length) { index -> index } },
        roleTokenIds = roleTokens
    )

    @Test
    fun exact512TokenBoundaryIsAccepted() {
        val tokens = serializer.serializeWithinLimit(
            messages = listOf(ChatTurn(role = ChatRole.USER, text = "u".repeat(507))),
            systemPrompt = "",
            maxTokens = 512,
            allowHistoryTruncation = false
        )

        assertEquals(512, tokens.size)
    }

    @Test
    fun oneTokenOverBoundaryIsRejected() {
        val error = runCatching {
            serializer.serializeWithinLimit(
                messages = listOf(ChatTurn(role = ChatRole.USER, text = "u".repeat(508))),
                systemPrompt = "",
                maxTokens = 512,
                allowHistoryTruncation = false
            )
        }.exceptionOrNull()

        assertNotNull(error)
        assertTrue(error!!.message.orEmpty().contains("513"))
    }

    @Test
    fun outputReserveReducesTheExactAdmissiblePromptBoundary() {
        val backend = ExactFakeBackend(serializer)
        val exact = InferenceRequest(
            history = listOf(ChatTurn(role = ChatRole.USER, text = "u".repeat(379))),
            thinkingEnabled = false,
            modelInstruction = "",
            outputTokenReserve = 128
        )
        val over = exact.copy(
            history = listOf(ChatTurn(role = ChatRole.USER, text = "u".repeat(380)))
        )

        assertEquals(384, backend.requirePromptFits(exact))
        assertNotNull(runCatching { backend.requirePromptFits(over) }.exceptionOrNull())
    }

    @Test
    fun longSystemPromptFailsWithoutDroppingRoleBlocks() {
        val error = runCatching {
            serializer.serializeWithinLimit(
                messages = listOf(ChatTurn(role = ChatRole.USER, text = "question")),
                systemPrompt = "s".repeat(510),
                maxTokens = 512,
                allowHistoryTruncation = false
            )
        }.exceptionOrNull()

        assertNotNull(error)
        assertTrue(error!!.message.orEmpty().contains("system instruction", ignoreCase = true))
    }

    @Test
    fun multipleHistoricalRoleBlocksAndFinalAssistantMarkerAreCounted() {
        val tokens = QwenChatPromptSerializer(
            tokenize = { text -> IntArray(text.length) },
            roleTokenIds = RoleTokenIds(
                systemStart = listOf(1, 2, 3),
                userStart = listOf(4, 5, 6),
                assistantStart = listOf(7, 8, 9),
                endToken = 10
            )
        ).serializeAll(
            messages = listOf(
                ChatTurn(role = ChatRole.USER, text = "abc"),
                ChatTurn(role = ChatRole.ASSISTANT, text = "de")
            ),
            systemPrompt = "system"
        )

        assertEquals(26, tokens.size)
    }

    @Test
    fun oversizedFirstChunkIsSplitAndEveryIntermediatePromptFits() {
        val history = listOf(
            ChatTurn(role = ChatRole.USER, text = "earlier question"),
            ChatTurn(role = ChatRole.ASSISTANT, text = "earlier answer")
        )
        val limit = 512
        val fits: (String) -> Boolean = { prompt ->
            serializer.serializeAll(
                history + ChatTurn(role = ChatRole.USER, text = prompt),
                "system"
            ).size <= limit
        }
        val oversized = AttachmentChunk(
            id = "oversized",
            ordinal = 0,
            text = (1..180).joinToString(" ") { "word$it" },
            source = AttachmentSourceRef(pageNumber = 3),
            estimatedTokens = 1_000
        )

        val plan = AttachmentPromptPlanner(fits).plan(
            chunks = listOf(oversized),
            userPrompt = "Summarize everything",
            inputTokenBudget = limit
        )

        assertTrue(plan.batches.size > 1)
        assertTrue(plan.batches.flatMap { it.chunks }.none { it.id == oversized.id })
        assertTrue(plan.batches.all { fits(it.prompt) })
        assertTrue(plan.batches.all { "[p. 3]" in it.prompt })
    }

    @Test
    fun questionRetrievalAndTransformationBatchesUseCompletePromptChecks() {
        val limit = 420
        val history = listOf(
            ChatTurn(role = ChatRole.USER, text = "old user"),
            ChatTurn(role = ChatRole.ASSISTANT, text = "old assistant")
        )
        val fits: (String) -> Boolean = { prompt ->
            serializer.serializeAll(
                history + ChatTurn(role = ChatRole.USER, text = prompt),
                "system instruction"
            ).size <= limit
        }
        val chunks = listOf(
            AttachmentChunk(
                id = "relevant",
                ordinal = 0,
                text = "android audio codec ".repeat(35),
                source = AttachmentSourceRef(pageNumber = 1),
                estimatedTokens = 500
            ),
            AttachmentChunk(
                id = "other",
                ordinal = 1,
                text = "gardening notes ".repeat(25),
                source = AttachmentSourceRef(pageNumber = 2),
                estimatedTokens = 300
            )
        )

        val question = AttachmentPromptPlanner(fits).plan(
            chunks,
            "Which audio codec is described?",
            inputTokenBudget = limit
        )
        val transformation = AttachmentPromptPlanner(fits).plan(
            chunks,
            "Translate all of this",
            inputTokenBudget = limit
        )

        assertEquals(AttachmentTask.QUESTION, question.task)
        assertTrue(question.batches.all { fits(it.prompt) })
        assertTrue(question.batches.single().prompt.contains("[p. 1]"))
        assertEquals(AttachmentTask.TRANSFORMATION, transformation.task)
        assertTrue(transformation.batches.size > 1)
        assertTrue(transformation.batches.all { fits(it.prompt) })
        assertEquals(
            setOf(1, 2),
            transformation.batches.flatMap { it.chunks }.mapNotNull { it.source.pageNumber }.toSet()
        )
    }

    @Test
    fun minimumPromptFailureIsExplicit() {
        val error = runCatching {
            AttachmentPromptPlanner { false }.plan(
                chunks = listOf(
                    AttachmentChunk(
                        id = "tiny",
                        ordinal = 0,
                        text = "x",
                        source = AttachmentSourceRef(pageNumber = 1),
                        estimatedTokens = 1
                    )
                ),
                userPrompt = "Summarize",
                inputTokenBudget = 1
            )
        }.exceptionOrNull()

        assertNotNull(error)
        assertTrue(error!!.message.orEmpty().contains("minimum valid attachment prompt"))
    }

    @Test
    fun wrapperSourceLabelsIntermediateAndSynthesisPromptsAllFit() {
        val history = listOf(
            ChatTurn(role = ChatRole.USER, text = "historical user block"),
            ChatTurn(role = ChatRole.ASSISTANT, text = "historical assistant block")
        )
        val limit = 512
        val generatedRequests = mutableListOf<String>()
        val fits: (String) -> Boolean = { prompt ->
            val fitsLimit = serializer.serializeAll(
                history + ChatTurn(role = ChatRole.USER, text = prompt),
                "long system instruction with policy text"
            ).size <= limit
            if (fitsLimit) generatedRequests += prompt
            fitsLimit
        }
        val chunks = (1..4).map { page ->
            AttachmentChunk(
                id = "page-$page",
                ordinal = page,
                text = "content-$page ".repeat(45),
                source = AttachmentSourceRef(pageNumber = page),
                estimatedTokens = 500
            )
        }
        val plan = AttachmentPromptPlanner(fits).plan(
            chunks,
            "Summarize the source",
            inputTokenBudget = limit
        )
        val synthesis = planAttachmentSynthesisBatches(
            texts = plan.batches.mapIndexed { index, _ -> "[p. ${index + 1}] result ".repeat(25) },
            originalPrompt = "Summarize the source",
            promptFits = fits
        )

        assertTrue(plan.batches.all { fits(it.prompt) })
        assertTrue(synthesis.all { fits(it.prompt) })
        assertTrue(plan.batches.flatMap { it.chunks }.all { chunk -> "[${chunk.source.label()}]" in plan.batches.first { chunk in it.chunks }.prompt })
        assertTrue(generatedRequests.isNotEmpty())
        assertTrue(generatedRequests.all { prompt ->
            serializer.serializeAll(
                history + ChatTurn(role = ChatRole.USER, text = prompt),
                "long system instruction with policy text"
            ).size <= limit
        })
    }

    private class ExactFakeBackend(
        private val serializer: QwenChatPromptSerializer
    ) : ChatBackend {
        override val capabilities = BackendCapabilities(contextWindowTokens = 512)

        override fun estimateSerializedPromptTokens(request: InferenceRequest): Int {
            return serializer.serializeAll(request.history, request.modelInstruction).size
        }

        override suspend fun initialize() = Unit

        override suspend fun resetConversation(
            history: List<ChatTurn>,
            thinkingEnabled: Boolean,
            modelInstruction: String
        ) = Unit

        override suspend fun streamReply(
            request: InferenceRequest,
            onPartial: (BackendResponse) -> Unit
        ): BackendResponse = BackendResponse("")

        override fun cancelGeneration() = Unit

        override fun close() = Unit
    }
}
