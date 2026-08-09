package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AttachmentTextPipelineTest {
    @Test
    fun normalizationPreservesParagraphsAndUnicode() {
        val normalized = AttachmentTextNormalizer.normalize("Cafe\u0301  heading\r\n\r\nRow\tvalue")

        assertEquals("Café heading\n\nRow value", normalized)
    }

    @Test
    fun chunkingPreservesPageReferencesAndBudget() {
        val chunks = AttachmentChunker { text -> text.split(Regex("\\s+")).size }.chunk(
            sections = listOf(
                AttachmentTextSection(
                    text = (1..30).joinToString(" ") { "word$it" },
                    source = AttachmentSourceRef(pageNumber = 7)
                )
            ),
            targetTokens = 10,
            overlapTokens = 2
        )

        assertTrue(chunks.size > 1)
        assertTrue(chunks.all { it.source.pageNumber == 7 })
        assertTrue(chunks.all { it.estimatedTokens <= 10 })
    }

    @Test
    fun bm25RanksRelevantChunkAndReturnsSourceOrder() {
        val chunks = listOf(
            chunk(0, "oranges and apples", 1),
            chunk(1, "android audio encoder details", 2),
            chunk(2, "audio encoder fallback rules", 3)
        )

        val selected = Bm25AttachmentRetriever(chunks).retrieve("audio encoder", 8)

        assertFalse(selected.isEmpty())
        assertTrue(selected.all { it.ordinal in setOf(1, 2) })
        assertEquals(selected.sortedBy { it.ordinal }, selected)
    }

    @Test
    fun routerSeparatesSummaryQuestionAndTransformation() {
        assertEquals(AttachmentTask.SUMMARY, AttachmentTaskRouter.route("Summarize the whole PDF"))
        assertEquals(AttachmentTask.TRANSFORMATION, AttachmentTaskRouter.route("Translate all of this"))
        assertEquals(AttachmentTask.QUESTION, AttachmentTaskRouter.route("Who signed it?"))
    }

    @Test
    fun summaryPlannerCoversEveryChunkInOrderedBatches() {
        val chunks = (0 until 5).map { index -> chunk(index, "section $index text", index + 1, tokens = 5) }

        val plan = AttachmentPromptPlanner().plan(chunks, "Summarize everything", inputTokenBudget = 10)

        assertTrue(plan.requiresFinalSynthesis)
        assertEquals(chunks.map { it.id }, plan.batches.flatMap { it.chunks }.map { it.id })
    }

    @Test
    fun overlapDeduplicationRemovesRepeatedLeadingWords() {
        val next = deduplicateSegmentOverlap(
            "the quick brown fox jumps over",
            "brown fox jumps over the fence"
        )

        assertEquals("the fence", next)
    }

    private fun chunk(ordinal: Int, text: String, page: Int, tokens: Int = 3) = AttachmentChunk(
        id = "chunk-$ordinal",
        ordinal = ordinal,
        text = text,
        source = AttachmentSourceRef(pageNumber = page),
        estimatedTokens = tokens
    )
}
