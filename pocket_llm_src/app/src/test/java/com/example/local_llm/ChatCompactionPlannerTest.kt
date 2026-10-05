package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatCompactionPlannerTest {
    private fun user(id: String, size: Int = 100) = ChatTurn(id = id, role = ChatRole.USER, text = "u".repeat(size))
    private fun assistant(id: String, size: Int = 100) = ChatTurn(id = id, role = ChatRole.ASSISTANT, text = "a".repeat(size))
    private val estimate: (ChatTurn) -> Int = { it.text.length }

    private fun conversation(exchanges: Int, size: Int = 100): List<ChatTurn> =
        (0 until exchanges).flatMap { listOf(user("u$it", size), assistant("a$it", size)) } + user("last", size)

    @Test
    fun triggersOnlyPastThreeQuartersOfTheContext() {
        assertFalse(ChatCompactionPlanner.shouldCompact(750, 1_000))
        assertTrue(ChatCompactionPlanner.shouldCompact(751, 1_000))
    }

    @Test
    fun tailKeepsRecentTurnsAndStartsAtAUserTurn() {
        val turns = conversation(exchanges = 10)
        val start = ChatCompactionPlanner.tailStart(turns, contextTokens = 1_000, estimateTurn = estimate)

        assertTrue(start > 0)
        assertEquals(ChatRole.USER, turns[start].role)
        assertTrue(turns.subList(start, turns.size).sumOf(estimate) <= 350)
        assertEquals("last", turns.last().id)
    }

    @Test
    fun oversizedLatestTurnIsStillKeptWhole() {
        val turns = conversation(exchanges = 3) + listOf(assistant("big-answer", 5_000), user("final", 100))
        val start = ChatCompactionPlanner.tailStart(turns, contextTokens = 1_000, estimateTurn = estimate)

        assertEquals("final", turns[start].id)
    }

    @Test
    fun nothingToSummarizeWhenOnlyTheLatestExchangeExists() {
        assertEquals(0, ChatCompactionPlanner.tailStart(listOf(user("only")), 1_000, estimate))
    }

    @Test
    fun visibleTurnsSkipSummarizedOnesAndIgnoreStaleCompaction() {
        val turns = conversation(exchanges = 3)
        val compaction = ChatCompaction("summary", "a1")

        assertEquals(listOf("u2", "a2", "last"), ChatCompactionPlanner.visibleTurns(turns, compaction).map { it.id })
        assertTrue(ChatCompactionPlanner.isActive(turns, compaction))

        val stale = ChatCompaction("summary", "missing")
        assertEquals(turns, ChatCompactionPlanner.visibleTurns(turns, stale))
        assertFalse(ChatCompactionPlanner.isActive(turns, stale))
    }

    @Test
    fun batchesRespectTheBudgetAndKeepOrder() {
        val turns = conversation(exchanges = 5)
        val batches = ChatCompactionPlanner.batches(turns, budgetTokens = 250, estimateTurn = estimate)

        assertEquals(turns, batches.flatten())
        assertTrue(batches.all { batch -> batch.sumOf(estimate) <= 250 })
    }

    @Test
    fun summaryIsStrippedOfThinkingAndCapped() {
        val raw = "<think>planning</think>First fact. Second fact. Third fact that runs long."
        assertEquals("First fact. Second fact. Third fact that runs long.", ChatCompactionPlanner.cleanSummary(raw, 500))
        assertEquals("First fact. Second fact.", ChatCompactionPlanner.cleanSummary(raw, 30))
    }

    @Test
    fun summaryIsAppendedToTheInstruction() {
        val instruction = ChatCompactionPlanner.instructionWithSummary("Be helpful.", "User likes tea.")
        assertTrue(instruction.startsWith("Be helpful."))
        assertTrue(instruction.endsWith("User likes tea."))
        assertEquals("Be helpful.", ChatCompactionPlanner.instructionWithSummary("Be helpful.", null))
    }

    @Test
    fun promptIncludesPreviousSummaryAndSpeakers() {
        val prompt = ChatCompactionPlanner.summaryPrompt("Earlier stuff.", listOf(user("u", 3), assistant("a", 3)), 100, 50)
        assertTrue(prompt.contains("Earlier stuff."))
        assertTrue(prompt.contains("User: uuu"))
        assertTrue(prompt.contains("Assistant: aaa"))
    }
}
