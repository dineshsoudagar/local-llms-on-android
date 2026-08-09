package com.example.local_llm

import java.text.Normalizer
import kotlin.math.ln

data class AttachmentTextSection(
    val text: String,
    val source: AttachmentSourceRef
)

data class AttachmentPromptBatch(
    val chunks: List<AttachmentChunk>,
    val prompt: String
)

data class AttachmentPromptPlan(
    val task: AttachmentTask,
    val batches: List<AttachmentPromptBatch>,
    val requiresFinalSynthesis: Boolean
)

object AttachmentTextNormalizer {
    fun normalize(raw: String): String {
        val normalized = Normalizer.normalize(raw, Normalizer.Form.NFC)
            .replace("\r\n", "\n")
            .replace('\r', '\n')
            .replace('\u0000', ' ')
        return normalized
            .lineSequence()
            .map { line -> line.replace(Regex("[\\t\\x0B\\f ]+"), " ").trimEnd() }
            .joinToString("\n")
            .replace(Regex("\n{4,}"), "\n\n\n")
            .trim()
    }
}

object AttachmentTaskRouter {
    private val summaryWords = setOf("summarize", "summary", "overview", "outline", "analyse", "analyze")
    private val transformationWords = setOf(
        "rewrite", "translate", "convert", "reformat", "extract all", "proofread", "paraphrase"
    )

    fun route(prompt: String): AttachmentTask {
        val normalized = prompt.lowercase()
        return when {
            transformationWords.any(normalized::contains) -> AttachmentTask.TRANSFORMATION
            summaryWords.any(normalized::contains) -> AttachmentTask.SUMMARY
            else -> AttachmentTask.QUESTION
        }
    }
}

class AttachmentChunker(
    private val estimateTokens: (String) -> Int = ::conservativeTokenEstimate
) {
    fun chunk(
        sections: List<AttachmentTextSection>,
        targetTokens: Int,
        overlapTokens: Int = (targetTokens / 10).coerceAtLeast(8)
    ): List<AttachmentChunk> {
        require(targetTokens > 0)
        val result = mutableListOf<AttachmentChunk>()
        var ordinal = 0

        sections.forEach { section ->
            val paragraphs = section.text
                .split(Regex("\n{2,}"))
                .map(AttachmentTextNormalizer::normalize)
                .filter(String::isNotBlank)
            var current = StringBuilder()

            fun flush() {
                val text = current.toString().trim()
                if (text.isBlank()) return
                result += AttachmentChunk(
                    ordinal = ordinal++,
                    text = text,
                    source = section.source,
                    estimatedTokens = estimateTokens(text)
                )
                val overlap = takeTailWithinBudget(text, overlapTokens, estimateTokens)
                current = StringBuilder(overlap)
            }

            paragraphs.forEach { paragraph ->
                if (estimateTokens(paragraph) > targetTokens) {
                    if (current.isNotBlank()) flush()
                    splitOversizedParagraph(paragraph, targetTokens, overlapTokens, estimateTokens)
                        .forEach { piece ->
                            result += AttachmentChunk(
                                ordinal = ordinal++,
                                text = piece,
                                source = section.source,
                                estimatedTokens = estimateTokens(piece)
                            )
                        }
                    current = StringBuilder()
                } else {
                    val candidate = if (current.isBlank()) paragraph else "$current\n\n$paragraph"
                    if (estimateTokens(candidate) > targetTokens && current.isNotBlank()) flush()
                    if (current.isNotBlank()) current.append("\n\n")
                    current.append(paragraph)
                }
            }
            if (current.isNotBlank()) {
                val text = current.toString().trim()
                result += AttachmentChunk(
                    ordinal = ordinal++,
                    text = text,
                    source = section.source,
                    estimatedTokens = estimateTokens(text)
                )
            }
        }
        return result
    }

    private fun splitOversizedParagraph(
        paragraph: String,
        targetTokens: Int,
        overlapTokens: Int,
        estimator: (String) -> Int
    ): List<String> {
        val words = paragraph.split(Regex("\\s+")).filter(String::isNotBlank)
        if (words.isEmpty()) return emptyList()
        val pieces = mutableListOf<String>()
        var start = 0
        while (start < words.size) {
            var end = start + 1
            while (end <= words.size && estimator(words.subList(start, end).joinToString(" ")) <= targetTokens) {
                end++
            }
            end = (end - 1).coerceAtLeast(start + 1)
            pieces += words.subList(start, end).joinToString(" ")
            if (end >= words.size) break
            var nextStart = end
            while (nextStart > start && estimator(words.subList(nextStart - 1, end).joinToString(" ")) < overlapTokens) {
                nextStart--
            }
            start = nextStart.coerceAtMost(end - 1)
        }
        return pieces
    }

    private fun takeTailWithinBudget(
        text: String,
        budget: Int,
        estimator: (String) -> Int
    ): String {
        if (budget <= 0) return ""
        val words = text.split(Regex("\\s+")).filter(String::isNotBlank)
        var start = words.size
        while (start > 0 && estimator(words.subList(start - 1, words.size).joinToString(" ")) <= budget) {
            start--
        }
        return words.subList(start, words.size).joinToString(" ")
    }
}

class Bm25AttachmentRetriever(private val chunks: List<AttachmentChunk>) {
    private val tokenized = chunks.map { tokenize(it.text) }
    private val averageLength = tokenized.map(List<String>::size).average().takeIf { !it.isNaN() } ?: 0.0
    private val documentFrequency = buildMap<String, Int> {
        tokenized.forEach { document -> document.toSet().forEach { token -> put(token, getOrDefault(token, 0) + 1) } }
    }

    fun retrieve(query: String, tokenBudget: Int): List<AttachmentChunk> {
        if (chunks.isEmpty() || tokenBudget <= 0) return emptyList()
        val queryTokens = tokenize(query).distinct()
        val ranked = chunks.indices
            .map { index -> chunks[index] to score(tokenized[index], queryTokens) }
            .sortedWith(compareByDescending<Pair<AttachmentChunk, Double>> { it.second }.thenBy { it.first.ordinal })
        val selected = mutableListOf<AttachmentChunk>()
        var used = 0
        for ((chunk, _) in ranked) {
            if (used + chunk.estimatedTokens > tokenBudget && selected.isNotEmpty()) continue
            selected += chunk
            used += chunk.estimatedTokens
            if (used >= tokenBudget) break
        }
        return selected.sortedBy(AttachmentChunk::ordinal)
    }

    private fun score(document: List<String>, query: List<String>): Double {
        if (document.isEmpty() || query.isEmpty()) return 0.0
        val frequencies = document.groupingBy { it }.eachCount()
        return query.sumOf { term ->
            val frequency = frequencies[term]?.toDouble() ?: 0.0
            if (frequency == 0.0) return@sumOf 0.0
            val frequencyInDocuments = documentFrequency[term]?.toDouble() ?: 0.0
            val inverseDocumentFrequency = ln(1.0 + (chunks.size - frequencyInDocuments + 0.5) / (frequencyInDocuments + 0.5))
            val lengthNormalization = 1.2 * (1.0 - 0.75 + 0.75 * document.size / averageLength.coerceAtLeast(1.0))
            inverseDocumentFrequency * (frequency * 2.2) / (frequency + lengthNormalization)
        }
    }

    private fun tokenize(text: String): List<String> {
        return Regex("[\\p{L}\\p{N}']+")
            .findAll(text.lowercase())
            .map { it.value }
            .filter { it.length > 1 }
            .toList()
    }
}

class AttachmentPromptPlanner {
    fun plan(chunks: List<AttachmentChunk>, userPrompt: String, inputTokenBudget: Int): AttachmentPromptPlan {
        require(inputTokenBudget > 0)
        val task = AttachmentTaskRouter.route(userPrompt)
        val allTokens = chunks.sumOf(AttachmentChunk::estimatedTokens)
        val chosenGroups = when {
            allTokens <= inputTokenBudget -> listOf(chunks)
            task == AttachmentTask.QUESTION -> listOf(Bm25AttachmentRetriever(chunks).retrieve(userPrompt, inputTokenBudget))
            else -> packInOrder(chunks, inputTokenBudget)
        }
        return AttachmentPromptPlan(
            task = task,
            batches = chosenGroups.mapIndexed { index, group ->
                AttachmentPromptBatch(group, buildBatchPrompt(task, userPrompt, index, chosenGroups.size, group))
            },
            requiresFinalSynthesis = chosenGroups.size > 1
        )
    }

    private fun packInOrder(chunks: List<AttachmentChunk>, budget: Int): List<List<AttachmentChunk>> {
        val groups = mutableListOf<MutableList<AttachmentChunk>>()
        var used = 0
        chunks.forEach { chunk ->
            if (groups.isEmpty() || used + chunk.estimatedTokens > budget) {
                groups += mutableListOf<AttachmentChunk>()
                used = 0
            }
            groups.last() += chunk
            used += chunk.estimatedTokens
        }
        return groups
    }

    private fun buildBatchPrompt(
        task: AttachmentTask,
        userPrompt: String,
        batchIndex: Int,
        batchCount: Int,
        chunks: List<AttachmentChunk>
    ): String = buildString {
        append("Use only the attached source excerpts below. Cite their bracketed source labels.\n")
        if (batchCount > 1) append("This is ordered batch ${batchIndex + 1} of $batchCount.\n")
        append("User request: $userPrompt\n\n")
        chunks.forEach { chunk -> append("[${chunk.source.label()}]\n${chunk.text}\n\n") }
        if (task == AttachmentTask.SUMMARY && batchCount > 1) {
            append("Return a faithful intermediate summary for later synthesis, preserving important source labels.")
        }
    }.trim()
}

fun conservativeTokenEstimate(text: String): Int {
    if (text.isBlank()) return 0
    return ((text.length + 3) / 4).coerceAtLeast(1)
}

fun deduplicateSegmentOverlap(previous: String, next: String, maxWords: Int = 40): String {
    val left = previous.trim().split(Regex("\\s+")).filter(String::isNotBlank)
    val right = next.trim().split(Regex("\\s+")).filter(String::isNotBlank)
    val maxOverlap = minOf(maxWords, left.size, right.size)
    for (size in maxOverlap downTo 2) {
        if (left.takeLast(size).map(String::lowercase) == right.take(size).map(String::lowercase)) {
            return right.drop(size).joinToString(" ")
        }
    }
    return next.trim()
}
