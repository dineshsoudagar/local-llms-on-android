package com.example.local_llm

import kotlin.math.ceil

/**
 * Upper-bound token estimate for LiteRT prompts, which expose no tokenizer before prefill.
 * Counting every UTF-8 byte as a token was safe but used only about a quarter of the context
 * for English chat. These weights stay above what Gemma and Qwen tokenizers produce for common
 * text: digits split one per token, letters and spaces merge into words, and non-Latin scripts
 * are counted per character.
 */
object PromptTokenEstimator {
    private const val WORD_CHAR_WEIGHT = 0.3
    private const val PUNCTUATION_WEIGHT = 0.6
    private const val NEWLINE_WEIGHT = 1.0
    private const val DIGIT_WEIGHT = 1.0
    private const val OTHER_CHAR_WEIGHT = 1.0
    private const val SUPPLEMENTARY_CHAR_WEIGHT = 3.0
    private const val FIXED_OVERHEAD = 4

    /** Images and audio clips are not in the text; reserve room for their encoder tokens. */
    const val TOKENS_PER_IMAGE = 1_280
    const val TOKENS_PER_AUDIO_CLIP = 1_280

    fun estimate(text: String): Int {
        if (text.isEmpty()) return 0
        var weight = 0.0
        var index = 0
        while (index < text.length) {
            val codePoint = text.codePointAt(index)
            weight += when {
                codePoint == '\n'.code -> NEWLINE_WEIGHT
                codePoint in '0'.code..'9'.code -> DIGIT_WEIGHT
                codePoint < 128 && (Character.isLetter(codePoint) || Character.isWhitespace(codePoint)) -> WORD_CHAR_WEIGHT
                codePoint < 128 -> PUNCTUATION_WEIGHT
                Character.isSupplementaryCodePoint(codePoint) -> SUPPLEMENTARY_CHAR_WEIGHT
                else -> OTHER_CHAR_WEIGHT
            }
            index += Character.charCount(codePoint)
        }
        return ceil(weight).toInt() + FIXED_OVERHEAD
    }
}

/**
 * Raises the estimate when the runtime reports more tokens than predicted. The scale only grows,
 * with a margin, so a measured under-estimate never repeats in this model session.
 */
class TokenEstimateCalibration {
    @Volatile
    var scale: Double = 1.0
        private set

    fun apply(rawEstimate: Int): Int = ceil(rawEstimate * scale).toInt()

    fun observe(rawEstimate: Int, measuredTokens: Int) {
        if (rawEstimate <= 0 || measuredTokens <= 0) return
        val needed = measuredTokens * SAFETY_MARGIN / rawEstimate
        if (needed > scale) scale = needed.coerceAtMost(MAX_SCALE)
    }

    private companion object {
        const val SAFETY_MARGIN = 1.1
        const val MAX_SCALE = 4.0
    }
}

/** LiteRT plain chat passes no output reserve, so keep room for a reply of reasonable length. */
internal fun liteRtMinimumOutputReserve(contextTokens: Int): Int = (contextTokens / 8).coerceIn(256, 2_048)
