package com.example.deen_translator

import android.content.Context
import org.json.JSONObject
import java.text.Normalizer
import java.io.InputStream

class BpeTokenizer(context: Context) {

    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>
    private val merges: List<Pair<String, String>>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val specialTokens: Map<String, Int>
    private val nfcNormalize: Boolean

    init {
        val tokenizerJson = loadTokenizerJson(context)

        // Read vocab
        vocab = tokenizerJson.getJSONObject("model")
            .getJSONObject("vocab").toIntMap()
        idToToken = vocab.entries.associate { (k, v) -> v to k }

        // Read merges
        val mergeList = tokenizerJson.getJSONObject("model").getJSONArray("merges")
        merges = (0 until mergeList.length()).map {
            val pairArray = mergeList.getJSONArray(it)
            Pair(pairArray.getString(0), pairArray.getString(1))
        }
        bpeRanks = merges.withIndex().associate { it.value to it.index }

        // Special tokens
        val addedTokens = tokenizerJson.optJSONArray("added_tokens")
        specialTokens = if (addedTokens != null) {
            (0 until addedTokens.length()).associate {
                val obj = addedTokens.getJSONObject(it)
                obj.getString("content") to obj.getInt("id")
            }
        } else emptyMap()

        // Normalize?
        nfcNormalize = tokenizerJson.optJSONObject("normalizer")
            ?.optString("type") == "NFC"
    }

    fun tokenize(text: String, addSpecialTokens: Boolean = false): IntArray {
        val tokens = mutableListOf<Int>()

        if (addSpecialTokens) {
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
        }

        val processed = if (nfcNormalize) Normalizer.normalize(text, Normalizer.Form.NFC) else text
        val preTokens = preTokenize(processed)
        for (token in preTokens) {
            if (specialTokens.containsKey(token)) {
                tokens.add(specialTokens[token]!!)
            } else {
                val bpeTokens = bpe(token)
                bpeTokens.forEach { bpeToken ->
                    // Only add token if it's in the vocab
                    vocab[bpeToken]?.let { tokens.add(it) }
                }
            }
        }

        if (addSpecialTokens) {
            specialTokens["<|im_end|>"]?.let { tokens.add(it) }
        }

        return tokens.toIntArray()
    }

    fun decode(tokenIds: IntArray): String {
        val builder = StringBuilder()
        for (id in tokenIds) {
            val token = idToToken[id]
            if (token != null) {
                builder.append(token)
            } else {
                val special = specialTokens.entries.find { it.value == id }?.key
                if (special != null) {
                    builder.append(special)
                } else {
                    builder.append("<unk>")
                }
            }
        }
        val raw = builder.toString()
        val cleaned = raw
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("▁", " ")
        return if (nfcNormalize) Normalizer.normalize(cleaned, Normalizer.Form.NFC) else cleaned
    }

    private fun preTokenize(text: String): List<String> {
        val regex = Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
        return regex.findAll(text).map { it.value }.toList()
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return (0 until word.size - 1).map { Pair(word[it], word[it + 1]) }.toSet()
    }

    private fun bpe(token: String): List<String> {
        var word = token.map { it.toString() }.toMutableList()
        var pairs = getPairs(word)
        while (true) {
            val best = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(best)) break

            val (first, second) = best
            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            pairs = getPairs(word)
        }
        return word
    }

    private fun loadTokenizerJson(context: Context): JSONObject {
        val inputStream: InputStream = context.assets.open("tokenizer.json")
        val jsonStr = inputStream.bufferedReader().use { it.readText() }
        return JSONObject(jsonStr)
    }

    private fun JSONObject.toIntMap(): Map<String, Int> {
        return keys().asSequence().associateWith { this.getInt(it) }
    }
}
