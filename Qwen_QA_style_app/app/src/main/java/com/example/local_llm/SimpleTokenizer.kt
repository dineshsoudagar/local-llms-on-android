package com.example.local_llm

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.InputStream
import java.text.Normalizer

class BpeTokenizer(context: Context) {

    companion object {
        private val PRETOKENIZE_REGEX = Regex(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        )
    }

    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>
    private val merges: List<Pair<String, String>>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val specialTokens: Map<String, Int>
    private val nfcNormalize: Boolean
    private val byteEncoder: Map<Int, Char> = bytesToUnicode()
    private val bpeCache = mutableMapOf<String, List<String>>()

    init {
        val tokenizerJson = loadTokenizerJson(context)

        // Read vocab
        vocab = tokenizerJson.getJSONObject("model")
            .getJSONObject("vocab").toIntMap()
        idToToken = vocab.entries.associate { (k, v) -> v to k }

        // Read merges
        val mergeList = tokenizerJson.getJSONObject("model").getJSONArray("merges")
        merges = (0 until mergeList.length()).map { i ->
            when (val entry = mergeList.get(i)) {
                is String -> {
                    val parts = entry.split(" ")
                    require(parts.size == 2) { "Invalid merge string: $entry" }
                    parts[0] to parts[1]
                }
                is org.json.JSONArray -> {
                    require(entry.length() == 2) { "Invalid merge array: $entry" }
                    entry.getString(0) to entry.getString(1)
                }
                else -> throw IllegalArgumentException("Unsupported merge entry type: ${entry::class.java}")
            }
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

        preTokenize(processed).forEach { piece ->
            val byteLevelPiece = encodeToByteLevel(piece)
            val bpeTokens = bpe(byteLevelPiece)
            bpeTokens.forEach { bpeToken ->
                val tokenId = vocab[bpeToken]
                    ?: throw IllegalArgumentException(
                        "Token piece '$bpeToken' from '$piece' was not found in the tokenizer vocab."
                    )
                tokens.add(tokenId)
            }
        }

        if (addSpecialTokens) {
            specialTokens["<|im_end|>"]?.let { tokens.add(it) }
        }
        Log.d("tokenized", "ID=${tokens}")

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

    private val decodedTokenCache: Map<Int, String> = buildMap {
        idToToken.forEach { (id, token) ->
            val decoded = when {
                token == "Ċ" -> "\n"
                token == "▁" -> " "
                token.startsWith("Ġ") -> " " + token.removePrefix("Ġ")
                token.contains("Ċ") -> token.replace("Ċ", "\n")
                token.contains("▁") -> token.replace("▁", " ")
                else -> token
            }
            put(id, decoded)
        }

        specialTokens.forEach { (key, id) ->
            putIfAbsent(id, key)
        }
    }

    fun decodeSingleToken(tokenId: Int): String {
        return decodedTokenCache[tokenId] ?: "<unk>"
    }

    fun getTokenId(token: String): Int {
        return specialTokens[token]
            ?: vocab[token]
            ?: throw IllegalArgumentException("Token '$token' not found in vocab or special tokens.")
    }


    private fun preTokenize(text: String): List<String> {
        return PRETOKENIZE_REGEX.findAll(text).map { it.value }.toList()
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return (0 until word.size - 1).map { Pair(word[it], word[it + 1]) }.toSet()
    }

    private fun bpe(token: String): List<String> {
        bpeCache[token]?.let { return it }

        var word = token.toCharArray().map { it.toString() }.toMutableList()
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
        return word.also { bpeCache[token] = it }
    }

    private fun encodeToByteLevel(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        val builder = StringBuilder(bytes.size)
        for (byteValue in bytes) {
            val mappedChar = byteEncoder[byteValue.toInt() and 0xFF]
                ?: throw IllegalStateException("Byte ${byteValue.toInt() and 0xFF} is missing from the byte encoder.")
            builder.append(mappedChar)
        }
        return builder.toString()
    }

    private fun bytesToUnicode(): Map<Int, Char> {
        val bs = mutableListOf<Int>()
        bs.addAll(33..126)
        bs.addAll(161..172)
        bs.addAll(174..255)

        val cs = bs.toMutableList()
        var extra = 0
        for (byteValue in 0..255) {
            if (byteValue !in bs) {
                bs.add(byteValue)
                cs.add(256 + extra)
                extra += 1
            }
        }

        return bs.zip(cs.map(Int::toChar)).toMap()
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
