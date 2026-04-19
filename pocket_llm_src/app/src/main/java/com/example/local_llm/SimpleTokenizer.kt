package com.example.local_llm

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.InputStream
import java.text.Normalizer

class BpeTokenizer(
    private val context: Context,
    private val modelDescriptor: OnnxQwenSpec,
    private val modelFileResolver: ModelFileResolver
) {

    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>
    private val merges: List<Pair<String, String>>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val specialTokens: Map<String, Int>
    private val specialTokensById: Map<Int, String>
    private val nfcNormalize: Boolean
    private val byteToUnicode: Map<Int, Char>
    private val unicodeToByte: Map<Char, Int>
    private val splitPattern: Regex
    private val bpeCache = mutableMapOf<String, List<String>>()

    companion object {
        private const val TAG = "BpeTokenizer"
        private val DEFAULT_SPLIT_PATTERN = Regex(
            """'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    }

    inner class StreamDecoder {
        private val pendingBytes = ArrayList<Byte>()

        fun append(tokenId: Int): String {
            val token = idToToken[tokenId]
            if (token == null) {
                val special = specialTokensById[tokenId]
                val flushed = flushPending(force = true)
                return flushed + (special ?: "<unk>")
            }

            pendingBytes.addAll(encodedToBytes(token).toList())
            return flushPending(force = false)
        }

        fun flush(): String {
            return flushPending(force = true)
        }

        private fun flushPending(force: Boolean): String {
            if (pendingBytes.isEmpty()) return ""

            val bytes = pendingBytes.toByteArray()
            val completeLength = if (force) {
                bytes.size
            } else {
                findCompleteUtf8PrefixLength(bytes)
            }

            if (completeLength <= 0) {
                return ""
            }

            val decoded = bytes.copyOfRange(0, completeLength).toString(Charsets.UTF_8)
            repeat(completeLength) { pendingBytes.removeAt(0) }
            return if (nfcNormalize) Normalizer.normalize(decoded, Normalizer.Form.NFC) else decoded
        }
    }

    init {
        val tokenizerJson = loadTokenizerJson()

        // Load base vocabulary
        vocab = tokenizerJson.getJSONObject("model").getJSONObject("vocab").toIntMap()
        idToToken = vocab.entries.associate { (k, v) -> v to k }

        // Load BPE merge rules
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

        // Load special tokens like <|im_start|> and <|im_end|>
        val addedTokens = tokenizerJson.optJSONArray("added_tokens")
        specialTokens = if (addedTokens != null) {
            (0 until addedTokens.length()).associate {
                val obj = addedTokens.getJSONObject(it)
                obj.getString("content") to obj.getInt("id")
            }
        } else emptyMap()
        specialTokensById = specialTokens.entries.associate { (token, id) -> id to token }

        // Check if NFC normalization is enabled
        nfcNormalize = tokenizerJson.optJSONObject("normalizer")
            ?.optString("type") == "NFC"

        byteToUnicode = buildByteToUnicodeMap()
        unicodeToByte = byteToUnicode.entries.associate { (byte, symbol) -> symbol to byte }
        splitPattern = loadSplitPattern(tokenizerJson)

        // Log tokenizer summary
        Log.d(TAG, "Tokenizer loaded successfully: vocab=${vocab.size}, merges=${merges.size}, specialTokens=${specialTokens.size}, NFC=$nfcNormalize")
    }

    // Converts input text into a list of token IDs using BPE and optional special tokens
    fun tokenize(text: String, addSpecialTokens: Boolean = false): IntArray {
        val tokens = mutableListOf<Int>()

        if (addSpecialTokens) {
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
        }

        val processed = if (nfcNormalize) Normalizer.normalize(text, Normalizer.Form.NFC) else text

        preTokenize(processed).forEach { piece ->
            val byteLevelPiece = encodeByteLevel(piece)
            val bpeTokens = bpe(byteLevelPiece)
            bpeTokens.forEach { bpeToken ->
                val tokenId = vocab[bpeToken]
                    ?: throw IllegalArgumentException("Token '$bpeToken' not found in vocab.")
                tokens.add(tokenId)
            }
        }

        if (addSpecialTokens) {
            specialTokens["<|im_end|>"]?.let { tokens.add(it) }
        }

        Log.d(TAG, "Tokenized: \"$text\" → $tokens")
        return tokens.toIntArray()
    }

    // Decodes a list of token IDs back into a readable string
    fun decode(tokenIds: IntArray): String {
        val builder = StringBuilder()
        val pendingTokenBytes = StringBuilder()
        for (id in tokenIds) {
            val token = idToToken[id]
            if (token != null) {
                pendingTokenBytes.append(token)
            } else {
                if (pendingTokenBytes.isNotEmpty()) {
                    builder.append(decodeByteLevelString(pendingTokenBytes.toString()))
                    pendingTokenBytes.setLength(0)
                }
                val special = specialTokensById[id]
                builder.append(special ?: "<unk>")
                if (special == null) Log.w(TAG, "Unknown token ID: $id")
            }
        }
        if (pendingTokenBytes.isNotEmpty()) {
            builder.append(decodeByteLevelString(pendingTokenBytes.toString()))
        }
        val decoded = builder.toString()
        return if (nfcNormalize) Normalizer.normalize(decoded, Normalizer.Form.NFC) else decoded
    }

    // Decodes a single token ID into a string using cached values
    fun decodeSingleToken(tokenId: Int): String {
        return decodedTokenCache[tokenId] ?: "<unk>"
    }

    fun createStreamDecoder(): StreamDecoder = StreamDecoder()

    // Returns the token ID for a string (special tokens included)
    fun getTokenId(token: String): Int {
        return specialTokens[token]
            ?: vocab[token]
            ?: throw IllegalArgumentException("Token '$token' not found in vocab or special tokens.")
    }

    // Splits input string into space and word chunks before BPE merging
    private fun preTokenize(text: String): List<String> {
        if (text.isEmpty()) return emptyList()
        return splitPattern.findAll(text).map { it.value }.toList()
    }

    // Applies BPE merge rules to a single token string
    private fun bpe(token: String): List<String> {
        bpeCache[token]?.let { return it }
        var word = token.toCharArray().map { it.toString() }.toMutableList() // preTokenize
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
                    i++
                }
            }
            word = newWord
            pairs = getPairs(word)
        }

        return word.toList().also { bpeCache[token] = it }
    }

    // Returns all adjacent pairs of characters or merged subwords
    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return (0 until word.size - 1).map { word[it] to word[it + 1] }.toSet()
    }

    // Loads the tokenizer.json file from the app's assets folder
    private fun loadTokenizerJson(): JSONObject {
        val tokenizerFile = modelFileResolver.resolveFile(modelDescriptor, modelDescriptor.tokenizerAssetName)
        Log.d(TAG, "Loading tokenizer from ${tokenizerFile.absolutePath}")
        val inputStream: InputStream = tokenizerFile.inputStream()
        val jsonStr = inputStream.bufferedReader().use { it.readText() }
        return JSONObject(jsonStr)
    }

    private fun loadSplitPattern(tokenizerJson: JSONObject): Regex {
        val preTokenizer = tokenizerJson.optJSONObject("pre_tokenizer")
            ?: return DEFAULT_SPLIT_PATTERN

        val pretokenizers = preTokenizer.optJSONArray("pretokenizers")
        val candidates = if (pretokenizers != null) {
            (0 until pretokenizers.length()).mapNotNull { index ->
                pretokenizers.optJSONObject(index)
            }
        } else {
            listOf(preTokenizer)
        }

        val splitRegex = candidates.firstNotNullOfOrNull { candidate ->
            candidate.takeIf { it.optString("type").equals("Split", ignoreCase = true) }
                ?.optJSONObject("pattern")
                ?.let { pattern ->
                    pattern.optString("Regex").takeIf { it.isNotBlank() }
                        ?: pattern.optString("String").takeIf { it.isNotBlank() }
                }
        }

        return splitRegex?.let(::Regex) ?: DEFAULT_SPLIT_PATTERN
    }

    // Converts a JSON object of string→int mappings into a Kotlin map
    private fun JSONObject.toIntMap(): Map<String, Int> {
        return keys().asSequence().associateWith { this.getInt(it) }
    }

    private fun encodeByteLevel(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        val builder = StringBuilder(bytes.size)
        bytes.forEach { byte ->
            builder.append(
                byteToUnicode[byte.toInt() and 0xFF]
                    ?: throw IllegalStateException("Missing byte-level mapping for byte ${byte.toInt() and 0xFF}.")
            )
        }
        return builder.toString()
    }

    private fun decodeByteLevelString(encoded: String): String {
        if (encoded.isEmpty()) return ""
        val bytes = encodedToBytes(encoded)
        return bytes.toString(Charsets.UTF_8)
    }

    private fun encodedToBytes(encoded: String): ByteArray {
        val bytes = ByteArray(encoded.length)
        encoded.forEachIndexed { index, symbol ->
            val value = unicodeToByte[symbol]
            if (value == null) {
                Log.w(TAG, "Unknown byte-level symbol '$symbol' while decoding.")
                throw IllegalArgumentException("Unknown byte-level symbol '$symbol'.")
            }
            bytes[index] = value.toByte()
        }
        return bytes
    }

    private fun buildByteToUnicodeMap(): Map<Int, Char> {
        val bytes = mutableListOf<Int>()
        for (value in 33..126) bytes.add(value)
        for (value in 161..172) bytes.add(value)
        for (value in 174..255) bytes.add(value)

        val chars = bytes.toMutableList()
        var extra = 0
        for (value in 0..255) {
            if (value !in bytes) {
                bytes.add(value)
                chars.add(256 + extra)
                extra += 1
            }
        }

        return bytes.indices.associate { index -> bytes[index] to chars[index].toChar() }
    }

    private fun findCompleteUtf8PrefixLength(bytes: ByteArray): Int {
        var index = 0

        while (index < bytes.size) {
            val firstByte = bytes[index].toInt() and 0xFF
            val expectedLength = when {
                firstByte and 0b1000_0000 == 0 -> 1
                firstByte and 0b1110_0000 == 0b1100_0000 -> 2
                firstByte and 0b1111_0000 == 0b1110_0000 -> 3
                firstByte and 0b1111_1000 == 0b1111_0000 -> 4
                else -> 1
            }

            if (index + expectedLength > bytes.size) {
                break
            }

            var valid = true
            for (offset in 1 until expectedLength) {
                val nextByte = bytes[index + offset].toInt() and 0xFF
                if (nextByte and 0b1100_0000 != 0b1000_0000) {
                    valid = false
                    break
                }
            }

            index += if (valid) expectedLength else 1
        }

        return index
    }

    // Precomputed cache for fast single-token decoding
    private val decodedTokenCache: Map<Int, String> = buildMap {
        idToToken.forEach { (id, token) ->
            val decoded = decodeByteLevelString(token)
            put(id, decoded)
        }
        specialTokens.forEach { (key, id) -> putIfAbsent(id, key) }
    }
}
