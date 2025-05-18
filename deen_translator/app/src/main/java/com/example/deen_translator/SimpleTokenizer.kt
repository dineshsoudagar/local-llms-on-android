package com.example.yourapp

import android.content.Context
import org.json.JSONObject
import java.io.InputStream

class SimpleTokenizer(context: Context) {
    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>

    init {
        val vocabData = loadVocab(context)
        vocab = vocabData
        idToToken = vocabData.entries.associate { (k, v) -> v to k }
    }

    private fun loadVocab(context: Context): Map<String, Int> {
        val map = mutableMapOf<String, Int>()
        try {
            val inputStream: InputStream = context.assets.open("vocab.json")
            val json = inputStream.bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(json)
            val keys = jsonObject.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                map[key] = jsonObject.getInt(key)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return map
    }

    fun tokenize(text: String): IntArray {
        val tokens = text.trim().split("\\s+".toRegex())
        return tokens.map { vocab[it] ?: vocab["<unk>"] ?: 0 }.toIntArray()
    }

    fun decode(tokenIds: IntArray): String {
        return tokenIds.map { idToToken[it] ?: "<unk>" }.joinToString(" ")
    }
}
