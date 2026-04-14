package com.example.local_llm

import android.content.Context
import android.util.Log
import org.json.JSONObject

class TokenDisplayMapper(
    context: Context,
    modelName: String,
    assetFilename: String? = null
) {

    private val tokenToDisplay: Map<Int, String> = if (modelName.startsWith("Qwen", ignoreCase = true)) {
        try {
            val mappingAsset = assetFilename
                ?: throw IllegalArgumentException("Qwen models require a token display mapping asset.")
            val inputStream = context.assets.open(mappingAsset)
            val jsonStr = inputStream.bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonStr)
            val mapJson = jsonObject.getJSONObject("token_to_display")

            mapJson.keys().asSequence().associate { key ->
                key.toInt() to mapJson.getString(key)
            }
        } catch (e: Exception) {
            Log.e("TokenDisplayMapper", "❌ Failed to load token display mapping", e)
            throw RuntimeException("Failed to load Qwen token display mapping: ${e.message}", e)
        }
    } else {
        emptyMap()
    }

    fun map(tokenId: Int): String {
        return tokenToDisplay[tokenId] ?: "<$tokenId>"
    }
}
