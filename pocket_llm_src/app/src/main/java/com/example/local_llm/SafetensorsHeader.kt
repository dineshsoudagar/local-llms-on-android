package com.example.local_llm

import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Reads only the tensor names of a .safetensors file. stable-diffusion.cpp needs a complete single-file
 * checkpoint (diffusion model, text encoder and VAE); a UNet-only file fails to load with no useful reason.
 */
object SafetensorsHeader {
    private const val MAX_HEADER_BYTES = 32L * 1024 * 1024
    private val TEXT_ENCODER_MARKERS = listOf("cond_stage_model", "conditioner", "text_model", "text_encoders", "t5xxl", "clip_l")

    fun hasTextEncoder(keys: Set<String>): Boolean =
        keys.any { key -> TEXT_ENCODER_MARKERS.any { key.contains(it) } }

    fun parseKeys(header: String): Set<String> =
        JSONObject(header).keys().asSequence().filter { it != "__metadata__" }.toSet()

    private fun headerLength(first8: ByteArray): Long {
        var length = 0L
        for (i in 7 downTo 0) length = (length shl 8) or (first8[i].toLong() and 0xff)
        return length
    }

    fun readLocal(file: File): Set<String>? = runCatching {
        RandomAccessFile(file, "r").use { input ->
            val first = ByteArray(8).also { input.readFully(it) }
            val length = headerLength(first)
            if (length <= 0 || length > MAX_HEADER_BYTES) return null
            val header = ByteArray(length.toInt()).also { input.readFully(it) }
            parseKeys(String(header, Charsets.UTF_8))
        }
    }.getOrNull()

    /** Null when the header could not be read (offline, unexpected format): callers then do not block. */
    suspend fun readRemote(url: String): Set<String>? = withContext(Dispatchers.IO) {
        runCatching {
            val first = range(url, 0, 7)
            val length = headerLength(first)
            if (first.size != 8 || length <= 0 || length > MAX_HEADER_BYTES) return@runCatching null
            parseKeys(String(range(url, 8, 8 + length - 1), Charsets.UTF_8))
        }.getOrNull()
    }

    private fun range(url: String, from: Long, to: Long): ByteArray {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            setRequestProperty("Range", "bytes=$from-$to")
        }
        try {
            return connection.inputStream.use { it.readBytes() }
        } finally {
            connection.disconnect()
        }
    }
}
