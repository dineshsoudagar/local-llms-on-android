package com.example.local_llm

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

/** Owns private copies of user-selected LiteRT-LM files; no provider URI is retained. */
class CustomLiteRtModelStore(private val context: Context) {
    private val modelsDir: File get() = File(context.filesDir, "models")

    fun loadAll(): List<CustomLiteRtSpec> = modelsDir.listFiles().orEmpty()
        .filter { it.isDirectory && it.name.startsWith("custom_") }
        .mapNotNull { directory ->
            runCatching {
                val metadata = JSONObject(File(directory, "custom_model.json").readText())
                val file = File(directory, "model.litertlm")
                val expectedBytes = metadata.getLong("bytes")
                if (!file.isFile || expectedBytes <= 0L) return@runCatching null
                CustomLiteRtSpec(file.name, expectedBytes, directory.name, metadata.getString("name"))
            }.getOrNull()
        }

    fun import(uri: Uri): CustomLiteRtSpec {
        val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }?.substringAfterLast('/') ?: throw IllegalArgumentException("Could not read the selected file name.")
        require(name.endsWith(".litertlm", ignoreCase = true)) {
            "Choose a .litertlm model file."
        }

        val id = "custom_${UUID.randomUUID()}"
        val directory = File(modelsDir, id)
        check(directory.mkdirs()) { "Could not create model storage." }
        try {
            val temporary = File(directory, "model.litertlm.importing")
            val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(temporary).use { output ->
                    val copied = input.copyTo(output)
                    output.fd.sync()
                    copied
                }
            } ?: throw IllegalArgumentException("Could not open the selected file.")
            require(bytes > 0L) { "The selected model file is empty." }
            check(temporary.renameTo(File(directory, "model.litertlm"))) {
                "Could not finish importing the model."
            }
            val displayName = name.substringBeforeLast('.').ifBlank { "Local model" }
            File(directory, "custom_model.json").writeText(
                JSONObject().put("name", displayName).put("bytes", bytes).toString()
            )
            return CustomLiteRtSpec("model.litertlm", bytes, id, displayName)
        } catch (error: Throwable) {
            directory.deleteRecursively()
            throw error
        }
    }
}
