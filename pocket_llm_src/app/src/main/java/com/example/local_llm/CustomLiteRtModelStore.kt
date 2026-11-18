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
                val url = metadata.optString("url")
                // Entries added from Hugging Face exist before their file is downloaded.
                if ((!file.isFile && url.isEmpty()) || expectedBytes <= 0L) return@runCatching null
                val inputs = metadata.optJSONObject("inputs")?.let {
                    CustomModelCapabilities(it.getBoolean("text"), it.getBoolean("vision"),
                        it.getBoolean("audio"), it.getBoolean("video"))
                }
                CustomLiteRtSpec(file.name, expectedBytes, directory.name, metadata.getString("name"), inputs, url)
            }.getOrNull()
        }

    /** Registers a .litertlm file from Hugging Face; the download service fetches it like any other model. */
    fun addRemote(displayName: String, url: String, bytes: Long): CustomLiteRtSpec {
        require(bytes > 0L) { "The file size is unknown." }
        val id = "custom_hf_${UUID.randomUUID()}"
        val directory = File(modelsDir, id)
        check(directory.mkdirs()) { "Could not create model storage." }
        File(directory, "custom_model.json").writeText(
            JSONObject().put("name", displayName).put("bytes", bytes).put("url", url).toString()
        )
        return CustomLiteRtSpec("model.litertlm", bytes, id, displayName, null, url)
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
            val inputs = inspectCustomModel(File(directory, "model.litertlm"))
            val metadata = JSONObject().put("name", displayName).put("bytes", bytes)
            inputs?.let {
                metadata.put("inputs", JSONObject().put("text", it.text).put("vision", it.vision)
                    .put("audio", it.audio).put("video", it.video))
            }
            File(directory, "custom_model.json").writeText(
                metadata.toString()
            )
            return CustomLiteRtSpec("model.litertlm", bytes, id, displayName, inputs)
        } catch (error: Throwable) {
            directory.deleteRecursively()
            throw error
        }
    }
}
