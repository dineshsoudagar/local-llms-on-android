package com.example.local_llm

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

private const val SAVED_IMAGE_FALLBACK_NAME = "Image"
private const val CAMERA_PHOTO_NAME_PREFIX = "Camera photo"

data class SavedImageEntry(
    val imageId: String,
    val displayName: String,
    val source: OcrInput.Source,
    val createdAtMillis: Long,
    val filePath: String
)

data class SavedImageSummary(
    val imageId: String,
    val displayName: String,
    val source: OcrInput.Source,
    val createdAtMillis: Long,
    val filePath: String
)

class SavedImageStore(context: Context) {

    private val appContext = context.applicationContext
    private val imagesDir = File(appContext.filesDir, "saved_images").apply { mkdirs() }

    fun importImage(
        sourceUri: Uri,
        source: OcrInput.Source,
        sourceFilePath: String? = null
    ): SavedImageEntry {
        val sourceFile = sourceFilePath
            ?.let(::File)
            ?.takeIf { it.exists() && it.length() > 0L }
        val createdAtMillis = System.currentTimeMillis()
        val imageId = UUID.randomUUID().toString()
        val extension = resolveExtension(sourceUri, sourceFile)
        val imageFile = File(imagesDir, "$imageId.$extension")
        val entry = SavedImageEntry(
            imageId = imageId,
            displayName = resolveDisplayName(sourceUri, source, sourceFile, createdAtMillis),
            source = source,
            createdAtMillis = createdAtMillis,
            filePath = imageFile.absolutePath
        )

        try {
            if (sourceFile != null) {
                sourceFile.copyTo(imageFile, overwrite = false)
            } else {
                appContext.contentResolver.openInputStream(sourceUri)?.use { input ->
                    FileOutputStream(imageFile).use { output ->
                        input.copyTo(output)
                    }
                } ?: throw IOException("Could not read that image.")
            }

            if (!imageFile.exists() || imageFile.length() <= 0L) {
                throw IOException("Could not prepare that image.")
            }

            metadataFileFor(imageId).writeText(serializeEntry(entry).toString())
            return entry
        } catch (error: Exception) {
            runCatching { imageFile.delete() }
            runCatching { metadataFileFor(imageId).delete() }
            throw error
        }
    }

    fun load(imageId: String): SavedImageEntry? {
        val metadataFile = metadataFileFor(imageId)
        if (!metadataFile.exists()) {
            return null
        }

        return runCatching {
            deserializeEntry(JSONObject(metadataFile.readText()))
        }.getOrNull()
    }

    fun list(): List<SavedImageSummary> {
        return imagesDir.listFiles()
            ?.filter { it.extension.equals("json", ignoreCase = true) }
            ?.mapNotNull { metadataFile ->
                runCatching {
                    deserializeEntry(JSONObject(metadataFile.readText()))
                }.getOrNull()
            }
            ?.sortedByDescending { it.createdAtMillis }
            ?.map { entry ->
                SavedImageSummary(
                    imageId = entry.imageId,
                    displayName = entry.displayName,
                    source = entry.source,
                    createdAtMillis = entry.createdAtMillis,
                    filePath = entry.filePath
                )
            }
            ?: emptyList()
    }

    fun delete(imageId: String): Boolean {
        return runCatching {
            val entry = load(imageId)
            val metadataDeleted = !metadataFileFor(imageId).exists() || metadataFileFor(imageId).delete()
            val imageDeleted = entry == null || !File(entry.filePath).exists() || File(entry.filePath).delete()
            metadataDeleted && imageDeleted
        }.getOrDefault(false)
    }

    private fun metadataFileFor(imageId: String): File {
        return File(imagesDir, "$imageId.json")
    }

    private fun serializeEntry(entry: SavedImageEntry): JSONObject {
        return JSONObject().apply {
            put("imageId", entry.imageId)
            put("displayName", entry.displayName)
            put("source", entry.source.name)
            put("createdAtMillis", entry.createdAtMillis)
            put("filePath", entry.filePath)
        }
    }

    private fun deserializeEntry(json: JSONObject): SavedImageEntry {
        return SavedImageEntry(
            imageId = json.getString("imageId"),
            displayName = json.optString("displayName").ifBlank { SAVED_IMAGE_FALLBACK_NAME },
            source = OcrInput.Source.valueOf(json.optString("source", OcrInput.Source.GALLERY.name)),
            createdAtMillis = json.optLong("createdAtMillis"),
            filePath = json.getString("filePath")
        )
    }

    private fun resolveDisplayName(
        sourceUri: Uri,
        source: OcrInput.Source,
        sourceFile: File?,
        createdAtMillis: Long
    ): String {
        if (source == OcrInput.Source.CAMERA) {
            return "$CAMERA_PHOTO_NAME_PREFIX ${cameraTimestamp(createdAtMillis)}"
        }

        val queriedDisplayName = appContext.contentResolver.query(
            sourceUri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null
        )?.use { cursor ->
            val nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameColumn >= 0 && cursor.moveToFirst()) {
                cursor.getString(nameColumn)
            } else {
                null
            }
        }

        return queriedDisplayName
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?: sourceFile?.name?.takeIf { it.isNotBlank() }
            ?: SAVED_IMAGE_FALLBACK_NAME
    }

    private fun resolveExtension(sourceUri: Uri, sourceFile: File?): String {
        val fileExtension = sourceFile?.extension
            ?.trim()
            ?.lowercase(Locale.US)
            ?.takeIf { it.matches(Regex("[a-z0-9]{1,8}")) }
        if (fileExtension != null) {
            return fileExtension
        }

        val mimeType = appContext.contentResolver.getType(sourceUri)
        val mimeExtension = mimeType
            ?.let { MimeTypeMap.getSingleton().getExtensionFromMimeType(it) }
            ?.trim()
            ?.lowercase(Locale.US)
            ?.takeIf { it.matches(Regex("[a-z0-9]{1,8}")) }

        return mimeExtension ?: "jpg"
    }

    private fun cameraTimestamp(createdAtMillis: Long): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
        return formatter.format(Date(createdAtMillis))
    }
}
