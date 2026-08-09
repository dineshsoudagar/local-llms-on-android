package com.example.local_llm

import android.content.Context
import android.util.AtomicFile
import org.json.JSONObject
import org.json.JSONArray
import java.io.File

class AttachmentRepository(context: Context) {
    private val root = File(context.filesDir, "chat_attachments").apply { mkdirs() }

    fun attachmentDirectory(sessionId: String, attachmentId: String): File {
        requireSafeId(sessionId)
        requireSafeId(attachmentId)
        return File(File(root, sessionId), attachmentId)
    }

    fun sourceFile(descriptor: AttachmentDescriptor): File = File(descriptor.sourcePath)

    fun saveDescriptor(descriptor: AttachmentDescriptor) {
        val directory = attachmentDirectory(descriptor.sessionId, descriptor.id).apply { mkdirs() }
        atomicWrite(File(directory, MANIFEST_FILE), descriptor.toJson().toString())
    }

    fun loadDescriptor(sessionId: String, attachmentId: String): AttachmentDescriptor? {
        val file = File(attachmentDirectory(sessionId, attachmentId), MANIFEST_FILE)
        if (!file.exists()) return null
        return runCatching { descriptorFromJson(JSONObject(file.readText())) }.getOrNull()
    }

    fun listDescriptors(sessionId: String): List<AttachmentDescriptor> {
        requireSafeId(sessionId)
        return File(root, sessionId).listFiles()
            ?.filter(File::isDirectory)
            ?.mapNotNull { loadDescriptor(sessionId, it.name) }
            ?.sortedBy(AttachmentDescriptor::createdAtMillis)
            ?: emptyList()
    }

    fun saveChunks(descriptor: AttachmentDescriptor, chunks: List<AttachmentChunk>) {
        val directory = attachmentDirectory(descriptor.sessionId, descriptor.id).apply { mkdirs() }
        val content = chunks.joinToString("\n") { chunk -> chunk.toJson().toString() }
        atomicWrite(File(directory, CHUNKS_FILE), content)
    }

    fun saveBm25Index(descriptor: AttachmentDescriptor, chunks: List<AttachmentChunk>) {
        val documents = JSONArray()
        chunks.forEach { chunk ->
            val terms = Regex("[\\p{L}\\p{N}']+")
                .findAll(chunk.text.lowercase())
                .map { it.value }
                .filter { it.length > 1 }
                .groupingBy { it }
                .eachCount()
            documents.put(
                JSONObject().apply {
                    put("chunkId", chunk.id)
                    put("length", terms.values.sum())
                    put("terms", JSONObject(terms))
                }
            )
        }
        atomicWrite(
            File(attachmentDirectory(descriptor.sessionId, descriptor.id), BM25_INDEX_FILE),
            JSONObject().put("documents", documents).toString()
        )
    }

    fun loadChunks(descriptor: AttachmentDescriptor): List<AttachmentChunk> {
        val file = File(attachmentDirectory(descriptor.sessionId, descriptor.id), CHUNKS_FILE)
        if (!file.exists()) return emptyList()
        return file.useLines { lines ->
            lines.filter(String::isNotBlank)
                .mapNotNull { line -> runCatching { chunkFromJson(JSONObject(line)) }.getOrNull() }
                .toList()
        }
    }

    fun extractedTextFile(descriptor: AttachmentDescriptor): File {
        return File(attachmentDirectory(descriptor.sessionId, descriptor.id), EXTRACTED_TEXT_FILE)
    }

    fun saveExtractedText(descriptor: AttachmentDescriptor, text: String) {
        atomicWrite(extractedTextFile(descriptor), text)
    }

    fun deleteAttachment(sessionId: String, attachmentId: String): Boolean {
        val directory = attachmentDirectory(sessionId, attachmentId)
        return !directory.exists() || directory.deleteRecursively()
    }

    fun deleteSession(sessionId: String): Boolean {
        requireSafeId(sessionId)
        val directory = File(root, sessionId)
        return !directory.exists() || directory.deleteRecursively()
    }

    fun createSourceFile(sessionId: String, attachmentId: String, extension: String): File {
        val safeExtension = extension.lowercase().filter(Char::isLetterOrDigit).take(8).ifBlank { "bin" }
        val directory = attachmentDirectory(sessionId, attachmentId).apply { mkdirs() }
        return File(directory, "source.$safeExtension")
    }

    private fun atomicWrite(target: File, content: String) {
        target.parentFile?.mkdirs()
        val atomicFile = AtomicFile(target)
        val output = atomicFile.startWrite()
        try {
            output.write(content.toByteArray(Charsets.UTF_8))
            atomicFile.finishWrite(output)
        } catch (error: Throwable) {
            atomicFile.failWrite(output)
            throw error
        }
    }

    private fun requireSafeId(value: String) {
        require(SAFE_ID.matches(value)) { "Invalid attachment path identifier." }
    }

    private fun AttachmentDescriptor.toJson() = JSONObject().apply {
        put("id", id)
        put("sessionId", sessionId)
        put("displayName", displayName)
        put("mimeType", mimeType)
        put("kind", kind.name)
        put("status", status.name)
        put("processingRoute", processingRoute.name)
        put("sourcePath", sourcePath)
        put("sizeBytes", sizeBytes)
        put("createdAtMillis", createdAtMillis)
        put("updatedAtMillis", updatedAtMillis)
        pageCount?.let { put("pageCount", it) }
        durationMillis?.let { put("durationMillis", it) }
        put("extractedCharacters", extractedCharacters)
        errorMessage?.let { put("errorMessage", it) }
    }

    private fun AttachmentChunk.toJson() = JSONObject().apply {
        put("id", id)
        put("ordinal", ordinal)
        put("text", text)
        put("estimatedTokens", estimatedTokens)
        put("source", JSONObject().apply {
            source.pageNumber?.let { put("pageNumber", it) }
            source.section?.let { put("section", it) }
            source.startMillis?.let { put("startMillis", it) }
            source.endMillis?.let { put("endMillis", it) }
        })
    }

    private fun descriptorFromJson(json: JSONObject) = AttachmentDescriptor(
        id = json.getString("id"),
        sessionId = json.getString("sessionId"),
        displayName = json.getString("displayName"),
        mimeType = json.optString("mimeType", "application/octet-stream"),
        kind = AttachmentKind.valueOf(json.getString("kind")),
        status = AttachmentStatus.valueOf(json.getString("status")),
        processingRoute = AttachmentProcessingRoute.valueOf(json.getString("processingRoute")),
        sourcePath = json.getString("sourcePath"),
        sizeBytes = json.optLong("sizeBytes"),
        createdAtMillis = json.optLong("createdAtMillis"),
        updatedAtMillis = json.optLong("updatedAtMillis"),
        pageCount = json.optInt("pageCount").takeIf { json.has("pageCount") },
        durationMillis = json.optLong("durationMillis").takeIf { json.has("durationMillis") },
        extractedCharacters = json.optInt("extractedCharacters"),
        errorMessage = json.optString("errorMessage").takeIf(String::isNotBlank)
    )

    private fun chunkFromJson(json: JSONObject): AttachmentChunk {
        val source = json.optJSONObject("source") ?: JSONObject()
        return AttachmentChunk(
            id = json.getString("id"),
            ordinal = json.getInt("ordinal"),
            text = json.getString("text"),
            estimatedTokens = json.optInt("estimatedTokens", conservativeTokenEstimate(json.getString("text"))),
            source = AttachmentSourceRef(
                pageNumber = source.optInt("pageNumber").takeIf { source.has("pageNumber") },
                section = source.optString("section").takeIf(String::isNotBlank),
                startMillis = source.optLong("startMillis").takeIf { source.has("startMillis") },
                endMillis = source.optLong("endMillis").takeIf { source.has("endMillis") }
            )
        )
    }

    companion object {
        private const val MANIFEST_FILE = "manifest.json"
        private const val CHUNKS_FILE = "chunks.jsonl"
        private const val EXTRACTED_TEXT_FILE = "extracted.txt"
        private const val BM25_INDEX_FILE = "bm25-index.json"
        private val SAFE_ID = Regex("[A-Za-z0-9_-]{1,128}")
    }
}
