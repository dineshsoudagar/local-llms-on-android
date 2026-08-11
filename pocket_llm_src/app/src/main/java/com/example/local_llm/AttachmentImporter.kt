package com.example.local_llm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.util.UUID
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class AttachmentImporter(
    context: Context,
    private val repository: AttachmentRepository = AttachmentRepository(context),
    private val audioNormalizer: AudioNormalizer = AudioNormalizer()
) {
    private val appContext = context.applicationContext

    suspend fun import(
        uri: Uri,
        sessionId: String,
        requestedKind: AttachmentKind,
        useGemmaNativeAudio: Boolean,
        attachmentId: String = UUID.randomUUID().toString()
    ): AttachmentDescriptor = withContext(Dispatchers.IO) {
        val metadata = queryMetadata(uri)
        val kind = validateKind(metadata, requestedKind)
        val limit = when (kind) {
            AttachmentKind.TEXT -> AttachmentLimits.MAX_TEXT_BYTES
            AttachmentKind.PDF -> AttachmentLimits.MAX_PDF_BYTES
            AttachmentKind.AUDIO -> AttachmentLimits.MAX_AUDIO_BYTES
        }
        require(metadata.sizeBytes == null || metadata.sizeBytes <= limit) {
            sizeLimitMessage(kind)
        }

        val extension = metadata.displayName.substringAfterLast('.', missingDelimiterValue = defaultExtension(kind))
        val source = repository.createSourceFile(sessionId, attachmentId, extension)
        val copiedBytes = copyWithLimit(uri, source, limit, kind)
        var descriptor = AttachmentDescriptor(
            id = attachmentId,
            sessionId = sessionId,
            displayName = metadata.displayName,
            mimeType = metadata.mimeType,
            kind = kind,
            status = AttachmentStatus.IMPORTING,
            processingRoute = initialRoute(kind, useGemmaNativeAudio),
            sourcePath = source.absolutePath,
            sizeBytes = copiedBytes,
            retryOperation = AttachmentWorkOperation.IMPORT,
            useGemmaNativeAudio = useGemmaNativeAudio
        )
        repository.saveDescriptor(descriptor)

        try {
            descriptor = when (kind) {
                AttachmentKind.TEXT -> prepareText(descriptor)
                AttachmentKind.PDF -> preparePdf(descriptor)
                AttachmentKind.AUDIO -> prepareAudio(descriptor, useGemmaNativeAudio)
            }
            descriptor = descriptor.copy(retryOperation = null, errorMessage = null)
            repository.saveDescriptor(descriptor)
            descriptor
        } catch (error: Throwable) {
            descriptor = descriptor.copy(
                status = AttachmentStatus.FAILED,
                updatedAtMillis = System.currentTimeMillis(),
                errorMessage = actionableMessage(kind, error)
            )
            repository.saveDescriptor(descriptor)
            throw AttachmentImportException(descriptor, descriptor.errorMessage.orEmpty(), error)
        }
    }

    suspend fun retry(descriptor: AttachmentDescriptor): AttachmentDescriptor = withContext(Dispatchers.IO) {
        require(repository.sourceFile(descriptor).isFile) { "The app-owned attachment source is no longer available." }
        var retrying = descriptor.copy(
            status = AttachmentStatus.IMPORTING,
            updatedAtMillis = System.currentTimeMillis(),
            errorMessage = null,
            retryOperation = AttachmentWorkOperation.IMPORT
        )
        repository.saveDescriptor(retrying)
        try {
            retrying = when (retrying.kind) {
                AttachmentKind.TEXT -> prepareText(retrying)
                AttachmentKind.PDF -> preparePdf(retrying)
                AttachmentKind.AUDIO -> prepareAudio(retrying, retrying.useGemmaNativeAudio)
            }.copy(retryOperation = null, errorMessage = null)
            repository.saveDescriptor(retrying)
            retrying
        } catch (error: Throwable) {
            retrying = retrying.copy(
                status = AttachmentStatus.FAILED,
                updatedAtMillis = System.currentTimeMillis(),
                errorMessage = actionableMessage(retrying.kind, error)
            )
            repository.saveDescriptor(retrying)
            throw AttachmentImportException(retrying, retrying.errorMessage.orEmpty(), error)
        }
    }

    private fun prepareText(descriptor: AttachmentDescriptor): AttachmentDescriptor {
        val bytes = repository.sourceFile(descriptor).readBytes()
        val text = AttachmentTextNormalizer.normalize(decodeText(bytes))
        require(text.isNotBlank()) { "The document contains no readable text." }
        val sections = splitTextSections(text)
        persistText(descriptor, sections)
        return descriptor.copy(
            status = AttachmentStatus.READY,
            updatedAtMillis = System.currentTimeMillis(),
            extractedCharacters = text.length
        )
    }

    private suspend fun preparePdf(descriptor: AttachmentDescriptor): AttachmentDescriptor {
        PDFBoxResourceLoader.init(appContext)
        val source = repository.sourceFile(descriptor)
        val sections = mutableListOf<AttachmentTextSection>()
        var usedOcr = false
        var pageCount = 0
        PDDocument.load(source).use { document ->
            require(!document.isEncrypted) { "Encrypted PDFs are not supported. Remove the password and try again." }
            pageCount = document.numberOfPages
            require(pageCount in 1..AttachmentLimits.MAX_PDF_PAGES) {
                "PDFs must contain between 1 and ${AttachmentLimits.MAX_PDF_PAGES} pages."
            }
            val renderer = PdfRenderer(android.os.ParcelFileDescriptor.open(source, android.os.ParcelFileDescriptor.MODE_READ_ONLY))
            renderer.use {
                for (pageIndex in 0 until pageCount) {
                    val embedded = AttachmentTextNormalizer.normalize(
                        PDFTextStripper().apply {
                            startPage = pageIndex + 1
                            endPage = pageIndex + 1
                            sortByPosition = true
                        }.getText(document)
                    )
                    val pageText = if (embedded.length >= MIN_EMBEDDED_PAGE_CHARS) {
                        embedded
                    } else {
                        val ocrText = recognizePdfPage(renderer, pageIndex)
                        if (ocrText.isNotBlank()) {
                            usedOcr = true
                        }
                        PdfPageTextMerger.merge(embedded, ocrText)
                    }
                    sections += AttachmentTextSection(
                        text = pageText,
                        source = AttachmentSourceRef(pageNumber = pageIndex + 1)
                    )
                }
            }
        }
        val fullText = renderAttachmentSections(sections)
        require(fullText.isNotBlank()) { "The PDF contains no readable text." }
        persistText(descriptor, sections)
        return descriptor.copy(
            status = AttachmentStatus.READY,
            processingRoute = if (usedOcr) AttachmentProcessingRoute.PDF_OCR else AttachmentProcessingRoute.PDF_TEXT,
            updatedAtMillis = System.currentTimeMillis(),
            pageCount = pageCount,
            extractedCharacters = fullText.length
        )
    }

    private suspend fun prepareAudio(
        descriptor: AttachmentDescriptor,
        useGemmaNativeAudio: Boolean
    ): AttachmentDescriptor {
        val normalizedFile = File(
            repository.attachmentDirectory(descriptor.sessionId, descriptor.id),
            "normalized.wav"
        )
        val original = repository.sourceFile(descriptor)
        val normalized = audioNormalizer.normalize(original, normalizedFile)
        if (original.absolutePath != normalized.file.absolutePath) original.delete()
        return descriptor.copy(
            status = AttachmentStatus.READY,
            processingRoute = if (useGemmaNativeAudio) {
                AttachmentProcessingRoute.GEMMA_NATIVE_AUDIO
            } else {
                AttachmentProcessingRoute.SHERPA_WHISPER
            },
            sourcePath = normalized.file.absolutePath,
            sizeBytes = normalized.file.length(),
            durationMillis = normalized.durationMillis,
            updatedAtMillis = System.currentTimeMillis()
        )
    }

    private fun persistText(descriptor: AttachmentDescriptor, sections: List<AttachmentTextSection>) {
        val chunks = AttachmentChunker().chunk(sections, targetTokens = DEFAULT_CHUNK_TOKENS)
        val extractedTokens = chunks.sumOf(AttachmentChunk::estimatedTokens)
        require(extractedTokens <= AttachmentLimits.MAX_EXTRACTED_TOKENS) {
            "The extracted document exceeds the 500,000-token limit."
        }
        repository.saveExtractedText(
            descriptor,
            renderAttachmentSections(sections)
        )
        repository.saveChunks(descriptor, chunks)
        repository.saveBm25Index(descriptor, chunks)
    }

    private suspend fun recognizePdfPage(renderer: PdfRenderer, pageIndex: Int): String {
        val page = renderer.openPage(pageIndex)
        val scale = (MAX_OCR_RENDER_DIMENSION.toFloat() / maxOf(page.width, page.height)).coerceAtMost(2f)
        val bitmap = Bitmap.createBitmap(
            (page.width * scale).toInt().coerceAtLeast(1),
            (page.height * scale).toInt().coerceAtLeast(1),
            Bitmap.Config.ARGB_8888
        )
        try {
            page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            return try {
                val result = recognizer.process(InputImage.fromBitmap(bitmap, 0)).await()
                AttachmentTextNormalizer.normalize(result.text)
            } finally {
                recognizer.close()
            }
        } finally {
            bitmap.recycle()
            page.close()
        }
    }

    private fun splitTextSections(text: String): List<AttachmentTextSection> {
        val sections = text.split(Regex("""\n(?=\s*(?:#{1,6}\s+|[A-Z][^\n]{0,80}\n[-=]{3,}))"""))
        return sections.filter(String::isNotBlank).mapIndexed { index, section ->
            AttachmentTextSection(section, AttachmentSourceRef(section = "section ${index + 1}"))
        }
    }

    private fun decodeText(bytes: ByteArray): String {
        if (bytes.isEmpty()) return ""
        return when {
            bytes.size >= 2 && bytes[0] == 0xff.toByte() && bytes[1] == 0xfe.toByte() ->
                String(bytes, 2, bytes.size - 2, Charsets.UTF_16LE)
            bytes.size >= 2 && bytes[0] == 0xfe.toByte() && bytes[1] == 0xff.toByte() ->
                String(bytes, 2, bytes.size - 2, Charsets.UTF_16BE)
            else -> try {
                StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(bytes.dropUtf8Bom())
                    .toString()
            } catch (_: CharacterCodingException) {
                throw IllegalArgumentException("Text documents must use UTF-8 or UTF-16 encoding.")
            }
        }
    }

    private fun ByteArray.dropUtf8Bom(): ByteBuffer {
        val offset = if (
            size >= 3 && this[0] == 0xef.toByte() && this[1] == 0xbb.toByte() && this[2] == 0xbf.toByte()
        ) 3 else 0
        return ByteBuffer.wrap(this, offset, size - offset)
    }

    private fun copyWithLimit(uri: Uri, destination: File, limit: Long, kind: AttachmentKind): Long {
        destination.parentFile?.mkdirs()
        val input = appContext.contentResolver.openInputStream(uri)
            ?: throw IOException("Android could not open the selected file.")
        var total = 0L
        try {
            input.use { source ->
                destination.outputStream().buffered().use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = source.read(buffer)
                        if (read < 0) break
                        total += read
                        require(total <= limit) { sizeLimitMessage(kind) }
                        output.write(buffer, 0, read)
                    }
                }
            }
        } catch (error: Throwable) {
            destination.delete()
            throw error
        }
        return total
    }

    private fun queryMetadata(uri: Uri): SourceMetadata {
        if (uri.scheme == "file") {
            val file = File(requireNotNull(uri.path))
            val extension = file.extension.lowercase()
            return SourceMetadata(
                displayName = file.name,
                mimeType = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension)
                    ?: "application/octet-stream",
                sizeBytes = file.length()
            )
        }
        var name: String? = null
        var size: Long? = null
        appContext.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
            null,
            null,
            null
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                name = cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME))
                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) size = cursor.getLong(sizeIndex)
            }
        }
        return SourceMetadata(
            displayName = name?.takeIf(String::isNotBlank) ?: "attachment",
            mimeType = appContext.contentResolver.getType(uri) ?: "application/octet-stream",
            sizeBytes = size
        )
    }

    private fun validateKind(metadata: SourceMetadata, requested: AttachmentKind?): AttachmentKind {
        val extension = metadata.displayName.substringAfterLast('.', "").lowercase()
        val inferred = when {
            metadata.mimeType == "application/pdf" || extension == "pdf" -> AttachmentKind.PDF
            metadata.mimeType.startsWith("audio/") || extension in AUDIO_EXTENSIONS -> AttachmentKind.AUDIO
            metadata.mimeType.startsWith("text/") || extension in TEXT_EXTENSIONS -> AttachmentKind.TEXT
            else -> throw IllegalArgumentException(
                "Unsupported attachment type. Choose a PDF, safe text document, or Android-decodable audio file."
            )
        }
        require(requested == null || requested == inferred || requested == AttachmentKind.TEXT && inferred == AttachmentKind.PDF) {
            "The selected file does not match the requested attachment type."
        }
        return inferred
    }

    private fun initialRoute(kind: AttachmentKind, gemma: Boolean) = when (kind) {
        AttachmentKind.TEXT -> AttachmentProcessingRoute.TEXT_EXTRACTED
        AttachmentKind.PDF -> AttachmentProcessingRoute.PDF_TEXT
        AttachmentKind.AUDIO -> if (gemma) AttachmentProcessingRoute.GEMMA_NATIVE_AUDIO else AttachmentProcessingRoute.SHERPA_WHISPER
    }

    private fun defaultExtension(kind: AttachmentKind) = when (kind) {
        AttachmentKind.TEXT -> "txt"
        AttachmentKind.PDF -> "pdf"
        AttachmentKind.AUDIO -> "audio"
    }

    private fun sizeLimitMessage(kind: AttachmentKind) = when (kind) {
        AttachmentKind.TEXT -> "Text documents must be 16 MiB or smaller."
        AttachmentKind.PDF -> "PDFs must be 64 MiB or smaller."
        AttachmentKind.AUDIO -> "Audio files must be 512 MiB or smaller."
    }

    private fun actionableMessage(kind: AttachmentKind, error: Throwable): String {
        val detail = error.message?.takeIf(String::isNotBlank)
        return detail ?: when (kind) {
            AttachmentKind.TEXT -> "The text document could not be decoded. Save it as UTF-8 and try again."
            AttachmentKind.PDF -> "The PDF is malformed or unreadable. Repair or re-export it and try again."
            AttachmentKind.AUDIO -> "Android could not decode this audio codec. Convert it to WAV, MP3, M4A, or OGG and try again."
        }
    }

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { continuation ->
        addOnSuccessListener { value -> if (continuation.isActive) continuation.resume(value) }
        addOnFailureListener { error -> if (continuation.isActive) continuation.resumeWithException(error) }
        addOnCanceledListener { continuation.cancel() }
    }

    private data class SourceMetadata(
        val displayName: String,
        val mimeType: String,
        val sizeBytes: Long?
    )

    companion object {
        private const val DEFAULT_CHUNK_TOKENS = 384
        private const val MIN_EMBEDDED_PAGE_CHARS = 24
        private const val MAX_OCR_RENDER_DIMENSION = 2_200
        private val TEXT_EXTENSIONS = setOf(
            "txt", "md", "markdown", "csv", "tsv", "json", "xml", "yaml", "yml", "html", "htm",
            "kt", "kts", "java", "py", "js", "ts", "css", "c", "cpp", "h", "hpp", "rs", "go", "sql"
        )
        private val AUDIO_EXTENSIONS = setOf("wav", "mp3", "m4a", "aac", "ogg", "opus", "flac", "3gp", "mp4")
    }
}

class AttachmentImportException(
    val descriptor: AttachmentDescriptor,
    message: String,
    cause: Throwable
) : IOException(message, cause)

internal object PdfPageTextMerger {
    private const val EMPTY_PAGE_TEXT = "[Empty page]"

    fun merge(embeddedText: String, ocrText: String): String {
        val embedded = AttachmentTextNormalizer.normalize(embeddedText)
        val ocr = AttachmentTextNormalizer.normalize(ocrText)
        if (embedded.isBlank() && ocr.isBlank()) return EMPTY_PAGE_TEXT
        if (embedded.isBlank()) return ocr
        if (ocr.isBlank()) return embedded
        if (canonical(embedded) == canonical(ocr)) return embedded

        val embeddedLines = embedded.lineSequence().filter(String::isNotBlank).toList()
        val seenKeys = embeddedLines.map(::canonical).toMutableSet()
        val uniqueOcrLines = ocr.lineSequence()
            .filter(String::isNotBlank)
            .filter { seenKeys.add(canonical(it)) }
            .toList()
        if (uniqueOcrLines.isEmpty()) return embedded
        return "$embedded\n\n${uniqueOcrLines.joinToString("\n")}".trim()
    }

    private fun canonical(text: String): String {
        return text.trim().replace(Regex("\\s+"), " ").lowercase()
    }
}

internal fun renderAttachmentSections(sections: List<AttachmentTextSection>): String {
    return sections.joinToString("\n\n") { section ->
        "[${section.source.label()}]\n${section.text}"
    }
}
