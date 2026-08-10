package com.example.local_llm

import java.util.UUID

enum class AttachmentKind {
    TEXT,
    PDF,
    AUDIO
}

enum class AttachmentStatus {
    IMPORTING,
    READY,
    PROCESSING,
    FAILED
}

enum class AttachmentProcessingRoute {
    TEXT_EXTRACTED,
    PDF_TEXT,
    PDF_OCR,
    GEMMA_NATIVE_AUDIO,
    SHERPA_WHISPER
}

enum class AttachmentWorkOperation {
    IMPORT,
    WHISPER_TRANSCRIPTION
}

data class AttachmentWorkRecord(
    val workId: String,
    val sessionId: String,
    val attachmentId: String,
    val operation: AttachmentWorkOperation,
    val requestedKind: AttachmentKind,
    val useGemmaNativeAudio: Boolean = false,
    val createdAtMillis: Long = System.currentTimeMillis()
)

data class AttachmentSourceRef(
    val pageNumber: Int? = null,
    val section: String? = null,
    val startMillis: Long? = null,
    val endMillis: Long? = null
) {
    fun label(): String = when {
        pageNumber != null -> "p. $pageNumber"
        startMillis != null -> "${formatTimestamp(startMillis)}-${formatTimestamp(endMillis ?: startMillis)}"
        !section.isNullOrBlank() -> section
        else -> "source"
    }

    private fun formatTimestamp(valueMillis: Long): String {
        val totalSeconds = valueMillis.coerceAtLeast(0L) / 1_000L
        val hours = totalSeconds / 3_600L
        val minutes = (totalSeconds % 3_600L) / 60L
        val seconds = totalSeconds % 60L
        return if (hours > 0L) {
            "%02d:%02d:%02d".format(hours, minutes, seconds)
        } else {
            "%02d:%02d".format(minutes, seconds)
        }
    }
}

data class AttachmentChunk(
    val id: String = UUID.randomUUID().toString(),
    val ordinal: Int,
    val text: String,
    val source: AttachmentSourceRef,
    val estimatedTokens: Int
)

data class AttachmentDescriptor(
    val id: String = UUID.randomUUID().toString(),
    val sessionId: String,
    val displayName: String,
    val mimeType: String,
    val kind: AttachmentKind,
    val status: AttachmentStatus,
    val processingRoute: AttachmentProcessingRoute,
    val sourcePath: String,
    val sizeBytes: Long,
    val createdAtMillis: Long = System.currentTimeMillis(),
    val updatedAtMillis: Long = createdAtMillis,
    val pageCount: Int? = null,
    val durationMillis: Long? = null,
    val extractedCharacters: Int = 0,
    val errorMessage: String? = null,
    val retryOperation: AttachmentWorkOperation? = null,
    val useGemmaNativeAudio: Boolean = false
)

data class NativeAudioInput(
    val filePath: String,
    val startMillis: Long = 0L,
    val endMillis: Long? = null
)

data class AttachmentContext(
    val attachmentId: String,
    val displayName: String,
    val route: AttachmentProcessingRoute,
    val contextText: String = "",
    val sourceRefs: List<AttachmentSourceRef> = emptyList(),
    val nativeAudioInputs: List<NativeAudioInput> = emptyList(),
    val task: AttachmentTask = AttachmentTask.QUESTION,
    val promptBatches: List<String> = emptyList(),
    val requiresFinalSynthesis: Boolean = false,
    val temporaryFiles: List<String> = emptyList()
)

enum class AttachmentTask {
    QUESTION,
    SUMMARY,
    TRANSFORMATION
}

object AttachmentLimits {
    const val MAX_TEXT_BYTES = 16L * 1024L * 1024L
    const val MAX_PDF_BYTES = 64L * 1024L * 1024L
    const val MAX_AUDIO_BYTES = 512L * 1024L * 1024L
    const val MAX_PDF_PAGES = 500
    const val MAX_AUDIO_DURATION_MILLIS = 2L * 60L * 60L * 1_000L
    const val MAX_EXTRACTED_TOKENS = 500_000
    const val MAX_SUMMARY_SOURCE_TOKENS = 100_000
    const val MAX_TRANSFORMATION_SOURCE_TOKENS = 25_000
    const val GEMMA_MAX_AUDIO_INPUT_MILLIS = 30_000L
    const val AUDIO_SEGMENT_MILLIS = 28_000L
    const val AUDIO_SEGMENT_OVERLAP_MILLIS = 2_000L
}
