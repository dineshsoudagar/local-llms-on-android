package com.example.local_llm

object AttachmentWorkPolicy {
    fun selectForRestoration(
        sessionId: String?,
        records: List<AttachmentWorkRecord>
    ): AttachmentWorkRecord? = records
        .asSequence()
        .filter { sessionId == null || it.sessionId == sessionId }
        .maxByOrNull(AttachmentWorkRecord::createdAtMillis)

    fun findDuplicate(
        records: List<AttachmentWorkRecord>,
        sessionId: String,
        attachmentId: String,
        operation: AttachmentWorkOperation
    ): AttachmentWorkRecord? = records.firstOrNull {
        it.sessionId == sessionId && it.attachmentId == attachmentId && it.operation == operation
    }

    fun failedDescriptor(
        descriptor: AttachmentDescriptor,
        operation: AttachmentWorkOperation,
        message: String,
        nowMillis: Long = System.currentTimeMillis()
    ): AttachmentDescriptor = descriptor.copy(
        status = AttachmentStatus.FAILED,
        updatedAtMillis = nowMillis,
        errorMessage = message,
        retryOperation = operation
    )
}
