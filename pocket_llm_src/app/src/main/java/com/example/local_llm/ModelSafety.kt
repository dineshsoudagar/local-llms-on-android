package com.example.local_llm

import android.app.ActivityManager
import android.content.Context
import android.os.StatFs
import org.json.JSONObject
import java.io.File
import kotlin.math.max

private const val MIB = 1024L * 1024L
private const val STORAGE_FLOOR_BYTES = 256L * MIB

data class DeviceResourceSnapshot(
    val totalMemoryBytes: Long,
    val availableMemoryBytes: Long,
    val lowMemory: Boolean,
    val availableStorageBytes: Long
)

enum class ModelPreflightKind { READY, BLOCKED }

data class ModelPreflightResult(
    val kind: ModelPreflightKind,
    val message: String? = null,
    val estimatedWorkingSetBytes: Long? = null,
    val allowCpuFallback: Boolean = true
)

object ModelPreflightEvaluator {
    fun forLoad(
        descriptor: ModelDescriptor,
        modelBytes: Long,
        resources: DeviceResourceSnapshot
    ): ModelPreflightResult {
        val storageHeadroom = max(STORAGE_FLOOR_BYTES, modelBytes / 10L)
        if (resources.availableStorageBytes < storageHeadroom) {
            return ModelPreflightResult(
                ModelPreflightKind.BLOCKED,
                "Not enough free storage to load ${descriptor.displayName}. Free at least ${formatBytes(storageHeadroom)} for runtime files."
            )
        }

        val backendOverhead = when (descriptor) {
            is OnnxQwenSpec -> 768L * MIB
            is GemmaLiteRtSpec, is QwenLiteRtSpec -> 1024L * MIB
        }
        val estimatedWorkingSet = modelBytes + backendOverhead
        val risky = resources.lowMemory ||
            resources.availableMemoryBytes < estimatedWorkingSet ||
            resources.totalMemoryBytes < estimatedWorkingSet + estimatedWorkingSet / 3L
        // Memory estimates never block the load or show a warning. They only prevent a
        // risky automatic CPU fallback, which can terminate the process for large models.
        return ModelPreflightResult(
            kind = ModelPreflightKind.READY,
            estimatedWorkingSetBytes = estimatedWorkingSet,
            allowCpuFallback = !risky
        )
    }

    fun forDownload(descriptor: ModelDescriptor, remainingBytes: Long, availableStorageBytes: Long): ModelPreflightResult {
        val safetyMargin = max(STORAGE_FLOOR_BYTES, descriptor.approxDownloadBytes / 10L)
        val required = remainingBytes.coerceAtLeast(0L) + safetyMargin
        return if (availableStorageBytes < required) {
            ModelPreflightResult(
                ModelPreflightKind.BLOCKED,
                "Not enough free storage to download ${descriptor.displayName}. About ${formatBytes(required)} is required."
            )
        } else {
            ModelPreflightResult(ModelPreflightKind.READY)
        }
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1024L * MIB -> "%.1f GB".format(bytes.toDouble() / (1024L * MIB))
        else -> "%.0f MB".format(bytes.toDouble() / MIB)
    }
}

class ModelPreflightChecker(
    private val context: Context,
    private val resolver: ModelFileResolver
) {
    fun checkLoad(descriptor: ModelDescriptor, modelBytes: Long): ModelPreflightResult {
        val memoryInfo = ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memoryInfo)
        return ModelPreflightEvaluator.forLoad(
            descriptor,
            modelBytes,
            DeviceResourceSnapshot(
                totalMemoryBytes = memoryInfo.totalMem,
                availableMemoryBytes = memoryInfo.availMem,
                lowMemory = memoryInfo.lowMemory,
                availableStorageBytes = StatFs(context.filesDir.absolutePath).availableBytes
            )
        )
    }

    fun checkDownload(descriptor: ModelDescriptor): ModelPreflightResult {
        val completedBytes = descriptor.downloadFiles.sumOf { file ->
            resolver.getDownloadedFile(descriptor, file.localFileName)
                .takeIf { it.exists() }
                ?.length()
                ?: File(resolver.getDownloadedFile(descriptor, file.localFileName).absolutePath + ".download")
                    .takeIf { it.exists() }
                    ?.length()
                ?: 0L
        }
        val remaining = (descriptor.approxDownloadBytes - completedBytes).coerceAtLeast(0L)
        return ModelPreflightEvaluator.forDownload(
            descriptor,
            remaining,
            StatFs(context.filesDir.absolutePath).availableBytes
        )
    }
}

class ModelDownloadManifest(private val modelDirectory: File) {
    private val file = File(modelDirectory, "download_manifest.json")

    fun expectedLength(fileName: String): Long? = readEntries().optLong(fileName, -1L).takeIf { it > 0L }

    fun record(fileName: String, length: Long) {
        val entries = readEntries()
        entries.put(fileName, length)
        val temp = File(modelDirectory, "download_manifest.json.tmp")
        temp.writeText(entries.toString())
        if (file.exists()) file.delete()
        check(temp.renameTo(file)) { "Could not save the model download manifest." }
    }

    private fun readEntries(): JSONObject {
        if (!file.exists()) return JSONObject()
        return runCatching { JSONObject(file.readText()) }.getOrElse { JSONObject() }
    }
}

data class ModelValidationResult(val valid: Boolean, val message: String? = null, val totalBytes: Long = 0L)

class ModelFileValidator(private val resolver: ModelFileResolver) {
    fun validate(descriptor: ModelDescriptor): ModelValidationResult {
        var total = 0L
        val manifest = ModelDownloadManifest(resolver.getModelDirectory(descriptor))
        for (artifact in descriptor.downloadFiles) {
            val file = runCatching { resolver.resolveFile(descriptor, artifact.localFileName) }
                .getOrElse { return ModelValidationResult(false, it.message ?: "A required model file is missing.") }
            val problem = ModelArtifactInspector.findProblem(
                file,
                artifact,
                manifest.expectedLength(artifact.localFileName)
            )
            if (problem != null) {
                return ModelValidationResult(false, problem)
            }
            total += file.length()
        }
        return ModelValidationResult(true, totalBytes = total)
    }
}

object ModelArtifactInspector {
    fun findProblem(file: File, artifact: ModelDownloadFile, manifestLength: Long? = null): String? {
        val minimum = if (artifact.localFileName.endsWith(".json", true)) 128L else MIB
        if (!file.exists() || file.length() < minimum) {
            return "${artifact.localFileName} is missing, incomplete, or too small."
        }
        val expected = artifact.expectedBytes ?: manifestLength
        if (expected != null && file.length() != expected) {
            return "${artifact.localFileName} has the wrong size and may be corrupted."
        }
        if (artifact.localFileName.endsWith(".json", true)) {
            if (runCatching { JSONObject(file.readText()) }.isFailure) {
                return "${artifact.localFileName} is not valid JSON."
            }
        } else if (looksLikeErrorDocument(file)) {
            return "${artifact.localFileName} contains an error document instead of a model."
        }
        return null
    }

    private fun looksLikeErrorDocument(file: File): Boolean {
        val prefix = ByteArray(minOf(256L, file.length()).toInt())
        file.inputStream().use { input -> input.read(prefix) }
        val text = prefix.toString(Charsets.UTF_8).trimStart().lowercase()
        return text.startsWith("<!doctype html") || text.startsWith("<html") ||
            text.startsWith("{\"error\"") || text.startsWith("access denied")
    }
}

data class BackendInitializationPolicy(val allowCpuFallback: Boolean = true)
