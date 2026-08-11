package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelPreflightEvaluatorTest {
    private val model = ModelRegistry.qwen3LiteRt
    private val gib = 1024L * 1024L * 1024L

    @Test
    fun safeDeviceAllowsNormalFallback() {
        val result = ModelPreflightEvaluator.forLoad(
            model,
            model.approxDownloadBytes,
            DeviceResourceSnapshot(12 * gib, 8 * gib, false, 4 * gib)
        )

        assertEquals(ModelPreflightKind.READY, result.kind)
        assertTrue(result.allowCpuFallback)
    }

    @Test
    fun memoryRiskDoesNotBlockLoadOrDisableFallback() {
        val result = ModelPreflightEvaluator.forLoad(
            model,
            model.approxDownloadBytes,
            DeviceResourceSnapshot(2 * gib, gib / 2, true, 4 * gib)
        )

        assertEquals(ModelPreflightKind.READY, result.kind)
        assertFalse(result.allowCpuFallback)
        assertTrue(result.message == null)
    }

    @Test
    fun insufficientStorageBlocksLoadAndDownload() {
        val resources = DeviceResourceSnapshot(12 * gib, 8 * gib, false, 100L * 1024L * 1024L)
        assertEquals(
            ModelPreflightKind.BLOCKED,
            ModelPreflightEvaluator.forLoad(model, model.approxDownloadBytes, resources).kind
        )
        assertEquals(
            ModelPreflightKind.BLOCKED,
            ModelPreflightEvaluator.forDownload(model, model.approxDownloadBytes, resources.availableStorageBytes).kind
        )
    }
}
