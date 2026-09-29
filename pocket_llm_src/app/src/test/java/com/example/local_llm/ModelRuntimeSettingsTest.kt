package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelRuntimeSettingsTest {
    @Test
    fun gemma4ModelsExposeTheir128kContextCeiling() {
        assertEquals(
            ModelRuntimeSettingsLimits.GEMMA_CONTEXT_LENGTH,
            ModelRuntimeSettingsLimits.maxFor(ModelRegistry.gemma4E2B)
        )
        assertEquals(
            ModelRuntimeSettingsLimits.GEMMA_CONTEXT_LENGTH,
            ModelRuntimeSettingsLimits.maxFor(ModelRegistry.gemma4E4B)
        )
    }

    @Test
    fun otherBackendsKeepTheirExistingContextCeilings() {
        assertEquals(
            ModelRuntimeSettingsLimits.LITERT_CONTEXT_LENGTH,
            ModelRuntimeSettingsLimits.maxFor(ModelRegistry.qwen3LiteRt)
        )
        assertEquals(
            ModelRuntimeSettingsLimits.ONNX_CONTEXT_LENGTH,
            ModelRuntimeSettingsLimits.maxFor(ModelRegistry.qwen3)
        )
    }

    @Test
    fun onlyValuesAbove8192RequireExplicitWarning() {
        assertFalse(ModelRuntimeSettingsLimits.requiresMemoryWarning(8_192))
        assertTrue(ModelRuntimeSettingsLimits.requiresMemoryWarning(8_193))
        assertTrue(ModelRuntimeSettingsLimits.requiresMemoryWarning(128_000))
    }
}
