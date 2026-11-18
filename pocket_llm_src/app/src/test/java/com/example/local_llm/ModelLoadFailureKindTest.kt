package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelLoadFailureKindTest {
    @Test
    fun recognisesUnknownArchitecture() {
        assertEquals(
            ModelLoadFailureKind.UNSUPPORTED_ARCHITECTURE,
            ModelLoadFailureKind.of("llama.cpp could not load /m.gguf: error loading model: unknown model architecture: 'lfm3'")
        )
    }

    @Test
    fun recognisesMemoryAndDamagedFiles() {
        assertEquals(ModelLoadFailureKind.OUT_OF_MEMORY, ModelLoadFailureKind.of("ggml: failed to allocate CPU buffer"))
        assertEquals(ModelLoadFailureKind.DAMAGED_FILE, ModelLoadFailureKind.of("gguf: invalid magic characters"))
    }

    @Test
    fun fallsBackToOther() {
        assertEquals(ModelLoadFailureKind.OTHER, ModelLoadFailureKind.of(null))
        assertEquals(ModelLoadFailureKind.OTHER, ModelLoadFailureKind.of("llama.cpp could not load /m.gguf"))
    }
}