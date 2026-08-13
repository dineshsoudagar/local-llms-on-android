package com.example.local_llm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GemmaNativeAudioCompatibilityTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun gpuSuccessDoesNotAuthorizeCpuAndCpuSuccessDoesNotAuthorizeGpu() {
        val store = FakeStore()
        val cache = GemmaNativeAudioCompatibilityCache(store)
        val gpu = identity(backendAttempt = "text=gpu;vision=none;audio=gpu")
        val cpu = identity(backendAttempt = "text=gpu;vision=none;audio=cpu-4")

        cache.recordSuccess(gpu)
        assertTrue(cache.isAuthorized(gpu))
        assertFalse(cache.isAuthorized(cpu))

        cache.recordSuccess(cpu)
        assertTrue(cache.isAuthorized(cpu))
        assertTrue(cache.isAuthorized(gpu))
    }

    @Test
    fun sameSizeModelReplacementAndAnyHashChangeInvalidateTheCache() {
        val original = temporaryFolder.newFile("original.litertlm").apply {
            writeBytes(byteArrayOf(1, 2, 3, 4))
        }
        val replacement = temporaryFolder.newFile("replacement.litertlm").apply {
            writeBytes(byteArrayOf(4, 3, 2, 1))
        }
        val originalHash = GemmaNativeAudioCompatibility.modelSha256(original)
        val replacementHash = GemmaNativeAudioCompatibility.modelSha256(replacement)
        assertNotEquals(originalHash, replacementHash)

        val store = FakeStore()
        val cache = GemmaNativeAudioCompatibilityCache(store)
        val originalIdentity = identity(modelSha256 = originalHash)
        val replacementIdentity = identity(modelSha256 = replacementHash)
        cache.recordSuccess(originalIdentity)

        assertTrue(cache.isAuthorized(originalIdentity))
        assertFalse(cache.isAuthorized(replacementIdentity))
    }

    @Test
    fun fingerprintAndRuntimeVersionChangesInvalidateTheCache() {
        val store = FakeStore()
        val cache = GemmaNativeAudioCompatibilityCache(store)
        val original = identity()
        cache.recordSuccess(original)

        assertFalse(cache.isAuthorized(identity(buildFingerprint = "device-build-2")))
        assertFalse(cache.isAuthorized(identity(liteRtLmVersion = "0.10.3")))
        assertFalse(cache.isAuthorized(identity(smokeTestContractVersion = "gemma4-native-audio-v2")))
    }

    @Test
    fun legacyEntriesAreClearedBeforeLookup() {
        val store = FakeStore(
            schemaVersion = 1,
            successes = mutableSetOf(identity().cacheKey(), "legacy:length-only-key")
        )
        val cache = GemmaNativeAudioCompatibilityCache(store)

        assertFalse(cache.isAuthorized(identity()))
        assertTrue(store.successes.isEmpty())
        assertTrue(store.schemaVersion == GemmaNativeAudioCompatibility.CACHE_SCHEMA_VERSION)
    }

    @Test
    fun transientFailureIsRetriedAndDoesNotEraseARecordedSuccess() {
        val store = FakeStore()
        val cache = GemmaNativeAudioCompatibilityCache(store)
        val runtime = identity()

        cache.recordFailure(runtime)
        assertFalse(cache.isAuthorized(runtime))

        cache.recordSuccess(runtime)
        cache.recordFailure(runtime)
        assertTrue(cache.isAuthorized(runtime))
    }

    private fun identity(
        modelSha256: String = "a".repeat(64),
        backendAttempt: String = "text=gpu;vision=none;audio=gpu",
        liteRtLmVersion: String = GemmaNativeAudioCompatibility.LITERT_LM_RUNTIME_VERSION,
        smokeTestContractVersion: String = GemmaNativeAudioCompatibility.SMOKE_TEST_CONTRACT_VERSION,
        buildFingerprint: String = "device-build-1"
    ) = GemmaNativeAudioRuntimeIdentity(
        modelSha256 = modelSha256,
        backendAttempt = backendAttempt,
        liteRtLmVersion = liteRtLmVersion,
        smokeTestContractVersion = smokeTestContractVersion,
        buildFingerprint = buildFingerprint
    )

    private class FakeStore(
        var schemaVersion: Int = GemmaNativeAudioCompatibility.CACHE_SCHEMA_VERSION,
        val successes: MutableSet<String> = mutableSetOf()
    ) : GemmaNativeAudioCompatibilityStore {
        override fun schemaVersion(): Int = schemaVersion

        override fun replaceWithSchema(schemaVersion: Int) {
            successes.clear()
            this.schemaVersion = schemaVersion
        }

        override fun hasSuccess(cacheKey: String): Boolean = cacheKey in successes

        override fun recordSuccess(cacheKey: String) {
            successes += cacheKey
        }
    }
}
