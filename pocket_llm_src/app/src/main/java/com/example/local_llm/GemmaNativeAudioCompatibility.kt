package com.example.local_llm

import android.content.SharedPreferences
import androidx.core.content.edit
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

internal data class GemmaNativeAudioRuntimeIdentity(
    val modelSha256: String,
    val backendAttempt: String,
    val liteRtLmVersion: String,
    val smokeTestContractVersion: String,
    val buildFingerprint: String
) {
    fun cacheKey(): String {
        val canonical = listOf(
            modelSha256.lowercase(),
            backendAttempt,
            liteRtLmVersion,
            smokeTestContractVersion,
            buildFingerprint
        ).joinToString(separator = "|") { value -> "${value.length}:$value" }
        return GemmaNativeAudioCompatibility.SUCCESS_KEY_PREFIX + sha256(canonical.toByteArray())
    }
}

internal interface GemmaNativeAudioCompatibilityStore {
    fun schemaVersion(): Int
    fun replaceWithSchema(schemaVersion: Int)
    fun hasSuccess(cacheKey: String): Boolean
    fun recordSuccess(cacheKey: String)
}

internal class SharedPreferencesGemmaNativeAudioCompatibilityStore(
    private val preferences: SharedPreferences
) : GemmaNativeAudioCompatibilityStore {
    override fun schemaVersion(): Int = preferences.getInt(
        GemmaNativeAudioCompatibility.SCHEMA_VERSION_KEY,
        0
    )

    override fun replaceWithSchema(schemaVersion: Int) {
        preferences.edit(commit = true) {
            clear()
            putInt(GemmaNativeAudioCompatibility.SCHEMA_VERSION_KEY, schemaVersion)
        }
    }

    override fun hasSuccess(cacheKey: String): Boolean = preferences.getBoolean(cacheKey, false)

    override fun recordSuccess(cacheKey: String) {
        preferences.edit(commit = true) {
            putBoolean(cacheKey, true)
        }
    }
}

internal class GemmaNativeAudioCompatibilityCache(
    private val store: GemmaNativeAudioCompatibilityStore
) {
    fun isAuthorized(identity: GemmaNativeAudioRuntimeIdentity): Boolean {
        migrateLegacyEntries()
        return store.hasSuccess(identity.cacheKey())
    }

    fun recordSuccess(identity: GemmaNativeAudioRuntimeIdentity) {
        migrateLegacyEntries()
        store.recordSuccess(identity.cacheKey())
    }

    fun recordFailure(identity: GemmaNativeAudioRuntimeIdentity) {
        migrateLegacyEntries()
        // Failures are deliberately not persisted. A process kill, driver reset, or temporary
        // resource shortage must not permanently disable native audio for this runtime.
    }

    private fun migrateLegacyEntries() {
        if (store.schemaVersion() != GemmaNativeAudioCompatibility.CACHE_SCHEMA_VERSION) {
            store.replaceWithSchema(GemmaNativeAudioCompatibility.CACHE_SCHEMA_VERSION)
        }
    }
}

internal object GemmaNativeAudioCompatibility {
    const val CACHE_SCHEMA_VERSION = 2
    val LITERT_LM_RUNTIME_VERSION: String
        get() = BuildConfig.LITERT_LM_RUNTIME_VERSION
    const val SMOKE_TEST_CONTRACT_VERSION = "gemma4-native-audio-v3"
    const val SCHEMA_VERSION_KEY = "cache_schema_version"
    const val SUCCESS_KEY_PREFIX = "success_v2:"

    fun modelSha256(file: File): String = FileInputStream(file).use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
        }
        digest.digest().toHex()
    }
}

private fun sha256(bytes: ByteArray): String = MessageDigest
    .getInstance("SHA-256")
    .digest(bytes)
    .toHex()

private fun ByteArray.toHex(): String = joinToString(separator = "") { byte -> "%02x".format(byte) }
