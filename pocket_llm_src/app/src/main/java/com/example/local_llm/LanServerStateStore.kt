package com.example.local_llm

import android.content.Context
import android.util.Base64
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

enum class LanServerStatus {
    STOPPED,
    STARTING,
    RUNNING,
    FAILED
}

data class LanServerState(
    val status: LanServerStatus,
    val modelId: String?,
    val endpoint: String?,
    val errorMessage: String?
) {
    val isActive: Boolean
        get() = status == LanServerStatus.STARTING || status == LanServerStatus.RUNNING
}

object LanServerStateStore {
    const val MIN_PASSWORD_LENGTH = 8

    private const val PREFERENCES = "lan_server"
    private const val KEY_STATUS = "status"
    private const val KEY_MODEL_ID = "model_id"
    private const val KEY_ENDPOINT = "endpoint"
    private const val KEY_PASSWORD_SALT = "password_salt"
    private const val KEY_PASSWORD_HASH = "password_hash"
    private const val KEY_API_KEY = "api_key"
    private const val KEY_ERROR = "error"
    private const val PASSWORD_ITERATIONS = 120_000
    private const val PASSWORD_KEY_LENGTH_BITS = 256
    private const val API_KEY_PREFIX = "sk-pocket-"

    fun read(context: Context): LanServerState {
        val preferences = context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val storedEndpoint = preferences.getString(KEY_ENDPOINT, null)
        return LanServerState(
            status = runCatching {
                LanServerStatus.valueOf(
                    preferences.getString(KEY_STATUS, LanServerStatus.STOPPED.name)
                        ?: LanServerStatus.STOPPED.name
                )
            }.getOrDefault(LanServerStatus.STOPPED),
            modelId = preferences.getString(KEY_MODEL_ID, null),
            endpoint = storedEndpoint?.let { endpoint ->
                if (endpoint.endsWith("/ui")) endpoint else "${endpoint.trimEnd('/')}/ui"
            },
            errorMessage = preferences.getString(KEY_ERROR, null)
        )
    }

    fun hasPassword(context: Context): Boolean {
        val applicationContext = context.applicationContext
        val preferences = applicationContext.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        return !preferences.getString(KEY_PASSWORD_SALT, null).isNullOrBlank() &&
            !preferences.getString(KEY_PASSWORD_HASH, null).isNullOrBlank()
    }

    fun setPassword(context: Context, password: String) {
        require(password.length >= MIN_PASSWORD_LENGTH) {
            "The LAN password must be at least $MIN_PASSWORD_LENGTH characters."
        }
        val salt = ByteArray(16)
        SecureRandom().nextBytes(salt)
        val hash = hashPassword(password, salt)
        context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_PASSWORD_SALT, encode(salt))
            .putString(KEY_PASSWORD_HASH, encode(hash))
            .apply()
    }

    fun generatePassword(): String {
        val bytes = ByteArray(18)
        SecureRandom().nextBytes(bytes)
        return Base64.encodeToString(
            bytes,
            Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING
        )
    }

    fun verifyPassword(context: Context, password: String): Boolean {
        val preferences = context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
        val salt = preferences.getString(KEY_PASSWORD_SALT, null)?.let(::decode) ?: return false
        val expectedHash = preferences.getString(KEY_PASSWORD_HASH, null)?.let(::decode) ?: return false
        return MessageDigest.isEqual(expectedHash, hashPassword(password, salt))
    }

    fun getApiKey(context: Context): String? {
        return context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(KEY_API_KEY, null)
            ?.takeIf(String::isNotBlank)
    }

    fun ensureApiKey(context: Context): String {
        return getApiKey(context) ?: regenerateApiKey(context)
    }

    fun regenerateApiKey(context: Context): String {
        val bytes = ByteArray(32)
        SecureRandom().nextBytes(bytes)
        val apiKey = API_KEY_PREFIX + Base64.encodeToString(
            bytes,
            Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING
        )
        context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_API_KEY, apiKey)
            .apply()
        return apiKey
    }

    fun verifyApiKey(context: Context, apiKey: String): Boolean {
        val expected = getApiKey(context) ?: return false
        return MessageDigest.isEqual(
            expected.toByteArray(Charsets.UTF_8),
            apiKey.toByteArray(Charsets.UTF_8)
        )
    }

    fun markStarting(context: Context, modelId: String) {
        write(
            context,
            status = LanServerStatus.STARTING,
            modelId = modelId,
            endpoint = null,
            errorMessage = null
        )
    }

    fun markRunning(context: Context, modelId: String, endpoint: String) {
        write(
            context,
            status = LanServerStatus.RUNNING,
            modelId = modelId,
            endpoint = endpoint,
            errorMessage = null
        )
    }

    fun markFailed(context: Context, modelId: String?, message: String) {
        write(
            context,
            status = LanServerStatus.FAILED,
            modelId = modelId,
            endpoint = null,
            errorMessage = message
        )
    }

    fun markStopped(context: Context) {
        val previous = read(context)
        write(
            context,
            status = LanServerStatus.STOPPED,
            modelId = previous.modelId,
            endpoint = null,
            errorMessage = null
        )
    }

    private fun write(
        context: Context,
        status: LanServerStatus,
        modelId: String?,
        endpoint: String?,
        errorMessage: String?
    ) {
        context.applicationContext
            .getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_STATUS, status.name)
            .putString(KEY_MODEL_ID, modelId)
            .putString(KEY_ENDPOINT, endpoint)
            .putString(KEY_ERROR, errorMessage)
            .apply()
    }

    private fun hashPassword(password: String, salt: ByteArray): ByteArray {
        val keySpec = PBEKeySpec(
            password.toCharArray(),
            salt,
            PASSWORD_ITERATIONS,
            PASSWORD_KEY_LENGTH_BITS
        )
        return try {
            val factory = runCatching {
                SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
            }.getOrElse {
                SecretKeyFactory.getInstance("PBKDF2WithHmacSHA1")
            }
            factory.generateSecret(keySpec).encoded
        } finally {
            keySpec.clearPassword()
        }
    }

    private fun encode(value: ByteArray): String {
        return Base64.encodeToString(value, Base64.NO_WRAP)
    }

    private fun decode(value: String): ByteArray? {
        return runCatching { Base64.decode(value, Base64.NO_WRAP) }.getOrNull()
    }
}
