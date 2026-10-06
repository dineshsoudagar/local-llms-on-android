package com.example.local_llm

import android.content.Context

enum class LegalDocument(val assetPath: String) {
    PRIVACY_POLICY("legal/privacy_policy.md"),
    TERMS_OF_USE("legal/terms_of_use.md")
}

object LegalDocuments {
    // Bump when the terms or privacy policy change materially; users are asked to accept again.
    const val CURRENT_VERSION = 1

    const val PLACEHOLDER_DEVELOPER_NAME = "DEVELOPER_NAME"
    const val PLACEHOLDER_POSTAL_ADDRESS = "POSTAL_ADDRESS"
    const val PLACEHOLDER_CONTACT_EMAIL = "CONTACT_EMAIL"
    const val PLACEHOLDER_EFFECTIVE_DATE = "EFFECTIVE_DATE"

    val PLACEHOLDER_KEYS = setOf(
        PLACEHOLDER_DEVELOPER_NAME,
        PLACEHOLDER_POSTAL_ADDRESS,
        PLACEHOLDER_CONTACT_EMAIL,
        PLACEHOLDER_EFFECTIVE_DATE
    )

    private val placeholderPattern = Regex("""\{\{([A-Z_]+)\}\}""")

    fun load(context: Context, document: LegalDocument): String {
        val raw = context.assets.open(document.assetPath).bufferedReader().use { it.readText() }
        return fill(raw, contactValues(context))
    }

    fun contactValues(context: Context): Map<String, String> = mapOf(
        PLACEHOLDER_DEVELOPER_NAME to context.getString(R.string.legal_developer_name),
        PLACEHOLDER_POSTAL_ADDRESS to postalAddressSuffix(context.getString(R.string.legal_postal_address)),
        PLACEHOLDER_CONTACT_EMAIL to context.getString(R.string.legal_contact_email),
        PLACEHOLDER_EFFECTIVE_DATE to context.getString(R.string.legal_effective_date)
    )

    /** The postal address is optional; the free builds can leave it empty and show only name and email. */
    fun postalAddressSuffix(address: String): String {
        val trimmed = address.trim()
        return if (trimmed.isEmpty()) "" else ", $trimmed"
    }

    fun fill(text: String, values: Map<String, String>): String {
        return placeholderPattern.replace(text) { match ->
            values[match.groupValues[1]] ?: match.value
        }
    }

    fun placeholdersIn(text: String): Set<String> {
        return placeholderPattern.findAll(text).map { it.groupValues[1] }.toSet()
    }

    /** Returns the contact email, or null while it is still a placeholder. */
    fun contactEmail(context: Context): String? {
        return context.getString(R.string.legal_contact_email)
            .trim()
            .takeIf { it.contains('@') }
    }
}

class LegalConsentStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "pocket_legal"
        private const val KEY_ACCEPTED_VERSION = "accepted_version"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun hasAcceptedCurrentVersion(): Boolean {
        return prefs.getInt(KEY_ACCEPTED_VERSION, 0) >= LegalDocuments.CURRENT_VERSION
    }

    fun acceptCurrentVersion() {
        prefs.edit().putInt(KEY_ACCEPTED_VERSION, LegalDocuments.CURRENT_VERSION).apply()
    }
}
