package com.example.local_llm

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LegalDocumentsTest {
    @Test
    fun fillReplacesKnownPlaceholdersAndKeepsUnknownOnes() {
        val filled = LegalDocuments.fill(
            "Contact {{CONTACT_EMAIL}} or {{UNKNOWN}}.",
            mapOf(LegalDocuments.PLACEHOLDER_CONTACT_EMAIL to "dev@example.org")
        )

        assertEquals("Contact dev@example.org or {{UNKNOWN}}.", filled)
    }

    @Test
    fun emptyPostalAddressLeavesNoDanglingComma() {
        assertEquals("", LegalDocuments.postalAddressSuffix("  "))
        assertEquals(", Main St 1, Dresden", LegalDocuments.postalAddressSuffix("Main St 1, Dresden"))
    }

    @Test
    fun bundledDocumentsOnlyUseSupportedPlaceholders() {
        LegalDocument.entries.forEach { document ->
            val text = mainSourceFile("assets/${document.assetPath}").readText()
            val unsupported = LegalDocuments.placeholdersIn(text) - LegalDocuments.PLACEHOLDER_KEYS
            assertTrue("${document.assetPath} uses $unsupported", unsupported.isEmpty())
        }
    }

    @Test
    fun contactResourcesDefineEveryPlaceholder() {
        val resources = mainSourceFile("res/values/legal_contact.xml").readText()
        listOf(
            "legal_developer_name",
            "legal_postal_address",
            "legal_contact_email",
            "legal_effective_date"
        ).forEach { name ->
            assertTrue("missing $name", resources.contains("name=\"$name\""))
        }
    }

    private fun mainSourceFile(relativePath: String): File {
        val moduleRelative = File("src/main", relativePath)
        return if (moduleRelative.exists()) moduleRelative else File("app/src/main", relativePath)
    }
}
