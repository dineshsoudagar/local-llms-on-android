package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PdfPageTextMergerTest {
    @Test
    fun shortEmbeddedHeadingAndScannedBodyAreBothPreserved() {
        val merged = PdfPageTextMerger.merge(
            embeddedText = "Annual Report",
            ocrText = "Annual Report\nRevenue increased this year."
        )

        assertEquals("Annual Report\n\nRevenue increased this year.", merged)
    }

    @Test
    fun exactOcrDuplicateIsNotRepeated() {
        assertEquals(
            "Page label",
            PdfPageTextMerger.merge("Page label", "Page label")
        )
    }

    @Test
    fun partialLineOverlapKeepsUniqueEmbeddedAndOcrContentInOrder() {
        val merged = PdfPageTextMerger.merge(
            "Quarterly report\nApproved",
            "Approved\nRevenue increased"
        )

        assertEquals("Quarterly report\nApproved\n\nRevenue increased", merged)
    }

    @Test
    fun embeddedOnlyOcrOnlyAndGenuinelyEmptyPagesAreStable() {
        assertEquals("Embedded text", PdfPageTextMerger.merge("Embedded text", ""))
        assertEquals("OCR body", PdfPageTextMerger.merge("", "OCR body"))
        assertEquals("[Empty page]", PdfPageTextMerger.merge("", ""))
    }

    @Test
    fun multiPageSourceReferencesRemainAttachedToTheirPages() {
        val rendered = renderAttachmentSections(
            listOf(
                AttachmentTextSection(
                    PdfPageTextMerger.merge("Heading", "Heading\nBody one"),
                    AttachmentSourceRef(pageNumber = 1)
                ),
                AttachmentTextSection(
                    PdfPageTextMerger.merge("", "Body two"),
                    AttachmentSourceRef(pageNumber = 2)
                ),
                AttachmentTextSection(
                    PdfPageTextMerger.merge("", ""),
                    AttachmentSourceRef(pageNumber = 3)
                )
            )
        )

        assertTrue(rendered.indexOf("[p. 1]") < rendered.indexOf("[p. 2]"))
        assertTrue(rendered.indexOf("[p. 2]") < rendered.indexOf("[p. 3]"))
        assertTrue(rendered.contains("[p. 1]\nHeading\n\nBody one"))
        assertTrue(rendered.contains("[p. 3]\n[Empty page]"))
        assertFalse(rendered.contains("Heading\nHeading"))
    }
}
