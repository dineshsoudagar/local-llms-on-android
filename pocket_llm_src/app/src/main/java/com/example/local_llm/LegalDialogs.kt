package com.example.local_llm

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import io.noties.markwon.Markwon

/** First-run consent, legal document viewer and AI response reporting. */
object LegalDialogs {

    private const val MAX_EMAIL_RESPONSE_CHARS = 4000
    private const val MAX_ISSUE_RESPONSE_CHARS = 1500

    fun showConsentIfNeeded(activity: Activity, consentStore: LegalConsentStore) {
        if (consentStore.hasAcceptedCurrentVersion() || activity.isFinishing) {
            return
        }
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.legal_consent_title)
            .setMessage(R.string.legal_consent_message)
            .setCancelable(false)
            .setPositiveButton(R.string.legal_accept) { _, _ ->
                consentStore.acceptCurrentVersion()
            }
            .setNegativeButton(R.string.legal_decline) { _, _ ->
                activity.finish()
            }
            .setNeutralButton(R.string.legal_read_terms) { _, _ ->
                showDocument(activity, LegalDocument.TERMS_OF_USE) {
                    showDocument(activity, LegalDocument.PRIVACY_POLICY) {
                        showConsentIfNeeded(activity, consentStore)
                    }
                }
            }
            .show()
    }

    fun showDocument(activity: Activity, document: LegalDocument, onClosed: () -> Unit = {}) {
        val markdown = runCatching { LegalDocuments.load(activity, document) }.getOrNull()
        if (markdown == null) {
            Toast.makeText(activity, R.string.legal_document_unavailable, Toast.LENGTH_SHORT).show()
            onClosed()
            return
        }
        val padding = (20 * activity.resources.displayMetrics.density).toInt()
        val textView = TextView(activity).apply {
            setPadding(padding, padding / 2, padding, padding / 2)
            setTextIsSelectable(true)
        }
        Markwon.create(activity).setMarkdown(textView, markdown)
        val scrollView = ScrollView(activity).apply { addView(textView) }
        MaterialAlertDialogBuilder(activity)
            .setView(scrollView)
            .setPositiveButton(android.R.string.ok, null)
            .setOnDismissListener { onClosed() }
            .show()
    }

    fun showReportResponse(activity: Activity, modelName: String?, versionName: String, responseText: String) {
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.report_response_title)
            .setMessage(R.string.report_response_message)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.report_response_send) { _, _ ->
                sendReport(activity, modelName, versionName, responseText)
            }
            .show()
    }

    private fun sendReport(activity: Activity, modelName: String?, versionName: String, responseText: String) {
        val subject = activity.getString(R.string.report_response_subject)
        fun body(maxChars: Int) = activity.getString(
            R.string.report_response_body,
            modelName ?: "-",
            versionName,
            BuildConfig.DISTRIBUTION_CHANNEL,
            responseText.take(maxChars)
        )

        val email = LegalDocuments.contactEmail(activity)
        if (email != null) {
            val emailIntent = Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:")).apply {
                putExtra(Intent.EXTRA_EMAIL, arrayOf(email))
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body(MAX_EMAIL_RESPONSE_CHARS))
            }
            try {
                activity.startActivity(emailIntent)
                return
            } catch (error: ActivityNotFoundException) {
                // Fall back to the public issue tracker below.
            }
        }

        val issueUri = Uri.parse(activity.getString(R.string.report_issue_url)).buildUpon()
            .appendQueryParameter("title", subject)
            .appendQueryParameter("body", body(MAX_ISSUE_RESPONSE_CHARS))
            .build()
        try {
            activity.startActivity(Intent(Intent.ACTION_VIEW, issueUri))
        } catch (error: ActivityNotFoundException) {
            Toast.makeText(activity, R.string.report_response_no_app, Toast.LENGTH_SHORT).show()
        }
    }
}
