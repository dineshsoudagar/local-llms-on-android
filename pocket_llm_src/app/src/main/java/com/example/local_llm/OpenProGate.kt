package com.example.local_llm

import android.content.Context
import androidx.appcompat.app.AppCompatActivity

/** The open-source builds sell nothing: every feature they contain is free. */
@Suppress("UNUSED_PARAMETER")
fun createProGate(context: Context): ProGate = OpenProGate

private object OpenProGate : ProGate {
    override fun isUnlocked(): Boolean = true

    override fun requireUnlocked(activity: AppCompatActivity, feature: ProFeature, onUnlocked: () -> Unit) = onUnlocked()

    override fun showProPage(activity: AppCompatActivity, onUnlocked: () -> Unit) = onUnlocked()

    override fun purchase(activity: AppCompatActivity, onUnlocked: () -> Unit) = onUnlocked()

    override fun restore(activity: AppCompatActivity, onUnlocked: () -> Unit) = onUnlocked()

    override fun loadPrice(onLoaded: (String?) -> Unit) = onLoaded(null)

    override fun close() = Unit
}
