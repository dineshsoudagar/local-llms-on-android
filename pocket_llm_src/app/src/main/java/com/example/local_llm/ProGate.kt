package com.example.local_llm

import androidx.appcompat.app.AppCompatActivity

/** Features that the Google Play build sells as a one-time "Pocket LLM Pro" unlock. */
enum class ProFeature {
    IMAGE_GENERATION,
    WEB_SEARCH,
    TRANSLATOR,
    /** The Pro row in the drawer, which offers the whole unlock rather than one feature. */
    PRO_MENU,
}

/**
 * Decides whether Pro features are available.
 *
 * Each distribution flavor provides a top-level `createProGate(context)`. The open-source builds
 * unlock everything they contain.
 */
/** Implemented by screens that can open a Pro feature, so the Welcome to Pro page can link to them. */
interface ProFeatureLauncher {
    fun openProFeature(feature: ProFeature)
}

interface ProGate {
    fun isUnlocked(): Boolean

    /** Runs [onUnlocked] right away when Pro is unlocked, otherwise offers the purchase first. */
    fun requireUnlocked(activity: AppCompatActivity, feature: ProFeature, onUnlocked: () -> Unit)

    /** Shows what Pro includes: the owned page when unlocked, otherwise the purchase offer. */
    fun showProPage(activity: AppCompatActivity, onUnlocked: () -> Unit)

    /** Starts the Google Play purchase right away, without the offer page. */
    fun purchase(activity: AppCompatActivity, onUnlocked: () -> Unit)

    fun restore(activity: AppCompatActivity, onUnlocked: () -> Unit)

    /** The local price, such as "2,99 €", or null while the store is unreachable. */
    fun loadPrice(onLoaded: (String?) -> Unit)

    fun close()
}
