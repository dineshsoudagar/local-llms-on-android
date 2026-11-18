package com.example.local_llm

import android.view.View
import androidx.annotation.StringRes
import androidx.appcompat.app.AppCompatActivity

interface ProUiFeature {
    fun requireProOrTrial(activity: AppCompatActivity, gate: ProGate, feature: ProFeature, action: () -> Unit)
    fun proLabel(activity: AppCompatActivity, gate: ProGate, @StringRes label: Int, feature: ProFeature): CharSequence
    fun refreshProBadges(activity: AppCompatActivity, gate: ProGate)
    fun playShine(view: View, repeat: Boolean = false)
    fun useTrial(activity: AppCompatActivity, gate: ProGate, feature: ProFeature, onUnlocked: () -> Unit = {}): Boolean
    /** True when Pro is unlocked or a try is left, without using it; otherwise shows the Pro offer. */
    fun hasTrial(activity: AppCompatActivity, gate: ProGate, feature: ProFeature, onUnlocked: () -> Unit = {}): Boolean
    /** Free tries left for [feature]; Int.MAX_VALUE when Pro is unlocked. */
    fun triesLeft(activity: AppCompatActivity, gate: ProGate, feature: ProFeature): Int
    fun magicActionsNote(activity: AppCompatActivity, gate: ProGate): ChatAdapter.MagicNote

    companion object {
        private const val IMPLEMENTATION = "com.example.local_llm.ProUiFeatureImpl"

        val instance: ProUiFeature? by lazy {
            try {
                Class.forName(IMPLEMENTATION).getField("INSTANCE").get(null) as ProUiFeature
            } catch (_: ClassNotFoundException) {
                null
            }
        }

        val isAvailable: Boolean get() = instance != null
    }
}
