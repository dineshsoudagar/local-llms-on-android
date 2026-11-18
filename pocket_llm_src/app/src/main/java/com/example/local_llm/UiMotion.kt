package com.example.local_llm

import android.annotation.SuppressLint
import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.animation.OvershootInterpolator
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip

/** Scale-down on touch with a small spring back. Does not consume the touch, so clicks still work. */
@SuppressLint("ClickableViewAccessibility")
fun View.springPress() {
    if (getTag(R.id.tag_spring_press) == true) return
    setTag(R.id.tag_spring_press, true)
    val overshoot = OvershootInterpolator(2.2f)
    setOnTouchListener { v, event ->
        if (!v.context.animationsEnabled()) return@setOnTouchListener false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN ->
                v.animate().scaleX(0.96f).scaleY(0.96f).setDuration(90).setInterpolator(null).start()
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                v.animate().scaleX(1f).scaleY(1f).setDuration(260).setInterpolator(overshoot).start()
        }
        false
    }
}

/** Adds [springPress] to every button, chip and card below [root]. */
fun installPressFeedback(root: View) {
    if (root is MaterialButton || root is Chip || root is MaterialCardView) {
        if (root.isClickable) root.springPress()
    }
    if (root is ViewGroup) {
        for (i in 0 until root.childCount) installPressFeedback(root.getChildAt(i))
    }
}

/** Applies press feedback to each screen once it has started. Redesign motion lives here, in one place. */
class PocketApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityStarted(activity: Activity) {
                installPressFeedback(activity.window.decorView)
            }
            override fun onActivityCreated(a: Activity, s: Bundle?) = Unit
            override fun onActivityResumed(a: Activity) = Unit
            override fun onActivityPaused(a: Activity) = Unit
            override fun onActivityStopped(a: Activity) = Unit
            override fun onActivitySaveInstanceState(a: Activity, o: Bundle) = Unit
            override fun onActivityDestroyed(a: Activity) = Unit
        })
    }
}
