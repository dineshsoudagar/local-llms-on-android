package com.example.local_llm

import android.graphics.Color
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge

/** Draws behind transparent system bars, with bar icons matching the current light or dark theme. */
fun ComponentActivity.applyEdgeToEdge() {
    val attrs = theme.obtainStyledAttributes(intArrayOf(android.R.attr.windowLightStatusBar))
    val light = attrs.getBoolean(0, false)
    attrs.recycle()
    val style = if (light) SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT)
    else SystemBarStyle.dark(Color.TRANSPARENT)
    enableEdgeToEdge(statusBarStyle = style, navigationBarStyle = style)
}
