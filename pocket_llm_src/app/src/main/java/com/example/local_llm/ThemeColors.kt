package com.example.local_llm

import android.content.Context
import android.util.TypedValue

/** Resolves a colour attribute of the current theme. */
fun Context.themeColor(attr: Int): Int {
    val value = TypedValue()
    theme.resolveAttribute(attr, value, true)
    return value.data
}
