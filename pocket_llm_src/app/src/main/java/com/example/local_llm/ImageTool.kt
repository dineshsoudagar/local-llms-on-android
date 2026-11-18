package com.example.local_llm

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.annotation.StringRes
import java.io.File
import kotlin.math.min

/** The tabs of the Image tools page. All of them are Pro and share the same free tries. */
enum class ImageTool(@StringRes val actionLabel: Int, val usesModel: Boolean) {
    CREATE(R.string.image_gen_generate, true),
    EDIT(R.string.image_tool_edit_action, true),
    REMOVE_BACKGROUND(R.string.image_tool_remove_background, false),
    REMOVE_OBJECT(R.string.image_tool_remove_object, false),
    ENHANCE(R.string.image_tool_enhance, false),
    UPSCALE(R.string.image_tool_upscale, false);

    companion object {
        fun fromName(name: String?): ImageTool =
            entries.firstOrNull { it.name == name } ?: CREATE
    }
}

