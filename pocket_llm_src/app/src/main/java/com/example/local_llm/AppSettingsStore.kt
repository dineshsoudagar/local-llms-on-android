package com.example.local_llm

import android.content.Context
import androidx.annotation.ColorRes
import androidx.annotation.StringRes

enum class AppAppearanceMode {
    DARK,
    LIGHT
}

enum class AppAccentOption(
    val darkStyleRes: Int,
    val lightStyleRes: Int,
    @field:StringRes val labelResId: Int,
    @field:ColorRes val swatchColorRes: Int
) {
    POCKET(R.style.Theme_local_llm, R.style.Theme_local_llm_Light, R.string.accent_pocket, R.color.send_button_fill),
    OCEAN(R.style.Theme_local_llm_Ocean, R.style.Theme_local_llm_Ocean_Light, R.string.accent_blue, R.color.ocean_send_fill),
    MIDNIGHT(R.style.Theme_local_llm_Midnight, R.style.Theme_local_llm_Midnight_Light, R.string.accent_indigo, R.color.midnight_send_fill),
    FOREST(R.style.Theme_local_llm_Forest, R.style.Theme_local_llm_Forest_Light, R.string.accent_green, R.color.forest_send_fill),
    VIOLET(R.style.Theme_local_llm_Violet, R.style.Theme_local_llm_Violet_Light, R.string.accent_violet, R.color.violet_send_fill),
    AMBER(R.style.Theme_local_llm_Amber, R.style.Theme_local_llm_Amber_Light, R.string.accent_amber, R.color.amber_send_fill),
    CORAL(R.style.Theme_local_llm_Coral, R.style.Theme_local_llm_Coral_Light, R.string.accent_coral, R.color.coral_send_fill);

    fun styleFor(appearance: AppAppearanceMode): Int {
        return when (appearance) {
            AppAppearanceMode.DARK -> darkStyleRes
            AppAppearanceMode.LIGHT -> lightStyleRes
        }
    }

    companion object {
        fun fromStoredName(name: String?): AppAccentOption {
            return when (name) {
                "TEAL" -> AMBER
                else -> entries.firstOrNull { it.name == name } ?: POCKET
            }
        }
    }
}

data class AppSettings(
    val accent: AppAccentOption = AppAccentOption.POCKET,
    val appearance: AppAppearanceMode = AppAppearanceMode.DARK,
    val chatFontSizeSp: Float = 16f
)

class AppSettingsStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "pocket_chat_settings"
        private const val KEY_ACCENT = "theme"
        private const val KEY_APPEARANCE = "appearance"
        private const val KEY_CHAT_FONT_SIZE = "chat_font_size"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): AppSettings {
        val accent = AppAccentOption.fromStoredName(
            prefs.getString(KEY_ACCENT, AppAccentOption.POCKET.name)
        )
        val appearance = runCatching {
            AppAppearanceMode.valueOf(prefs.getString(KEY_APPEARANCE, AppAppearanceMode.DARK.name)!!)
        }.getOrDefault(AppAppearanceMode.DARK)

        return AppSettings(
            accent = accent,
            appearance = appearance,
            chatFontSizeSp = prefs.getFloat(KEY_CHAT_FONT_SIZE, 16f).coerceIn(13f, 24f)
        )
    }

    fun save(settings: AppSettings) {
        prefs.edit()
            .putString(KEY_ACCENT, settings.accent.name)
            .putString(KEY_APPEARANCE, settings.appearance.name)
            .putFloat(KEY_CHAT_FONT_SIZE, settings.chatFontSizeSp)
            .apply()
    }
}
