package com.example.local_llm

import android.content.Context
import androidx.annotation.StringRes

enum class AppAppearanceMode {
    DARK,
    LIGHT
}

enum class AppAccentOption(
    val darkStyleRes: Int,
    val lightStyleRes: Int,
    @field:StringRes val labelResId: Int
) {
    OCEAN(R.style.Theme_local_llm, R.style.Theme_local_llm_Light, R.string.accent_blue),
    MIDNIGHT(R.style.Theme_local_llm_Midnight, R.style.Theme_local_llm_Midnight_Light, R.string.accent_indigo),
    FOREST(R.style.Theme_local_llm_Forest, R.style.Theme_local_llm_Forest_Light, R.string.accent_green),
    VIOLET(R.style.Theme_local_llm_Violet, R.style.Theme_local_llm_Violet_Light, R.string.accent_violet),
    AMBER(R.style.Theme_local_llm_Amber, R.style.Theme_local_llm_Amber_Light, R.string.accent_amber),
    CORAL(R.style.Theme_local_llm_Coral, R.style.Theme_local_llm_Coral_Light, R.string.accent_coral);

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
                else -> entries.firstOrNull { it.name == name } ?: OCEAN
            }
        }
    }
}

data class AppSettings(
    val accent: AppAccentOption = AppAccentOption.OCEAN,
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
            prefs.getString(KEY_ACCENT, AppAccentOption.OCEAN.name)
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
