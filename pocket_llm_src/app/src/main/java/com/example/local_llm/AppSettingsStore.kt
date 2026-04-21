package com.example.local_llm

import android.content.Context

enum class AppAppearanceMode {
    DARK,
    LIGHT
}

enum class AppAccentOption(
    val darkStyleRes: Int,
    val lightStyleRes: Int,
    val label: String
) {
    OCEAN(R.style.Theme_local_llm, R.style.Theme_local_llm_Light, "Blue"),
    MIDNIGHT(R.style.Theme_local_llm_Midnight, R.style.Theme_local_llm_Midnight_Light, "Indigo"),
    FOREST(R.style.Theme_local_llm_Forest, R.style.Theme_local_llm_Forest_Light, "Green"),
    VIOLET(R.style.Theme_local_llm_Violet, R.style.Theme_local_llm_Violet_Light, "Violet");

    fun styleFor(appearance: AppAppearanceMode): Int {
        return when (appearance) {
            AppAppearanceMode.DARK -> darkStyleRes
            AppAppearanceMode.LIGHT -> lightStyleRes
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
        val accent = runCatching {
            AppAccentOption.valueOf(prefs.getString(KEY_ACCENT, AppAccentOption.OCEAN.name)!!)
        }.getOrDefault(AppAccentOption.OCEAN)
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
