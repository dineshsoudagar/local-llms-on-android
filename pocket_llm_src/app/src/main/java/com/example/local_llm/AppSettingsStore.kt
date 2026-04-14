package com.example.local_llm

import android.content.Context

enum class AppThemeOption(val styleRes: Int, val label: String) {
    OCEAN(R.style.Theme_local_llm, "Ocean Blue"),
    MIDNIGHT(R.style.Theme_local_llm_Midnight, "Midnight Slate"),
    FOREST(R.style.Theme_local_llm_Forest, "Forest Night"),
    VIOLET(R.style.Theme_local_llm_Violet, "Violet Night")
}

data class AppSettings(
    val theme: AppThemeOption = AppThemeOption.OCEAN,
    val chatFontSizeSp: Float = 16f
)

class AppSettingsStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "pocket_chat_settings"
        private const val KEY_THEME = "theme"
        private const val KEY_CHAT_FONT_SIZE = "chat_font_size"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): AppSettings {
        val theme = runCatching {
            AppThemeOption.valueOf(prefs.getString(KEY_THEME, AppThemeOption.OCEAN.name)!!)
        }.getOrDefault(AppThemeOption.OCEAN)

        return AppSettings(
            theme = theme,
            chatFontSizeSp = prefs.getFloat(KEY_CHAT_FONT_SIZE, 16f).coerceIn(13f, 24f)
        )
    }

    fun save(settings: AppSettings) {
        prefs.edit()
            .putString(KEY_THEME, settings.theme.name)
            .putFloat(KEY_CHAT_FONT_SIZE, settings.chatFontSizeSp)
            .apply()
    }
}
