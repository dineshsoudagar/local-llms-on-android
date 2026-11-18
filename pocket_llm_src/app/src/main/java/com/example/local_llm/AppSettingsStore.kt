package com.example.local_llm

import android.content.Context
import android.content.res.Configuration
import androidx.annotation.ColorRes
import androidx.annotation.StringRes

enum class AppAppearanceMode {
    SYSTEM,
    DARK,
    LIGHT
}

enum class AppAccentOption(
    val darkStyleRes: Int,
    val lightStyleRes: Int,
    @field:StringRes val labelResId: Int,
    @field:ColorRes val swatchColorRes: Int
) {
    VIOLET(R.style.Theme_local_llm_Violet, R.style.Theme_local_llm_Violet_Light, R.string.accent_violet, R.color.violet_dark_accent),
    BLUE(R.style.Theme_local_llm_Blue, R.style.Theme_local_llm_Blue_Light, R.string.accent_blue, R.color.blue_dark_accent),
    MINT(R.style.Theme_local_llm_Mint, R.style.Theme_local_llm_Mint_Light, R.string.accent_mint, R.color.mint_dark_accent),
    ROSE(R.style.Theme_local_llm_Rose, R.style.Theme_local_llm_Rose_Light, R.string.accent_rose, R.color.rose_dark_accent),
    AMBER(R.style.Theme_local_llm_Amber, R.style.Theme_local_llm_Amber_Light, R.string.accent_amber, R.color.amber_dark_accent),
    CORAL(R.style.Theme_local_llm_Coral, R.style.Theme_local_llm_Coral_Light, R.string.accent_coral, R.color.coral_dark_accent);

    fun styleFor(appearance: AppAppearanceMode, context: Context): Int {
        return when (appearance) {
            AppAppearanceMode.DARK -> darkStyleRes
            AppAppearanceMode.LIGHT -> lightStyleRes
            AppAppearanceMode.SYSTEM -> {
                val night = context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK
                if (night == Configuration.UI_MODE_NIGHT_YES) darkStyleRes else lightStyleRes
            }
        }
    }

    companion object {
        fun fromStoredName(name: String?): AppAccentOption {
            // Saved choices from the earlier themes (Cobalt, Emerald, Coral and older) fall back to Violet.
            return entries.firstOrNull { it.name == name } ?: VIOLET
        }
    }
}

/** Dark-theme background. The glow backgrounds follow the accent colour; light themes ignore this. */
enum class AppBackgroundOption(@field:StringRes val labelResId: Int, val overlayRes: Int) {
    AURORA(R.string.background_aurora, R.style.ThemeOverlay_Pocket_Bg_Aurora),
    VIOLET_GLOW(R.string.background_violet_glow, R.style.ThemeOverlay_Pocket_Bg_VioletGlow),
    GRAPHITE(R.string.background_graphite, R.style.ThemeOverlay_Pocket_Bg_Graphite),
    MIDNIGHT(R.string.background_midnight, R.style.ThemeOverlay_Pocket_Bg_Midnight),
    TEAL(R.string.background_teal, R.style.ThemeOverlay_Pocket_Bg_Teal),
    OLED(R.string.background_oled, R.style.ThemeOverlay_Pocket_Bg_Oled);

    companion object {
        fun fromStoredName(name: String?): AppBackgroundOption =
            entries.firstOrNull { it.name == name } ?: AURORA
    }
}

data class AppSettings(
    val accent: AppAccentOption = AppAccentOption.VIOLET,
    val appearance: AppAppearanceMode = AppAppearanceMode.DARK,
    val background: AppBackgroundOption = AppBackgroundOption.AURORA,
    val chatFontSizeSp: Float = 16f
) {
    fun isDark(context: Context): Boolean = when (appearance) {
        AppAppearanceMode.DARK -> true
        AppAppearanceMode.LIGHT -> false
        AppAppearanceMode.SYSTEM ->
            (context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
    }
}

class AppSettingsStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "pocket_chat_settings"
        private const val KEY_ACCENT = "theme"
        private const val KEY_APPEARANCE = "appearance"
        private const val KEY_BACKGROUND = "background"
        private const val KEY_THINK_DEFAULT = "think_default"
        private const val KEY_CHAT_FONT_SIZE = "chat_font_size"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): AppSettings {
        val accent = AppAccentOption.fromStoredName(
            prefs.getString(KEY_ACCENT, AppAccentOption.VIOLET.name)
        )
        val appearance = runCatching {
            AppAppearanceMode.valueOf(prefs.getString(KEY_APPEARANCE, AppAppearanceMode.DARK.name)!!)
        }.getOrDefault(AppAppearanceMode.DARK)

        return AppSettings(
            accent = accent,
            appearance = appearance,
            background = AppBackgroundOption.fromStoredName(prefs.getString(KEY_BACKGROUND, null)),
            chatFontSizeSp = prefs.getFloat(KEY_CHAT_FONT_SIZE, 16f).coerceIn(13f, 24f)
        )
    }

    fun loadThinkByDefault(): Boolean = prefs.getBoolean(KEY_THINK_DEFAULT, false)

    fun saveThinkByDefault(value: Boolean) {
        prefs.edit().putBoolean(KEY_THINK_DEFAULT, value).apply()
    }

    fun save(settings: AppSettings) {
        prefs.edit()
            .putString(KEY_ACCENT, settings.accent.name)
            .putString(KEY_APPEARANCE, settings.appearance.name)
            .putString(KEY_BACKGROUND, settings.background.name)
            .putFloat(KEY_CHAT_FONT_SIZE, settings.chatFontSizeSp)
            .apply()
    }
}
