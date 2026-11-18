package com.example.local_llm

import android.content.Context
import java.util.Locale

/** A language Whisper can be told to expect. An empty [code] means "detect it". */
data class SpeechLanguageOption(val code: String, val label: String)

/**
 * Which language voice input transcribes. The small Whisper model is poor at guessing the language,
 * and a wrong guess (English speech read as Tamil, say) produces repeated nonsense, so the default is
 * the phone's own language and detection is only used when the user asks for it.
 */
object SpeechLanguages {

    const val SETTING_PHONE = "phone"
    const val SETTING_AUTO = "auto"

    val languages: List<SpeechLanguageOption> = listOf(
        SpeechLanguageOption("en", "English"),
        SpeechLanguageOption("de", "Deutsch"),
        SpeechLanguageOption("ta", "தமிழ்"),
        SpeechLanguageOption("hi", "हिन्दी"),
        SpeechLanguageOption("fr", "Français"),
        SpeechLanguageOption("es", "Español"),
        SpeechLanguageOption("it", "Italiano"),
        SpeechLanguageOption("pt", "Português"),
        SpeechLanguageOption("nl", "Nederlands"),
        SpeechLanguageOption("pl", "Polski"),
        SpeechLanguageOption("ru", "Русский"),
        SpeechLanguageOption("uk", "Українська"),
        SpeechLanguageOption("tr", "Türkçe"),
        SpeechLanguageOption("sv", "Svenska"),
        SpeechLanguageOption("ar", "العربية"),
        SpeechLanguageOption("zh", "中文"),
        SpeechLanguageOption("ja", "日本語"),
        SpeechLanguageOption("ko", "한국어")
    )

    private val supportedCodes = languages.map(SpeechLanguageOption::code).toSet()

    /** The Whisper language code for a saved setting; an empty string lets Whisper detect the language. */
    fun whisperCode(setting: String, deviceLocale: Locale): String = when (setting) {
        SETTING_AUTO -> ""
        SETTING_PHONE -> deviceLocale.language.lowercase().takeIf { it in supportedCodes }.orEmpty()
        else -> setting.takeIf { it in supportedCodes }.orEmpty()
    }
}

class SpeechLanguageStore(context: Context) {

    private val prefs = context.getSharedPreferences("pocket_speech", Context.MODE_PRIVATE)

    fun load(): String = prefs.getString(KEY_LANGUAGE, SpeechLanguages.SETTING_PHONE) ?: SpeechLanguages.SETTING_PHONE

    fun save(setting: String) {
        prefs.edit().putString(KEY_LANGUAGE, setting).apply()
    }

    fun whisperCode(): String = SpeechLanguages.whisperCode(load(), Locale.getDefault())

    private companion object {
        const val KEY_LANGUAGE = "language"
    }
}

/**
 * Whisper sometimes gets stuck and repeats one word or phrase many times. Real speech almost never
 * repeats the same words four times in a row, so such runs are cut down to a single copy.
 */
object RepeatedPhraseFilter {

    private const val MIN_REPEATS = 4
    private const val MAX_PHRASE_WORDS = 4

    fun collapse(text: String): String {
        var words = text.trim().split(Regex("\\s+")).filter(String::isNotEmpty)
        if (words.size < MIN_REPEATS) return text.trim()
        for (size in 1..MAX_PHRASE_WORDS) words = collapseRuns(words, size)
        return words.joinToString(" ")
    }

    private fun collapseRuns(words: List<String>, size: Int): List<String> {
        val result = mutableListOf<String>()
        var index = 0
        while (index < words.size) {
            if (index + size > words.size) {
                result += words.subList(index, words.size)
                break
            }
            val phrase = words.subList(index, index + size).map(String::lowercase)
            var repeats = 1
            while (index + (repeats + 1) * size <= words.size &&
                words.subList(index + repeats * size, index + (repeats + 1) * size).map(String::lowercase) == phrase
            ) {
                repeats++
            }
            if (repeats >= MIN_REPEATS) {
                result += words.subList(index, index + size)
                index += repeats * size
            } else {
                result += words[index]
                index++
            }
        }
        return result
    }
}