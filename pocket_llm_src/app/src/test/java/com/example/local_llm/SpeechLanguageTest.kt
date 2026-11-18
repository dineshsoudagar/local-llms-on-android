package com.example.local_llm

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Locale

class SpeechLanguageTest {

    @Test
    fun phoneSettingFollowsTheDeviceLanguage() {
        assertEquals("en", SpeechLanguages.whisperCode(SpeechLanguages.SETTING_PHONE, Locale.US))
        assertEquals("de", SpeechLanguages.whisperCode(SpeechLanguages.SETTING_PHONE, Locale.GERMANY))
        assertEquals("ta", SpeechLanguages.whisperCode(SpeechLanguages.SETTING_PHONE, Locale("ta", "IN")))
    }

    @Test
    fun unsupportedPhoneLanguageFallsBackToDetection() {
        assertEquals("", SpeechLanguages.whisperCode(SpeechLanguages.SETTING_PHONE, Locale("xx")))
    }

    @Test
    fun autoMeansDetect() {
        assertEquals("", SpeechLanguages.whisperCode(SpeechLanguages.SETTING_AUTO, Locale.US))
    }

    @Test
    fun explicitLanguageWinsOverThePhone() {
        assertEquals("ta", SpeechLanguages.whisperCode("ta", Locale.US))
        assertEquals("", SpeechLanguages.whisperCode("klingon", Locale.US))
    }

    @Test
    fun repeatedSingleWordIsCollapsed() {
        assertEquals("வருக்கும்", RepeatedPhraseFilter.collapse("வருக்கும் வருக்கும் வருக்கும் வருக்கும் வருக்கும் வருக்கும்"))
    }

    @Test
    fun repeatedPhraseIsCollapsedAndSurroundingWordsKept() {
        assertEquals("start a b end", RepeatedPhraseFilter.collapse("start a b a b a b a b end"))
    }

    @Test
    fun ordinaryRepetitionIsLeftAlone() {
        assertEquals("very very good indeed", RepeatedPhraseFilter.collapse("very very good indeed"))
        assertEquals("no no no way", RepeatedPhraseFilter.collapse("no no no way"))
    }

    @Test
    fun repeatsAreMatchedIgnoringCase() {
        assertEquals("Yes", RepeatedPhraseFilter.collapse("Yes yes YES yes"))
    }

    @Test
    fun shortAndEmptyTextIsReturnedTrimmed() {
        assertEquals("", RepeatedPhraseFilter.collapse("   "))
        assertEquals("hi", RepeatedPhraseFilter.collapse(" hi "))
    }
}