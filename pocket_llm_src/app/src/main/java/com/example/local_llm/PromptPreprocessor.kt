package com.example.local_llm

object PromptPreprocessor {
    private val inlineWhitespace = Regex("[\\t\\x0B\\f]+")
    private val excessBlankLines = Regex("\\n{3,}")

    fun normalize(text: String): String {
        return text
            .replace("\r\n", "\n")
            .replace('\r', '\n')
            .lineSequence()
            .map { line -> line.replace(inlineWhitespace, " ").trimEnd() }
            .joinToString("\n")
            .replace(excessBlankLines, "\n\n")
            .trim()
    }

    fun mergeTypedAndRecognized(
        typedText: String,
        recognizedText: String
    ): String {
        val typed = normalize(typedText)
        val recognized = normalize(recognizedText)

        return when {
            typed.isBlank() -> recognized
            recognized.isBlank() -> typed
            else -> "$typed\n\n$recognized"
        }
    }
}
