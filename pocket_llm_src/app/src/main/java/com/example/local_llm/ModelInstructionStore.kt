package com.example.local_llm

import android.content.Context

class ModelInstructionStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "model_instructions"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun loadInstruction(descriptor: ModelDescriptor): String {
        return prefs.getString(descriptor.id, null)
            ?.takeIf { it.isNotBlank() }
            ?: descriptor.defaultInstruction
    }

    fun saveInstruction(descriptor: ModelDescriptor, instruction: String) {
        val normalizedInstruction = instruction.trim()
        prefs.edit().apply {
            if (normalizedInstruction == descriptor.defaultInstruction) {
                remove(descriptor.id)
            } else {
                putString(descriptor.id, normalizedInstruction)
            }
        }.apply()
    }
}
