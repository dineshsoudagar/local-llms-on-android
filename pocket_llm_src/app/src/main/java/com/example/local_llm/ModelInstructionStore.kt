package com.example.local_llm

import android.content.Context

class ModelInstructionStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "model_instructions"
        private const val PRESET_SUFFIX = "_preset"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun loadInstruction(descriptor: ModelDescriptor): String {
        return prefs.getString(descriptor.id, null)
            ?.takeIf { it.isNotBlank() }
            ?: InstructionPreset.default.instruction
    }

    fun loadPreset(descriptor: ModelDescriptor): InstructionPreset {
        val savedPreset = InstructionPreset.fromId(prefs.getString(presetKey(descriptor), null))
        if (savedPreset != null) {
            return savedPreset
        }

        val savedInstruction = prefs.getString(descriptor.id, null)
            ?.takeIf { it.isNotBlank() }
            ?: return InstructionPreset.default

        return InstructionPreset.matchingInstruction(savedInstruction) ?: InstructionPreset.CUSTOM
    }

    fun saveInstruction(descriptor: ModelDescriptor, instruction: String) {
        val normalizedInstruction = instruction.trim()
        val preset = InstructionPreset.matchingInstruction(normalizedInstruction)
            ?: InstructionPreset.CUSTOM
        saveInstruction(descriptor, normalizedInstruction, preset)
    }

    fun saveInstruction(
        descriptor: ModelDescriptor,
        instruction: String,
        preset: InstructionPreset
    ) {
        val normalizedInstruction = instruction.trim()
        prefs.edit().apply {
            if (
                preset == InstructionPreset.default &&
                normalizedInstruction == InstructionPreset.default.instruction
            ) {
                remove(descriptor.id)
                remove(presetKey(descriptor))
            } else {
                putString(descriptor.id, normalizedInstruction)
                putString(presetKey(descriptor), preset.id)
            }
        }.apply()
    }

    private fun presetKey(descriptor: ModelDescriptor): String {
        return "${descriptor.id}$PRESET_SUFFIX"
    }
}
