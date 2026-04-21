package com.example.local_llm

import android.content.Context

class ModelSelectionStore(context: Context) {

    companion object {
        private const val PREFS_NAME = "pocket_chat_model_selection"
        private const val KEY_SELECTED_MODEL_ID = "selected_model_id"
        private const val KEY_SELECTED_IMAGE_MODEL_ID = "selected_image_model_id"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun loadSelectedModel(): ModelDescriptor? {
        return ModelRegistry.findById(prefs.getString(KEY_SELECTED_MODEL_ID, null))
    }

    fun saveSelectedModel(modelId: String) {
        prefs.edit()
            .putString(KEY_SELECTED_MODEL_ID, modelId)
            .apply()
    }

    fun clearSelectedModel() {
        prefs.edit()
            .remove(KEY_SELECTED_MODEL_ID)
            .apply()
    }

    fun loadSelectedImageModel(): FastVlmLiteRtSpec? {
        return ImageModelRegistry.findById(prefs.getString(KEY_SELECTED_IMAGE_MODEL_ID, null))
    }

    fun saveSelectedImageModel(modelId: String) {
        prefs.edit()
            .putString(KEY_SELECTED_IMAGE_MODEL_ID, modelId)
            .apply()
    }

    fun clearSelectedImageModel() {
        prefs.edit()
            .remove(KEY_SELECTED_IMAGE_MODEL_ID)
            .apply()
    }
}
