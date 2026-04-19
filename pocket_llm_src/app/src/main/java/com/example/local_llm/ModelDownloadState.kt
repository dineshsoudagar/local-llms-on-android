package com.example.local_llm

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

sealed interface ModelDownloadState {
    data object Idle : ModelDownloadState

    data class Running(
        val modelId: String,
        val modelName: String,
        val fileName: String?,
        val bytesDownloaded: Long,
        val totalBytes: Long?
    ) : ModelDownloadState

    data class Completed(
        val modelId: String,
        val modelName: String
    ) : ModelDownloadState

    data class Failed(
        val modelId: String,
        val modelName: String,
        val message: String
    ) : ModelDownloadState
}

object ModelDownloadStateStore {
    private val _state = MutableStateFlow<ModelDownloadState>(ModelDownloadState.Idle)
    val state: StateFlow<ModelDownloadState> = _state.asStateFlow()

    fun update(state: ModelDownloadState) {
        _state.value = state
    }

    fun clearTerminalState(modelId: String) {
        val current = _state.value
        if (
            (current is ModelDownloadState.Completed && current.modelId == modelId) ||
            (current is ModelDownloadState.Failed && current.modelId == modelId)
        ) {
            _state.value = ModelDownloadState.Idle
        }
    }
}
