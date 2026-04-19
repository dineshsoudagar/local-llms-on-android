package com.example.local_llm

import androidx.lifecycle.ViewModel

class PocketChatViewModel : ViewModel() {
    var chatController: PersistentChatController? = null
    var modelId: String? = null

    override fun onCleared() {
        chatController?.close()
        chatController = null
        super.onCleared()
    }
}
