package com.example.local_llm

import android.content.Context

object LanServerControllerRegistry {
    private data class Entry(
        val modelId: String,
        val controller: PersistentChatController
    )

    private var entry: Entry? = null

    @Synchronized
    fun getOrCreate(
        context: Context,
        descriptor: ModelDescriptor,
        initializationPolicy: BackendInitializationPolicy = BackendInitializationPolicy()
    ): PersistentChatController {
        val current = entry
        if (current?.modelId == descriptor.id) {
            return current.controller
        }

        current?.controller?.close()
        val controller = PersistentChatController(context.applicationContext, descriptor, initializationPolicy)
        entry = Entry(descriptor.id, controller)
        return controller
    }

    @Synchronized
    fun register(modelId: String, controller: PersistentChatController) {
        entry = Entry(modelId, controller)
    }

    @Synchronized
    fun peek(modelId: String): PersistentChatController? {
        return entry?.takeIf { it.modelId == modelId }?.controller
    }

    @Synchronized
    fun clearIf(controller: PersistentChatController?) {
        if (controller != null && entry?.controller === controller) {
            entry = null
        }
    }
}
