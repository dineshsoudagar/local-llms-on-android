package com.example.local_llm

import android.content.Context
import com.google.gson.Gson
import java.io.File

object SessionStorage {
    private val gson = Gson()

    fun saveSession(context: Context, session: ChatSession) {
        val file = File(context.filesDir, "${session.sessionId}.json")
        file.writeText(gson.toJson(session))
    }

    fun loadSession(context: Context, sessionId: String): ChatSession? {
        val file = File(context.filesDir, "$sessionId.json")
        return if (file.exists()) gson.fromJson(file.readText(), ChatSession::class.java) else null
    }

    fun listSessions(context: Context): List<String> {
        return context.filesDir.listFiles()?.filter {
            it.name.endsWith(".json")
        }?.map { it.name.removeSuffix(".json") } ?: emptyList()
    }

    fun deleteSession(context: Context, sessionId: String): Boolean {
        return File(context.filesDir, "$sessionId.json").delete() &&
                File(context.filesDir, "$sessionId.kvcache").delete()
    }
}
