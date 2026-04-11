package com.example.local_llm

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class PersistedChatSession(
    val sessionId: String,
    val title: String,
    val modelId: String,
    val modelDisplayName: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    val turns: List<ChatTurn>
)

data class ChatSessionSummary(
    val sessionId: String,
    val title: String,
    val modelId: String,
    val modelDisplayName: String,
    val updatedAtMillis: Long,
    val preview: String
)

class ChatSessionStore(context: Context) {

    private val sessionsDir = File(context.filesDir, "chat_sessions").apply { mkdirs() }

    fun save(session: PersistedChatSession) {
        fileFor(session.sessionId).writeText(serializeSession(session).toString())
    }

    fun load(sessionId: String): PersistedChatSession? {
        val file = fileFor(sessionId)
        if (!file.exists()) {
            return null
        }

        return runCatching {
            deserializeSession(JSONObject(file.readText()))
        }.getOrNull()
    }

    fun delete(sessionId: String): Boolean {
        return runCatching {
            val file = fileFor(sessionId)
            !file.exists() || file.delete()
        }.getOrDefault(false)
    }

    fun list(): List<ChatSessionSummary> {
        return sessionsDir.listFiles()
            ?.filter { it.extension.equals("json", ignoreCase = true) }
            ?.mapNotNull { file ->
                runCatching {
                    val session = deserializeSession(JSONObject(file.readText()))
                    ChatSessionSummary(
                        sessionId = session.sessionId,
                        title = session.title,
                        modelId = session.modelId,
                        modelDisplayName = session.modelDisplayName,
                        updatedAtMillis = session.updatedAtMillis,
                        preview = buildPreview(session.turns)
                    )
                }.getOrNull()
            }
            ?.sortedByDescending { it.updatedAtMillis }
            ?: emptyList()
    }

    private fun fileFor(sessionId: String): File {
        return File(sessionsDir, "$sessionId.json")
    }

    private fun serializeSession(session: PersistedChatSession): JSONObject {
        return JSONObject().apply {
            put("sessionId", session.sessionId)
            put("title", session.title)
            put("modelId", session.modelId)
            put("modelDisplayName", session.modelDisplayName)
            put("createdAtMillis", session.createdAtMillis)
            put("updatedAtMillis", session.updatedAtMillis)
            put(
                "turns",
                JSONArray().apply {
                    session.turns.forEach { turn ->
                        put(
                            JSONObject().apply {
                                put("id", turn.id)
                                put("role", turn.role.name)
                                put("text", turn.text)
                                put("thinkingText", turn.thinkingText ?: JSONObject.NULL)
                                put("stopped", turn.stopped)
                                put("renderAsMarkdown", turn.renderAsMarkdown)
                            }
                        )
                    }
                }
            )
        }
    }

    private fun deserializeSession(json: JSONObject): PersistedChatSession {
        val turnsArray = json.optJSONArray("turns") ?: JSONArray()
        val turns = buildList {
            for (index in 0 until turnsArray.length()) {
                val turnJson = turnsArray.getJSONObject(index)
                add(
                    ChatTurn(
                        id = turnJson.optString("id").ifBlank { java.util.UUID.randomUUID().toString() },
                        role = ChatRole.valueOf(turnJson.getString("role")),
                        text = turnJson.optString("text"),
                        thinkingText = turnJson.optString("thinkingText").takeUnless { it.isNullOrBlank() || it == "null" },
                        stopped = turnJson.optBoolean("stopped"),
                        renderAsMarkdown = turnJson.optBoolean("renderAsMarkdown", true)
                    )
                )
            }
        }

        return PersistedChatSession(
            sessionId = json.getString("sessionId"),
            title = json.optString("title").ifBlank { "Untitled chat" },
            modelId = json.optString("modelId"),
            modelDisplayName = json.optString("modelDisplayName"),
            createdAtMillis = json.optLong("createdAtMillis"),
            updatedAtMillis = json.optLong("updatedAtMillis"),
            turns = turns
        )
    }

    private fun buildPreview(turns: List<ChatTurn>): String {
        return turns.lastOrNull { !it.isUser && it.text.isNotBlank() }?.text
            ?: turns.firstOrNull { it.text.isNotBlank() }?.text
            ?: ""
    }
}
