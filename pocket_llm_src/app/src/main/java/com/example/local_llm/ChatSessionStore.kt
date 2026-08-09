package com.example.local_llm

import android.content.Context
import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

private const val CHAT_SESSION_TITLE_MAX_CHARS = 42
private const val UNTITLED_CHAT_TITLE = "Untitled chat"

data class PersistedChatSession(
    val sessionId: String,
    val title: String,
    val modelId: String,
    val modelDisplayName: String,
    val createdAtMillis: Long,
    val updatedAtMillis: Long,
    val turns: List<ChatTurn>,
    val activeAttachmentId: String? = null
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
    private val attachmentRepository = AttachmentRepository(context)

    fun save(session: PersistedChatSession) {
        val target = fileFor(session.sessionId)
        val atomicFile = AtomicFile(target)
        val output = atomicFile.startWrite()
        try {
            output.write(serializeSession(session).toString().toByteArray(Charsets.UTF_8))
            atomicFile.finishWrite(output)
        } catch (error: Throwable) {
            atomicFile.failWrite(output)
            throw error
        }
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
            val sessionDeleted = !file.exists() || file.delete()
            val attachmentsDeleted = attachmentRepository.deleteSession(sessionId)
            sessionDeleted && attachmentsDeleted
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
                        title = buildChatSessionTitle(session.turns),
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
            session.activeAttachmentId?.let { put("activeAttachmentId", it) }
            put(
                "turns",
                JSONArray().apply {
                    session.turns.forEach { turn ->
                        put(
                            JSONObject().apply {
                                put("id", turn.id)
                                put("role", turn.role.name)
                                put("text", turn.text)
                                if (turn.displayText != null) {
                                    put("displayText", turn.displayText)
                                }
                                turn.thinkingText?.let { put("thinkingText", it) }
                                turn.thinkingDurationMillis?.let { put("thinkingDurationMillis", it) }
                                put("stopped", turn.stopped)
                                put("renderAsMarkdown", turn.renderAsMarkdown)
                                put("isStreaming", turn.isStreaming)
                                put("contentType", turn.contentType.name)
                                turn.imagePath?.let { put("imagePath", it) }
                                turn.attachmentId?.let { put("attachmentId", it) }
                                turn.attachmentName?.let { put("attachmentName", it) }
                                turn.attachmentKind?.let { put("attachmentKind", it.name) }
                                turn.attachmentProcessingRoute?.let {
                                    put("attachmentProcessingRoute", it.name)
                                }
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
                        displayText = if (turnJson.has("displayText")) turnJson.getString("displayText") else null,
                        thinkingText = turnJson.optString("thinkingText").takeIf { it.isNotBlank() },
                        thinkingDurationMillis = turnJson.optLong("thinkingDurationMillis")
                            .takeIf { turnJson.has("thinkingDurationMillis") },
                        stopped = turnJson.optBoolean("stopped"),
                        renderAsMarkdown = turnJson.optBoolean("renderAsMarkdown", true),
                        isStreaming = turnJson.optBoolean("isStreaming", false),
                        contentType = runCatching {
                            ChatTurnContentType.valueOf(
                                turnJson.optString("contentType", ChatTurnContentType.TEXT.name)
                            )
                        }.getOrDefault(ChatTurnContentType.TEXT),
                        imagePath = if (turnJson.has("imagePath")) turnJson.getString("imagePath") else null,
                        attachmentId = turnJson.optString("attachmentId").takeIf { it.isNotBlank() },
                        attachmentName = turnJson.optString("attachmentName").takeIf { it.isNotBlank() },
                        attachmentKind = turnJson.optString("attachmentKind").takeIf { it.isNotBlank() }
                            ?.let { runCatching { AttachmentKind.valueOf(it) }.getOrNull() },
                        attachmentProcessingRoute = turnJson.optString("attachmentProcessingRoute")
                            .takeIf { it.isNotBlank() }
                            ?.let { runCatching { AttachmentProcessingRoute.valueOf(it) }.getOrNull() }
                    )
                )
            }
        }

        return PersistedChatSession(
            sessionId = json.getString("sessionId"),
            title = json.optString("title").ifBlank { UNTITLED_CHAT_TITLE },
            modelId = json.optString("modelId"),
            modelDisplayName = json.optString("modelDisplayName"),
            createdAtMillis = json.optLong("createdAtMillis"),
            updatedAtMillis = json.optLong("updatedAtMillis"),
            turns = turns,
            activeAttachmentId = json.optString("activeAttachmentId").takeIf { it.isNotBlank() }
        )
    }

    private fun buildPreview(turns: List<ChatTurn>): String {
        return turns.lastOrNull { !it.isUser && it.text.isNotBlank() }?.text
            ?: turns.firstOrNull { it.transcriptText.isNotBlank() }
                ?.transcriptText
            ?: ""
    }
}

fun buildChatSessionTitle(turns: List<ChatTurn>): String {
    val compactPrompt = turns.lastOrNull { it.isUser }
        ?.transcriptText
        .orEmpty()
        .lineSequence()
        .joinToString(" ")
        .trim()
        .replace(Regex("\\s+"), " ")

    return when {
        compactPrompt.isBlank() -> UNTITLED_CHAT_TITLE
        compactPrompt.length <= CHAT_SESSION_TITLE_MAX_CHARS -> compactPrompt
        else -> compactPrompt.take(CHAT_SESSION_TITLE_MAX_CHARS).trimEnd() + "..."
    }
}
