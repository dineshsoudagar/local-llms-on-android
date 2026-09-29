package com.example.local_llm

import android.content.Context
import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Browser conversations live in the phone's private app storage, separate from native chat. */
internal class LanChatStore(context: Context) {
    private val file = AtomicFile(File(context.filesDir, "lan_browser_chats.json"))

    @Synchronized
    fun list(): JSONArray {
        val chats = read()
        val summaries = JSONArray()
        val sorted = (0 until chats.length()).mapNotNull(chats::optJSONObject)
            .sortedByDescending { it.optLong("updatedAt") }
        for (chat in sorted) {
            summaries.put(JSONObject()
                .put("id", chat.optString("id"))
                .put("title", chat.optString("title"))
                .put("updatedAt", chat.optLong("updatedAt")))
        }
        return summaries
    }

    @Synchronized
    fun get(id: String): JSONObject? {
        validateId(id)
        val chats = read()
        return (0 until chats.length()).mapNotNull(chats::optJSONObject)
            .firstOrNull { it.optString("id") == id }
    }

    @Synchronized
    fun save(request: JSONObject): JSONObject {
        val id = request.optString("id")
        validateId(id)
        val messages = request.optJSONArray("messages")
            ?: throw IllegalArgumentException("Messages are required.")
        require(messages.length() in 1..200) { "A chat must have 1 to 200 messages." }
        val safeMessages = JSONArray()
        for (index in 0 until messages.length()) {
            val message = messages.optJSONObject(index)
                ?: throw IllegalArgumentException("Invalid message.")
            val role = message.optString("role")
            require(role == "user" || role == "assistant") { "Invalid message role." }
            val content = message.optString("content")
            require(content.isNotBlank() && content.length <= 16_000) { "Invalid message content." }
            safeMessages.put(JSONObject().put("role", role).put("content", content))
        }
        val title = safeMessages.optJSONObject(0)?.optString("content")
            ?.replace(Regex("\\s+"), " ")?.trim()?.take(56).orEmpty()
        val chat = JSONObject().put("id", id).put("title", title)
            .put("updatedAt", System.currentTimeMillis()).put("messages", safeMessages)
        val existing = read()
        val output = JSONArray()
        for (index in 0 until existing.length()) {
            val item = existing.optJSONObject(index) ?: continue
            if (item.optString("id") != id) output.put(item)
        }
        require(output.length() < 100) { "The 100-chat storage limit has been reached." }
        output.put(chat)
        write(output)
        return chat
    }

    @Synchronized
    fun delete(id: String) {
        validateId(id)
        val existing = read()
        val output = JSONArray()
        for (index in 0 until existing.length()) {
            val item = existing.optJSONObject(index) ?: continue
            if (item.optString("id") != id) output.put(item)
        }
        write(output)
    }

    private fun validateId(id: String) {
        require(id.matches(Regex("[A-Za-z0-9_-]{1,64}"))) { "Invalid chat ID." }
    }

    private fun read(): JSONArray = try {
        JSONArray(String(file.openRead().use { it.readBytes() }, Charsets.UTF_8))
    } catch (_: java.io.FileNotFoundException) {
        JSONArray()
    }

    private fun write(chats: JSONArray) {
        val bytes = chats.toString().toByteArray(Charsets.UTF_8)
        require(bytes.size <= 2 * 1024 * 1024) { "Chat storage is full." }
        val stream = file.startWrite()
        try {
            stream.write(bytes)
            file.finishWrite(stream)
        } catch (error: Exception) {
            file.failWrite(stream)
            throw error
        }
    }
}
