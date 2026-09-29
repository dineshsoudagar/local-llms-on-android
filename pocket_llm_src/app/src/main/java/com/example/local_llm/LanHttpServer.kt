package com.example.local_llm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.net.Inet4Address
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.security.SecureRandom
import java.util.Collections
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class LanHttpServer(
    private val context: Context,
    private val controller: PersistentChatController,
    private val modelId: String,
    private val passwordVerifier: (String) -> Boolean,
    private val apiKeyVerifier: (String) -> Boolean,
    private val port: Int = DEFAULT_PORT
) {
    private enum class LanUiUploadKind { PDF, IMAGE }

    private data class LanUiUpload(
        val id: String,
        val chatId: String,
        val displayName: String,
        val kind: LanUiUploadKind,
        val attachmentId: String? = null,
        val file: File? = null
    )

    companion object {
        const val DEFAULT_PORT = 8080
        private const val UI_PATH = "/ui"
        private const val MAX_LINE_LENGTH = 16 * 1024
        private const val MAX_BODY_BYTES = 1 * 1024 * 1024
        private const val MAX_UPLOAD_BYTES = 16 * 1024 * 1024
        private const val MAX_IMAGE_UPLOAD_BYTES = 12 * 1024 * 1024
        private const val REQUEST_TIMEOUT_MS = 5 * 60 * 1000
        private const val SESSION_TTL_MS = 12 * 60 * 60 * 1000L


        private val STATUS_REASONS = mapOf(
            200 to "OK",
            400 to "Bad Request",
            401 to "Unauthorized",
            403 to "Forbidden",
            404 to "Not Found",
            405 to "Method Not Allowed",
            409 to "Conflict",
            413 to "Payload Too Large",
            500 to "Internal Server Error",
            503 to "Service Unavailable"
        )
    }

    private val acceptExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val requestExecutor: ExecutorService = Executors.newFixedThreadPool(2)
    private val sessionTokens = ConcurrentHashMap<String, Long>()
    private val secureRandom = SecureRandom()
    private val chatStore = LanChatStore(context)
    private val attachmentRepository = AttachmentRepository(context)
    private val attachmentImporter = AttachmentImporter(context, attachmentRepository)
    private val uploads = ConcurrentHashMap<String, LanUiUpload>()
    private val webUiHtml: String = context.assets.open("lan_ui.html")
        .bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
    private val logoBytes: ByteArray = runCatching {
        val bitmap = BitmapFactory.decodeResource(context.resources, R.mipmap.ic_launcher_2)
            ?: BitmapFactory.decodeResource(context.resources, R.mipmap.ic_launcher_2_foreground)
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        stream.toByteArray()
    }.getOrDefault(ByteArray(0))
    @Volatile
    private var serverSocket: ServerSocket? = null

    val endpoint: String
        get() = "http://${localLanAddress().hostAddress}:$port$UI_PATH"

    fun start() {
        check(serverSocket == null) { "The LAN server is already running." }
        val socket = ServerSocket(port, 16, InetAddress.getByName("0.0.0.0"))
        socket.reuseAddress = true
        serverSocket = socket
        acceptExecutor.execute {
            while (!socket.isClosed) {
                try {
                    val client = socket.accept()
                    requestExecutor.execute { handle(client) }
                } catch (_: IOException) {
                    if (!socket.isClosed) {
                        continue
                    }
                }
            }
        }
    }

    fun close() {
        serverSocket?.close()
        serverSocket = null
        acceptExecutor.shutdownNow()
        requestExecutor.shutdownNow()
        uploads.values.forEach { it.file?.delete() }
        uploads.clear()
    }

    private fun handle(socket: Socket) {
        socket.use { client ->
            client.soTimeout = REQUEST_TIMEOUT_MS
            val input = BufferedInputStream(client.getInputStream())
            val writer = BufferedWriter(OutputStreamWriter(client.getOutputStream(), StandardCharsets.UTF_8))

            if (!isAllowedLanAddress(client.inetAddress)) {
                writeJson(writer, 403, errorJson("Only private-network clients are allowed."))
                return
            }

            val requestLine = readLimitedLine(input)
            if (requestLine == null) {
                return
            }
            val requestParts = requestLine.split(' ', limit = 3)
            if (requestParts.size != 3) {
                writeJson(writer, 400, errorJson("Malformed HTTP request line."))
                return
            }

            val headers = mutableMapOf<String, String>()
            while (true) {
                val line = readLimitedLine(input) ?: run {
                    writeJson(writer, 400, errorJson("Malformed HTTP headers."))
                    return
                }
                if (line.isEmpty()) {
                    break
                }
                val separator = line.indexOf(':')
                if (separator <= 0) {
                    writeJson(writer, 400, errorJson("Malformed HTTP header."))
                    return
                }
                headers[line.substring(0, separator).trim().lowercase()] =
                    line.substring(separator + 1).trim()
            }

            val path = requestParts[1].substringBefore('?')
            if (requestParts[0] == "GET" && (path == "/" || path == UI_PATH)) {
                writeHtml(writer, 200, webUiHtml)
                return
            }
            if (requestParts[0] == "GET" && path == "/logo.webp") {
                writeBinary(client.getOutputStream(), 200, "image/webp", logoBytes)
                return
            }

            val isUpload = requestParts[0] == "POST" && path == "/ui/attachments"
            val maximumBodyBytes = if (isUpload) MAX_UPLOAD_BYTES else MAX_BODY_BYTES
            val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
            if (contentLength < 0 || contentLength > maximumBodyBytes) {
                writeJson(writer, 413, errorJson("Request body is too large."))
                return
            }

            val bodyBytes = ByteArray(contentLength)
            var read = 0
            while (read < contentLength) {
                val count = input.read(bodyBytes, read, contentLength - read)
                if (count < 0) {
                    writeJson(writer, 400, errorJson("Request body ended early."))
                    return
                }
                read += count
            }

            if (requestParts[0] == "POST" && path == "/auth/login") {
                handleLogin(writer, String(bodyBytes, StandardCharsets.UTF_8))
                return
            }

            if (!isAuthorized(headers["authorization"].orEmpty())) {
                writeJson(writer, 401, errorJson(authorizationError(headers["authorization"].orEmpty())))
                return
            }

            when {
                requestParts[0] == "POST" && path == "/ui/attachments" -> {
                    handleUiAttachmentUpload(writer, headers, bodyBytes)
                }

                path == "/ui/chats" || path == "/ui/chats/delete" || path.startsWith("/ui/chats/") -> {
                    handleUiChats(writer, requestParts[0], path, String(bodyBytes, StandardCharsets.UTF_8))
                }

                requestParts[0] == "GET" && path == "/v1/models" -> {
                    writeModels(writer)
                }

                requestParts[0] == "GET" && path == "/v1/models/$modelId" -> {
                    writeModel(writer)
                }

                requestParts[0] == "GET" && path == "/health" -> {
                    writeJson(
                        writer,
                        200,
                        JSONObject()
                            .put("status", "ok")
                            .put("model", controller.state.value.title)
                    )
                }

                requestParts[0] == "POST" && path == "/v1/chat/completions" -> {
                    handleChatRequest(writer, String(bodyBytes, StandardCharsets.UTF_8))
                }

                requestParts[0] != "GET" && requestParts[0] != "POST" -> {
                    writeJson(writer, 405, errorJson("Only GET and POST are supported."))
                }

                else -> writeJson(writer, 404, errorJson("Unknown endpoint."))
            }
        }
    }

    private fun handleUiChats(writer: BufferedWriter, method: String, path: String, body: String) {
        try {
            when {
                method == "GET" && path == "/ui/chats" ->
                    writeJson(writer, 200, JSONObject().put("chats", chatStore.list()))
                method == "GET" && path.startsWith("/ui/chats/") -> {
                    val chat = chatStore.get(path.removePrefix("/ui/chats/"))
                    if (chat == null) writeJson(writer, 404, errorJson("Chat not found."))
                    else writeJson(writer, 200, JSONObject().put("chat", chat))
                }
                method == "POST" && path == "/ui/chats" ->
                    writeJson(writer, 200, JSONObject().put("chat", chatStore.save(JSONObject(body))))
                method == "POST" && path == "/ui/chats/delete" -> {
                    val chatId = JSONObject(body).optString("id")
                    chatStore.delete(chatId)
                    attachmentRepository.deleteSession(chatId)
                    uploads.values
                        .filter { it.chatId == chatId }
                        .forEach { upload -> uploads.remove(upload.id)?.file?.delete() }
                    writeJson(writer, 200, JSONObject().put("deleted", true))
                }
                else -> writeJson(writer, 405, errorJson("Unsupported chat operation."))
            }
        } catch (error: IllegalArgumentException) {
            writeJson(writer, 400, errorJson(error.message ?: "Invalid chat."))
        } catch (_: org.json.JSONException) {
            writeJson(writer, 400, errorJson("Request body must be valid JSON."))
        } catch (_: IOException) {
            writeJson(writer, 500, errorJson("Could not save chat on the phone."))
        }
    }

    private fun handleUiAttachmentUpload(
        writer: BufferedWriter,
        headers: Map<String, String>,
        body: ByteArray
    ) {
        try {
            val chatId = decodeUploadHeader(headers["x-chat-id"])
            require(chatId.matches(Regex("[A-Za-z0-9_-]{1,64}"))) { "Invalid chat ID." }
            val displayName = decodeUploadHeader(headers["x-upload-name"])
                .takeIf(String::isNotBlank)?.take(160) ?: "attachment"
            val contentType = headers["content-type"].orEmpty().substringBefore(';').lowercase()
            val isPdf = contentType == "application/pdf" || displayName.endsWith(".pdf", ignoreCase = true)
            val isImage = contentType.startsWith("image/")
            require(isPdf || isImage) { "Choose a PDF or image file." }
            if (isImage) {
                require(body.size <= MAX_IMAGE_UPLOAD_BYTES) { "Images must be 12 MiB or smaller." }
                require(controller.state.value.supportsDirectImageInput) {
                    "The selected model does not support image input. Change the model in the app."
                }
            }

            val uploadId = UUID.randomUUID().toString()
            val extension = displayName.substringAfterLast('.', "").lowercase()
                .filter(Char::isLetterOrDigit).take(8).ifBlank { if (isPdf) "pdf" else "image" }
            val temporary = File(context.cacheDir, "lan-upload-$uploadId.$extension")
            temporary.writeBytes(body)
            if (isPdf) {
                val descriptor = try {
                    runBlocking(Dispatchers.IO) {
                        controller.withRuntimeLease {
                            attachmentImporter.import(
                                uri = Uri.fromFile(temporary),
                                sessionId = chatId,
                                requestedKind = AttachmentKind.PDF,
                                useGemmaNativeAudio = false
                            )
                        }
                    }
                } finally {
                    temporary.delete()
                }
                uploads[uploadId] = LanUiUpload(
                    id = uploadId,
                    chatId = chatId,
                    displayName = descriptor.displayName,
                    kind = LanUiUploadKind.PDF,
                    attachmentId = descriptor.id
                )
            } else {
                uploads[uploadId] = LanUiUpload(
                    id = uploadId,
                    chatId = chatId,
                    displayName = displayName,
                    kind = LanUiUploadKind.IMAGE,
                    file = temporary
                )
            }
            val upload = checkNotNull(uploads[uploadId])
            writeJson(writer, 200, JSONObject()
                .put("id", upload.id)
                .put("name", upload.displayName)
                .put("kind", upload.kind.name.lowercase()))
        } catch (error: IllegalArgumentException) {
            writeJson(writer, 400, errorJson(error.message ?: "Invalid attachment."))
        } catch (error: Exception) {
            writeJson(writer, 500, errorJson(error.message ?: "Attachment upload failed."))
        }
    }

    private fun decodeUploadHeader(value: String?): String {
        require(!value.isNullOrBlank()) { "Attachment metadata is missing." }
        return URLDecoder.decode(value, StandardCharsets.UTF_8.name())
    }

    private fun handleChatRequest(writer: BufferedWriter, body: String) {
        val request = try {
            parseChatRequest(body)
        } catch (error: IllegalArgumentException) {
            writeJson(writer, 400, errorJson(error.message ?: "Invalid chat request."))
            return
        } catch (_: Exception) {
            writeJson(writer, 400, errorJson("Request body must be valid JSON."))
            return
        }
        val uploadsForRequest = try {
            resolveUploads(body, request.history)
        } catch (error: IllegalArgumentException) {
            writeJson(writer, 400, errorJson(error.message ?: "Invalid attachment request."))
            return
        }
        val history = uploadsForRequest.history

        if (!request.model.isNullOrBlank() && request.model != modelId) {
            writeJson(writer, 400, errorJson("Only the configured model '$modelId' is available."))
            return
        }

        if (
            (request.tools.isNotEmpty() || request.hasToolMessages) &&
            !controller.supportsNativeToolCalling()
        ) {
            writeJson(
                writer,
                400,
                errorJson("The configured model backend does not support native tool calling. Select a LiteRT model.")
            )
            return
        }
        if (
            request.toolChoice.mode != ExternalToolChoiceMode.NONE &&
            request.toolChoice.mode != ExternalToolChoiceMode.AUTO &&
            request.toolsForInference().isEmpty()
        ) {
            writeJson(writer, 400, errorJson("tool_choice requires at least one declared function tool."))
            return
        }

        val completionId = "chatcmpl-${UUID.randomUUID()}"
        val created = System.currentTimeMillis() / 1000L
        try {
            if (request.stream) {
                writeSseHeaders(writer)
                writeSseChunk(
                    writer,
                    completionId,
                    created,
                    JSONObject().put("role", "assistant").put("content", ""),
                    null
                )
                var streamedText = ""
                var streamedToolCallCount = 0
                val response = runBlocking(Dispatchers.IO) {
                    controller.sendExternalChatAndAwait(
                        history = history,
                        systemInstruction = request.systemInstruction,
                        outputTokenReserve = request.outputTokenReserve,
                        tools = request.toolsForInference(),
                        toolChoice = request.toolChoice,
                        parallelToolCalls = request.parallelToolCalls,
                        imageFilePaths = uploadsForRequest.imageFilePaths,
                        onPartial = { partial ->
                            val delta = textDelta(streamedText, partial.text)
                            streamedText = partial.text
                            if (delta.isNotEmpty()) {
                                writeSseChunk(
                                    writer,
                                    completionId,
                                    created,
                                    JSONObject().put("content", delta),
                                    null
                                )
                            }
                            if (partial.toolCalls.size > streamedToolCallCount) {
                                val newToolCalls = partial.toolCalls.drop(streamedToolCallCount)
                                writeSseChunk(
                                    writer,
                                    completionId,
                                    created,
                                    JSONObject().put(
                                        "tool_calls",
                                        LanChatProtocol.toolCallsJson(
                                            newToolCalls,
                                            includeIndex = true,
                                            startIndex = streamedToolCallCount
                                        )
                                    ),
                                    null
                                )
                                streamedToolCallCount = partial.toolCalls.size
                            }
                        }
                    )
                }
                validateToolChoice(request, response)
                val finalDelta = textDelta(streamedText, response.text)
                if (finalDelta.isNotEmpty()) {
                    writeSseChunk(
                        writer,
                        completionId,
                        created,
                        JSONObject().put("content", finalDelta),
                        null
                    )
                }
                if (response.toolCalls.size > streamedToolCallCount) {
                    val newToolCalls = response.toolCalls.drop(streamedToolCallCount)
                    writeSseChunk(
                        writer,
                        completionId,
                        created,
                        JSONObject().put(
                            "tool_calls",
                            LanChatProtocol.toolCallsJson(
                                newToolCalls,
                                includeIndex = true,
                                startIndex = streamedToolCallCount
                            )
                        ),
                        null
                    )
                }
                writeSseChunk(
                    writer,
                    completionId,
                    created,
                    JSONObject(),
                    if (response.toolCalls.isEmpty()) "stop" else "tool_calls"
                )
                writer.write("data: [DONE]\n\n")
                writer.flush()
            } else {
                val response = runBlocking(Dispatchers.IO) {
                    controller.sendExternalChatAndAwait(
                        history = history,
                        systemInstruction = request.systemInstruction,
                        outputTokenReserve = request.outputTokenReserve,
                        tools = request.toolsForInference(),
                        toolChoice = request.toolChoice,
                    parallelToolCalls = request.parallelToolCalls,
                    imageFilePaths = uploadsForRequest.imageFilePaths
                    )
                }
                validateToolChoice(request, response)
                writeJson(writer, 200, LanChatProtocol.completionResponse(completionId, created, modelId, response))
            }
        } catch (error: ToolChoiceNotSatisfiedException) {
            writeChatError(writer, request.stream, 422, error.message ?: "The requested tool choice was not satisfied.")
        } catch (error: IllegalStateException) {
            val message = error.message ?: "The model is not ready."
            val status = if (message.contains("busy", ignoreCase = true)) 409 else 503
            writeChatError(writer, request.stream, status, message)
        } catch (error: Exception) {
            writeChatError(writer, request.stream, 500, error.message ?: "Inference failed.")
        } finally {
            uploadsForRequest.imageUploadIds.forEach { uploadId ->
                uploads.remove(uploadId)?.file?.delete()
            }
        }
    }

    private data class ResolvedUploads(
        val history: List<ChatTurn>,
        val imageFilePaths: List<String>,
        val imageUploadIds: List<String>
    )

    private fun resolveUploads(body: String, history: List<ChatTurn>): ResolvedUploads {
        val requestJson = JSONObject(body)
        val attachmentIds = requestJson.optJSONArray("attachments") ?: return ResolvedUploads(
            history = history,
            imageFilePaths = emptyList(),
            imageUploadIds = emptyList()
        )
        require(attachmentIds.length() <= 4) { "Attach up to four files per message." }
        val chatId = requestJson.optString("chat_id")
        require(chatId.matches(Regex("[A-Za-z0-9_-]{1,64}"))) { "Invalid chat ID." }
        val selected = (0 until attachmentIds.length()).map { index ->
            val id = attachmentIds.optString(index)
            uploads[id] ?: throw IllegalArgumentException("Attachment is no longer available. Upload it again.")
        }
        require(selected.all { it.chatId == chatId }) { "Attachments belong to a different chat." }
        val imageUploads = selected.filter { it.kind == LanUiUploadKind.IMAGE }
        val pdfUploads = selected.filter { it.kind == LanUiUploadKind.PDF }
        val enrichedHistory = history.toMutableList()
        if (pdfUploads.isNotEmpty()) {
            val lastUserIndex = enrichedHistory.indexOfLast { it.role == ChatRole.USER }
            require(lastUserIndex >= 0) { "A PDF needs a user message." }
            val prompt = enrichedHistory[lastUserIndex].text
            val excerpts = pdfUploads.map { upload ->
                val descriptor = attachmentRepository.loadDescriptor(chatId, checkNotNull(upload.attachmentId))
                    ?: throw IllegalArgumentException("The PDF is no longer available.")
                val chunks = attachmentRepository.loadChunks(descriptor)
                require(chunks.isNotEmpty()) { "The PDF contains no usable text." }
                val selectedChunks = Bm25AttachmentRetriever(chunks).retrieve(prompt, 900)
                "[PDF: ${upload.displayName}]\n" + selectedChunks.joinToString("\n\n") { chunk ->
                    "[${chunk.source.label()}]\n${chunk.text}"
                }
            }
            enrichedHistory[lastUserIndex] = enrichedHistory[lastUserIndex].copy(
                text = "$prompt\n\n${excerpts.joinToString("\n\n")}"
            )
        }
        return ResolvedUploads(
            history = enrichedHistory,
            imageFilePaths = imageUploads.map { checkNotNull(it.file).absolutePath },
            imageUploadIds = imageUploads.map(LanUiUpload::id)
        )
    }

    private class ToolChoiceNotSatisfiedException(message: String) : Exception(message)

    private fun validateToolChoice(request: ParsedChatRequest, response: BackendResponse) {
        when (request.toolChoice.mode) {
            ExternalToolChoiceMode.REQUIRED -> {
                if (response.toolCalls.isEmpty()) {
                    throw ToolChoiceNotSatisfiedException(
                        "The model did not return a tool call required by tool_choice."
                    )
                }
            }
            ExternalToolChoiceMode.FUNCTION -> {
                if (response.toolCalls.none { it.name == request.toolChoice.functionName }) {
                    throw ToolChoiceNotSatisfiedException(
                        "The model did not return the required tool '${request.toolChoice.functionName}'."
                    )
                }
            }
            else -> Unit
        }
    }

    private fun handleLogin(writer: BufferedWriter, body: String) {
        val request = try {
            JSONObject(body)
        } catch (_: Exception) {
            writeJson(writer, 400, errorJson("Request body must be valid JSON."))
            return
        }
        val password = request.optString("password", "")
        if (password.isBlank() || !passwordVerifier(password)) {
            writeJson(writer, 401, errorJson("Incorrect server password."))
            return
        }

        val tokenBytes = ByteArray(32)
        secureRandom.nextBytes(tokenBytes)
        val token = Base64.encodeToString(
            tokenBytes,
            Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING
        )
        sessionTokens[token] = System.currentTimeMillis() + SESSION_TTL_MS
        writeJson(
            writer,
            200,
            JSONObject()
                .put("token", token)
                .put("model", controller.state.value.title)
        )
    }

    private fun isAuthorized(authorization: String): Boolean {
        if (LanChatProtocol.isBearerCredentialAccepted(authorization, passwordVerifier, apiKeyVerifier)) {
            return true
        }
        if (!authorization.startsWith("Bearer ")) return false
        val token = authorization.removePrefix("Bearer ").trim()
        val expiresAt = sessionTokens[token] ?: return false
        if (expiresAt <= System.currentTimeMillis()) {
            sessionTokens.remove(token)
            return false
        }
        return true
    }

    private fun authorizationError(authorization: String): String {
        if (!authorization.startsWith("Bearer ")) {
            return "Provide Authorization: Bearer PASSWORD_OR_API_KEY. Use the LAN password or the optional generated API key."
        }
        val token = authorization.removePrefix("Bearer ").trim()
        return if (token.startsWith("sk-pocket-")) {
            "Invalid API key. Regenerate or copy the current API key from the phone."
        } else {
            "Invalid LAN password or API key. Use the password set in the app or the optional generated API key."
        }
    }

    private fun parseChatRequest(body: String): ParsedChatRequest {
        return LanChatProtocol.parseChatRequest(body)
    }

    private fun writeModels(writer: BufferedWriter) {
        writeJson(
            writer,
            200,
            JSONObject()
                .put("object", "list")
                .put("data", JSONArray().put(modelObject()))
        )
    }

    private fun writeModel(writer: BufferedWriter) {
        writeJson(writer, 200, modelObject())
    }

    private fun modelObject(): JSONObject {
        return JSONObject()
            .put("id", modelId)
            .put("object", "model")
            .put("created", 0)
            .put("owned_by", "pocket-llm")
    }

    private fun textDelta(previous: String, current: String): String {
        return if (current.startsWith(previous)) current.removePrefix(previous) else current
    }

    private fun writeChatError(writer: BufferedWriter, streaming: Boolean, statusCode: Int, message: String) {
        if (!streaming) {
            writeJson(writer, statusCode, errorJson(message))
            return
        }
        writeSseError(writer, message)
    }

    private fun writeSseHeaders(writer: BufferedWriter) {
        writer.write("HTTP/1.1 200 OK\r\n")
        writer.write("Content-Type: text/event-stream; charset=utf-8\r\n")
        writer.write("Cache-Control: no-cache\r\n")
        writer.write("Connection: close\r\n")
        writer.write("X-Accel-Buffering: no\r\n")
        writer.write("\r\n")
        writer.flush()
    }

    private fun writeSseChunk(
        writer: BufferedWriter,
        completionId: String,
        created: Long,
        delta: JSONObject,
        finishReason: String?
    ) {
        val choice = JSONObject()
            .put("index", 0)
            .put("delta", delta)
            .put("finish_reason", finishReason)
        val chunk = JSONObject()
            .put("id", completionId)
            .put("object", "chat.completion.chunk")
            .put("created", created)
            .put("model", modelId)
            .put("choices", JSONArray().put(choice))
        writer.write("data: ${chunk}\n\n")
        writer.flush()
    }

    private fun writeSseError(writer: BufferedWriter, message: String) {
        writer.write("data: ${errorJson(message)}\n\n")
        writer.write("data: [DONE]\n\n")
        writer.flush()
    }

    private fun writeJson(writer: BufferedWriter, statusCode: Int, body: JSONObject) {
        val bodyBytes = body.toString().toByteArray(StandardCharsets.UTF_8)
        writer.write("HTTP/1.1 $statusCode ${STATUS_REASONS[statusCode] ?: "Error"}\r\n")
        writer.write("Content-Type: application/json; charset=utf-8\r\n")
        writer.write("Content-Length: ${bodyBytes.size}\r\n")
        writer.write("Connection: close\r\n")
        writer.write("Cache-Control: no-store\r\n")
        writer.write("\r\n")
        writer.flush()
        writer.write(String(bodyBytes, StandardCharsets.UTF_8))
        writer.flush()
    }

    private fun writeHtml(writer: BufferedWriter, statusCode: Int, body: String) {
        val bodyBytes = body.toByteArray(StandardCharsets.UTF_8)
        writer.write("HTTP/1.1 $statusCode ${STATUS_REASONS[statusCode] ?: "Error"}\r\n")
        writer.write("Content-Type: text/html; charset=utf-8\r\n")
        writer.write("Content-Length: ${bodyBytes.size}\r\n")
        writer.write("Connection: close\r\n")
        writer.write("Cache-Control: no-store\r\n")
        writer.write("\r\n")
        writer.write(body)
        writer.flush()
    }

    private fun writeBinary(output: OutputStream, statusCode: Int, contentType: String, body: ByteArray) {
        val header = "HTTP/1.1 $statusCode ${STATUS_REASONS[statusCode] ?: "Error"}\r\n" +
            "Content-Type: $contentType\r\n" +
            "Content-Length: ${body.size}\r\n" +
            "Connection: close\r\n" +
            "Cache-Control: no-store\r\n" +
            "\r\n"
        output.write(header.toByteArray(StandardCharsets.UTF_8))
        output.write(body)
        output.flush()
    }

    private fun readLimitedLine(input: InputStream): String? {
        val bytes = ByteArrayOutputStream()
        while (true) {
            val value = input.read()
            if (value == -1) {
                if (bytes.size() == 0) return null
                break
            }
            if (value == '\n'.code) break
            if (value != '\r'.code) bytes.write(value)
            if (bytes.size() > MAX_LINE_LENGTH) {
                throw IOException("HTTP line is too long.")
            }
        }
        val line = bytes.toString(StandardCharsets.US_ASCII.name())
        if (line.length > MAX_LINE_LENGTH) {
            throw IOException("HTTP line is too long.")
        }
        return line
    }

    private fun errorJson(message: String): JSONObject {
        return JSONObject().put("error", JSONObject().put("message", message))
    }

    private fun isAllowedLanAddress(address: InetAddress): Boolean {
        return address.isLoopbackAddress ||
            address.isSiteLocalAddress ||
            address.isLinkLocalAddress
    }

    private fun localLanAddress(): InetAddress {
        val interfaces = Collections.list(NetworkInterface.getNetworkInterfaces())
        val addresses = interfaces.flatMap { networkInterface ->
            val interfacePriority = when {
                networkInterface.name.startsWith("wlan", ignoreCase = true) ||
                    networkInterface.displayName.contains("wifi", ignoreCase = true) -> 0
                networkInterface.name.startsWith("eth", ignoreCase = true) -> 1
                networkInterface.name.startsWith("rmnet", ignoreCase = true) -> 2
                else -> 3
            }
            Collections.list(networkInterface.inetAddresses).map { address ->
                interfacePriority to address
            }
        }
        return addresses
            .asSequence()
            .filter { (_, address) ->
                address is Inet4Address && !address.isLoopbackAddress && address.isSiteLocalAddress
            }
            .sortedWith(compareBy<Pair<Int, InetAddress>> { it.first }
                .thenBy { it.second.hostAddress })
            .map { (_, address) -> address }
            .firstOrNull()
            ?: InetAddress.getLoopbackAddress()
    }
}
