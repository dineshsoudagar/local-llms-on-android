package com.example.local_llm

import android.content.Context
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.net.Inet4Address
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
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
    companion object {
        const val DEFAULT_PORT = 8080
        private const val UI_PATH = "/ui"
        private const val MAX_LINE_LENGTH = 16 * 1024
        private const val MAX_BODY_BYTES = 1 * 1024 * 1024
        private const val REQUEST_TIMEOUT_MS = 5 * 60 * 1000
        private const val SESSION_TTL_MS = 12 * 60 * 60 * 1000L

        private val WEB_UI_HTML = """
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>Pocket LLM</title>
              <style>
                :root { color-scheme: light; font-family: system-ui, -apple-system, sans-serif; }
                * { box-sizing: border-box; }
                body { margin: 0; background: #f5f2f8; color: #27222f; }
                main { width: min(900px, calc(100% - 32px)); min-height: 100svh; margin: 0 auto; padding: 24px 0 32px; }
                h1 { margin: 48px 0 8px; font-size: clamp(28px, 5vw, 42px); letter-spacing: -.03em; }
                p { color: #6b6475; line-height: 1.5; }
                label { display: block; margin: 24px 0 8px; color: #42384f; font-weight: 800; }
                input, textarea { width: 100%; border: 1px solid #d9d0e2; border-radius: 14px; padding: 13px 15px; background: #fff; color: inherit; font: inherit; outline: none; }
                input:focus, textarea:focus { border-color: #7953d5; box-shadow: 0 0 0 3px #7953d526; }
                textarea { min-height: 56px; resize: vertical; }
                button { border: 0; border-radius: 999px; padding: 12px 22px; background: #7650d2; color: #fff; font: inherit; font-weight: 800; cursor: pointer; transition: transform .15s ease, opacity .15s ease; }
                button:hover { transform: translateY(-1px); }
                button:disabled { opacity: .55; cursor: wait; transform: none; }
                .brand { display: flex; align-items: center; gap: 12px; }
                .brand img { width: 48px; height: 48px; border-radius: 14px; object-fit: cover; }
                .brand strong, .brand span { display: block; }
                .brand strong { color: #5c35bd; font-size: 18px; }
                .brand span { margin-top: 2px; color: #81788f; font-size: 12px; }
                .login-view { width: min(520px, 100%); margin: 9vh auto 0; }
                .login-view button { margin-top: 20px; }
                .error { min-height: 24px; margin-top: 12px; color: #b32f45; }
                .topbar { display: flex; align-items: center; justify-content: space-between; padding-bottom: 18px; border-bottom: 1px solid #ddd4e5; }
                .quiet { padding: 9px 15px; background: transparent; color: #6b528e; border: 1px solid #d5c9df; }
                .messages { display: flex; flex-direction: column; gap: 14px; min-height: 52vh; padding: 28px 0; }
                .message { max-width: min(78%, 680px); padding: 14px 16px; white-space: pre-wrap; line-height: 1.5; border-radius: 18px; animation: rise .18s ease-out; }
                .message.user { align-self: flex-end; background: #7650d2; color: #fff; border-bottom-right-radius: 5px; }
                .message.assistant { align-self: flex-start; background: #fff; color: #312a3b; border: 1px solid #e1d9e8; border-bottom-left-radius: 5px; }
                .composer { display: flex; align-items: flex-end; gap: 10px; padding: 12px; background: #fff; border: 1px solid #d9d0e2; border-radius: 18px; box-shadow: 0 8px 24px #493b5a12; }
                .composer textarea { min-height: 44px; max-height: 180px; padding: 10px 4px; border: 0; box-shadow: none; resize: none; }
                .composer textarea:focus { box-shadow: none; }
                .composer button { flex: 0 0 auto; }
                .hint { margin-top: 12px; font-size: 13px; }
                [hidden] { display: none !important; }
                @keyframes rise { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
                @media (max-width: 560px) { main { width: min(100% - 24px, 900px); padding-top: 16px; } .message { max-width: 88%; } .topbar .brand img { width: 40px; height: 40px; } .topbar .brand strong { font-size: 16px; } }
              </style>
            </head>
            <body>
              <main>
                <section id="loginView" class="login-view">
                  <div class="brand"><img src="/logo.webp" alt=""><div><strong>Pocket LLM</strong><span>Private local server</span></div></div>
                  <h1>Sign in to Pocket LLM</h1>
                  <p>Enter the password set on the phone. The model stays on the phone and this browser only sends text over your private network.</p>
                  <label for="password">Server password</label>
                  <input id="password" type="password" autocomplete="current-password" placeholder="Enter your password">
                  <button id="login" type="button">Log in</button>
                  <div id="loginError" class="error" role="alert"></div>
                </section>
                <section id="chatView" hidden>
                  <header class="topbar">
                    <div class="brand"><img src="/logo.webp" alt=""><div><strong>Pocket LLM</strong><span>Private local server</span></div></div>
                    <button id="logout" class="quiet" type="button">Log out</button>
                  </header>
                  <div id="messages" class="messages" aria-live="polite"></div>
                  <form id="composer" class="composer">
                    <textarea id="prompt" rows="1" placeholder="Ask the local model something..."></textarea>
                    <button id="send" type="submit">Send</button>
                  </form>
                  <p class="hint">Text chat only for now. The selected model and its settings remain on the phone.</p>
                </section>
              </main>
              <script>
                const loginView = document.getElementById('loginView');
                const chatView = document.getElementById('chatView');
                const password = document.getElementById('password');
                const login = document.getElementById('login');
                const loginError = document.getElementById('loginError');
                const messages = document.getElementById('messages');
                const prompt = document.getElementById('prompt');
                const send = document.getElementById('send');
                const logout = document.getElementById('logout');
                const composer = document.getElementById('composer');
                let sessionToken = sessionStorage.getItem('pocket_llm_session');
                function showChat() {
                  loginView.hidden = true;
                  chatView.hidden = false;
                  if (!messages.children.length) addMessage('assistant', 'Connected. What would you like to ask?');
                  prompt.focus();
                }
                function addMessage(role, text) {
                  const item = document.createElement('div');
                  item.className = 'message ' + role;
                  item.textContent = text;
                  messages.appendChild(item);
                  messages.scrollTop = messages.scrollHeight;
                  return item;
                }
                async function loginToServer() {
                  const value = password.value.trim();
                  if (!value) { loginError.textContent = 'Enter the server password.'; return; }
                  login.disabled = true;
                  loginError.textContent = '';
                  try {
                    const result = await fetch('/auth/login', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ password: value })
                    });
                    const payload = await result.json();
                    if (!result.ok) throw new Error(payload.error && payload.error.message ? payload.error.message : 'Login failed.');
                    sessionToken = payload.token;
                    sessionStorage.setItem('pocket_llm_session', sessionToken);
                    showChat();
                  } catch (error) {
                    loginError.textContent = error.message || 'Login failed.';
                  } finally {
                    login.disabled = false;
                  }
                }
                login.addEventListener('click', loginToServer);
                password.addEventListener('keydown', event => { if (event.key === 'Enter') loginToServer(); });
                composer.addEventListener('submit', async event => {
                  event.preventDefault();
                  const text = prompt.value.trim();
                  if (!text || !sessionToken) return;
                  addMessage('user', text);
                  prompt.value = '';
                  const pending = addMessage('assistant', 'Thinking...');
                  send.disabled = true;
                  try {
                    const result = await fetch('/v1/chat/completions', {
                      method: 'POST',
                      headers: { 'Authorization': 'Bearer ' + sessionToken, 'Content-Type': 'application/json' },
                      body: JSON.stringify({ prompt: text })
                    });
                    const payload = await result.json();
                    if (result.status === 401) {
                      sessionStorage.removeItem('pocket_llm_session');
                      sessionToken = null;
                      loginError.textContent = 'Your session expired. Log in again.';
                      loginView.hidden = false;
                      chatView.hidden = true;
                      return;
                    }
                    if (!result.ok) throw new Error(payload.error && payload.error.message ? payload.error.message : 'Request failed.');
                    pending.textContent = payload.choices && payload.choices[0] && payload.choices[0].message
                      ? payload.choices[0].message.content
                      : 'The server returned no assistant message.';
                  } catch (error) {
                    pending.textContent = error.message || 'The request failed.';
                  } finally { send.disabled = false; }
                });
                logout.addEventListener('click', () => {
                  sessionStorage.removeItem('pocket_llm_session');
                  sessionToken = null;
                  messages.textContent = '';
                  chatView.hidden = true;
                  loginView.hidden = false;
                  password.value = '';
                  password.focus();
                });
                if (sessionToken) showChat();
              </script>
            </body>
            </html>
        """.trimIndent()

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
    private val logoBytes: ByteArray = runCatching {
        context.resources.openRawResource(R.mipmap.ic_launcher_2_foreground).use { it.readBytes() }
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
                writeHtml(writer, 200, WEB_UI_HTML)
                return
            }
            if (requestParts[0] == "GET" && path == "/logo.webp") {
                writeBinary(client.getOutputStream(), 200, "image/webp", logoBytes)
                return
            }

            val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
            if (contentLength < 0 || contentLength > MAX_BODY_BYTES) {
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
                        history = request.history,
                        systemInstruction = request.systemInstruction,
                        outputTokenReserve = request.outputTokenReserve,
                        tools = request.toolsForInference(),
                        toolChoice = request.toolChoice,
                        parallelToolCalls = request.parallelToolCalls,
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
                        history = request.history,
                        systemInstruction = request.systemInstruction,
                        outputTokenReserve = request.outputTokenReserve,
                        tools = request.toolsForInference(),
                        toolChoice = request.toolChoice,
                        parallelToolCalls = request.parallelToolCalls
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
        }
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
