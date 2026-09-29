# 🤖 Pocket LLM for Android (Offline, Private & Fast)

An Android application that brings local LLM chat, voice input, image input, documents, PDF OCR, audio attachments, and camera-based prompting to your phone.

Pocket LLM runs fully on device after model download. It supports ONNX-based Qwen models, LiteRT-based Qwen 3 and Gemma 4 models, streaming responses, persistent local chat history, markdown-rendered replies, downloadable models, in-app model switching, editable model instructions, and multiple image input workflows.

The app ships as a small base APK. Users download only the models they want, switch between them inside the app, and delete unused models later to save device storage.

---

[![Total APK downloads](https://img.shields.io/github/downloads/dineshsoudagar/local-llms-on-android/total?logo=github&label=Total%20APK%20downloads)](https://github.com/dineshsoudagar/local-llms-on-android/releases)

---

## 🆕 New in v1.6.0

Pocket LLM can now use your Android phone as a private local AI server for a computer on the same network.

- 🌐 Added a password-protected LAN server with an OpenAI-compatible API and built-in browser UI
- 💻 Start a chat on your computer while the selected model continues to run on your phone
- 🗂️ Added browser chat history saved in the phone's private storage, with per-chat deletion
- 📎 Added browser uploads for PDFs and images; PDF questions use lightweight on-device retrieval
- 🎨 Refined the browser chat UI with a focused transcript, purple light/dark themes, and keyboard sending
- ⚙️ Added a per-model context-length input with clear high-memory warnings
- 🔌 Added tested OpenAI-compatible connectivity for OpenCode and other compatible clients
- 🛡️ Added safer recovery after interrupted model initialization and clearer high-context memory guidance

#### ➡️ [See all releases](https://github.com/dineshsoudagar/local-llms-on-android/releases)

---

### 🔗 Also Check Out

**[local-document-intelligence](https://github.com/dineshsoudagar/local-document-intelligence)**  
A privacy-first offline document intelligence system with persistent local RAG, hybrid retrieval, and source-grounded answers.

---

## ✨ Features

- 📱 Fully on-device LLM chat for private offline use
- 🎙️ Voice input for faster prompting
- 🖼️ Image input with OCR and Gemma native image support
- 📎 One active document, PDF, or audio attachment per chat, with persistent source references
- 🔊 Native Gemma 4 audio understanding and on-device sherpa-onnx Whisper transcription for other models
- 📷 Camera capture with retake, crop, and photo review
- 💬 Persistent multi-turn chat with local history
- 📦 Download, switch, and delete models inside the app
- 🧠 Supports Qwen2.5, Qwen3, Qwen3 LiteRT, and Gemma 4 LiteRT models
- ⚡ ONNX and LiteRT backend support
- 🎛️ Editable model instructions with presets and custom prompts
- 🎨 Light mode, dark mode, accent colors, and adjustable chat font size
- 🔐 Offline after model download, with no telemetry

---

## 📸 Inference Preview

<table align="center">
  <tr>
    <td align="center">
      <img src="data/Chat.gif" alt="Model Output 1" width="260"/><br/>
      <sub><b>Chat Inference</b></sub>
    </td>
    <td align="center">
      <img src="data/Image support.gif" alt="Model Output 2" width="260"/><br/>
      <sub><b>Image Support</b></sub>
    </td>
    <td align="center">
      <img src="data/New ui.gif" alt="Chat UI Preview" width="260"/><br/>
      <sub><b>New UI</b></sub>
    </td>
  </tr>
</table>

<p align="center">
  <em>Figure: Pocket LLM showing offline chat, image input, and the updated Android UI.</em>
</p>

---

## 📦 Download APK - v1.6.0

The app ships as a **single smaller base APK**.

#### ➡️ [Download APK](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.6.0/pocket_llm_v1.6.0.apk)

Models are **not bundled inside the APK**. After installation, choose and download the models you want directly on device.

You can download **multiple models**, switch between them inside the app, and delete unused downloaded models later to free storage.

### Available chat models

- **Gemma 4 E4B LiteRT** - Best for **flagship mobiles**
- **Gemma 4 E2B LiteRT** - Best for **decent to mid-range mobiles**
- **Qwen3 0.6B LiteRT** - Best for **low-end mobiles**
- **Qwen3 0.6B Q4F16 ONNX** - Good for **low to mid-range mobiles**
- **Qwen2.5 0.5B ONNX** - Best for **mid to high-end mobiles**, **full precision**

### Image input support

- **OCR mode** - Extract text from images
- **Gemma native image mode** - Send images directly to supported Gemma models
- **Camera capture** - Take a photo, retake, crop, review, and send it as input

> Note: internet is required only for downloading models. Chat, OCR, image input, camera workflows, and inference remain fully on-device after the required models are installed.

## Document and audio attachments

The paperclip menu accepts safe UTF-8/UTF-16 text files, embedded-text or scanned PDFs, Android-decodable audio files, and microphone recordings. Text and PDFs are limited to 16 MiB and 64 MiB/500 pages respectively. Audio is limited to 512 MiB/two hours and is normalized to private mono 16-kHz PCM WAV. Document/audio and image inputs cannot be mixed in one send in the first version.

PDFBox extracts embedded text page by page; only pages without enough embedded text are rendered and passed through the existing on-device ML Kit OCR path. Questions use budget-fitting BM25 retrieval, while summaries and ordered transformations process all source chunks. Answers are prompted to preserve page, section, or timestamp markers.

Gemma 4 E2B/E4B uses the LiteRT-LM native audio backend directly. Inputs longer than 30 seconds are processed as overlapping 28-second native-audio segments; a failed audio-backend initialization disables Gemma audio and never falls back silently to Whisper. Other chat models use sherpa-onnx 1.13.4 with multilingual Whisper tiny int8 after a separate, explicit roughly 100 MB download confirmation. They do not switch to Gemma.

Attachment manifests, normalized sources, extracted text/transcripts, chunks, and indexes live beneath the owning private chat directory. Detach keeps that data and deleting the chat removes it. Long extraction and transcription run as cancellable foreground WorkManager jobs with visible progress; inference remains local, and raw attachment payloads are excluded from ordinary saved model history.

---

## LAN API and Web UI

Pocket LLM can expose the selected on-device model to a computer or another device on the same private network. Start the server in the Android app, then open the displayed `/ui` address on your computer to chat in a browser while inference remains on the phone. The same server also exposes an OpenAI-compatible API for scripts and compatible tools.

1. Load a model and open the navigation drawer.
2. Choose **LAN Server**, set a Web UI password of at least eight characters, then tap **Start**. You can use **Generate** for a random password.
3. The **Web UI password** is the simplest credential and works for both browser chat and API clients. A generated API key is optional for advanced clients.
4. From another device on the same network, use the phone's displayed URL with `/v1`, for example `http://PHONE_IP:8080/v1`.

The phone dialog shows the address chosen from the phone's local network interfaces. The computer must be on that same Wi-Fi or LAN; a VPN, guest network, mobile-data address, or a different interface can show a different IP and will not work. This is a private-LAN endpoint, not a public Internet URL.

### LAN screenshots

Add the server-start screenshot here:

<!-- ![Start the LAN server from the Android app](data/lan-server-start.png) -->

Add the browser-chat screenshot here:

<!-- ![Pocket LLM browser chat UI](data/lan-browser-ui.png) -->

### Browser Web UI

The `/ui` page provides saved browser conversations, a left-side chat history, light and dark purple themes, and Enter-to-send with Shift+Enter for a new line. Chat history is stored in the phone's private app storage. The selected model remains controlled in the Android app.

Use the attachment button to upload a PDF or image with a message:

- **PDFs:** Pocket LLM extracts embedded text, uses OCR only for pages that need it, then selects relevant chunks for question answering. This is lightweight retrieval for specific questions, not a full long-document research workflow. Retrieval and document workflows will continue to improve in future releases.
- **Images:** Direct image input is available only when the model currently selected in the Android app supports it.

### OpenAI-compatible endpoints

The API uses `Authorization: Bearer PASSWORD_OR_API_KEY` and exposes:

- `GET /v1/models` - returns the currently loaded model id.
- `GET /v1/models/MODEL_ID` - returns metadata for that model.
- `POST /v1/chat/completions` - accepts standard `messages`, `model`, `stream`, and `max_tokens` fields; `max_tokens` currently reserves context budget rather than hard-capping decoder output.

Example from PowerShell:

```powershell
$base = "http://PHONE_IP:8080/v1"
$password = "YOUR_WEB_UI_PASSWORD"

Invoke-RestMethod "$base/models" -Headers @{ Authorization = "Bearer $password" }

Invoke-RestMethod "$base/chat/completions" -Method Post `
  -Headers @{ Authorization = "Bearer $password" } `
  -ContentType "application/json" `
  -Body (@{
    model = "MODEL_ID_FROM_MODELS"
    messages = @(@{ role = "user"; content = "Explain photosynthesis in one sentence." })
  } | ConvertTo-Json -Depth 5)
```

### Python client

Install the standard client library and run the included example:

```powershell
python -m pip install openai
$env:POCKET_LLM_BASE_URL = "http://PHONE_IP:8080/v1"
$env:POCKET_LLM_PASSWORD = "YOUR_WEB_UI_PASSWORD"
python scripts/pocket_llm_openai.py "Give me three German words for travel."
python scripts/pocket_llm_openai.py --stream "Write a short greeting."
```

The script discovers the model id automatically. It also accepts `--model`, `--system`, `--base-url`, `--password`, and optional `--api-key` for automation environments.

For a small HTTP API example without installing a client library, use text alone or attach an image or PDF:

```powershell
$env:POCKET_LLM_BASE_URL = "http://PHONE_IP:8080/v1"
$env:POCKET_LLM_PASSWORD = "YOUR_WEB_UI_PASSWORD"
python scripts/pocket_llm_http.py "Give me three German words for travel."
python scripts/pocket_llm_http.py --image "C:\path\photo.jpg" "What is in this picture?"
python scripts/pocket_llm_http.py --pdf "C:\path\report.pdf" "Summarize the main findings."
```

The script selects the loaded model from `/v1/models`, sends the prompt to `/v1/chat/completions`, and prints the answer. For image or PDF input, it uploads up to four files through `/ui/attachments` with a unique temporary chat ID, then deletes that chat and its attachments after the request. Image input requires a model with direct image support. The original files stay at the paths you supplied. The server writes uploads to the phone's private storage temporarily while processing them; this example removes them afterward, including imported PDF data. If cleanup fails, the script reports it. The browser UI does not provide a permanent image preview.

### OpenCode coding-agent backend

OpenCode connectivity has been tested against the OpenAI-compatible LiteRT endpoint. Copy [`examples/opencode-pocket-llm.jsonc`](examples/opencode-pocket-llm.jsonc) into the OpenCode project configuration, replace `PHONE_IP`, and set the phone's LAN password in `POCKET_LLM_PASSWORD`. The example uses the actual model id `qwen3_litert`; if `GET /v1/models` returns a different selected model id, replace both occurrences in the example.

```powershell
$env:POCKET_LLM_PASSWORD = "YOUR_WEB_UI_PASSWORD"
opencode
```

OpenCode owns tool execution: Pocket LLM returns native LiteRT tool calls with OpenAI-compatible ids and arguments, then accepts the follow-up `role: "tool"` message. ONNX-backed models return an explicit unsupported-backend error for tool requests; `/v1/responses`, Codex integration, and native Ollama routes are not part of this phase.

> OpenCode can connect, but coding-agent workloads often need more context than the validated mobile range. Treat this as tested compatibility, not a recommended long-context coding setup.

For Open WebUI or another OpenAI-compatible client, add a connection with base URL `http://PHONE_IP:8080/v1`, enter the same LAN password as the credential, and select the model id returned by `GET /v1/models`.

Compatible tools can use the password directly:

```bash
curl -X POST http://PHONE_IP:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_WEB_UI_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"model":"MODEL_ID_FROM_MODELS","messages":[{"role":"user","content":"Explain photosynthesis in one sentence."}]}'
```

The password is stored on the phone as a salted hash and is accepted as a Bearer credential for the API. The generated API key is optional and can be regenerated when needed. Stop the LAN server before changing the password or model; leaving the password field blank keeps the current password. Android may still stop background work because of device power-management policy, so this is intended for local-network use rather than unattended public hosting.

### Context-length setting

LiteRT models expose a context-length field in the Android app's model settings. This is a device-memory trade-off: a larger context can require substantially more native and GPU memory.

- **Recommended tested baseline:** 8K tokens.
- **Limited manual testing:** 10K to 15K worked on the tested device, but this is not broad device validation.
- **Configurable maximum:** Qwen LiteRT models allow up to 40K and Gemma LiteRT models up to 128K. Those values are configuration ceilings, not guarantees that a phone can initialize or run safely at that size.

The app warns above 8K. A native LiteRT-LM crash was observed with Gemma 4 E2B at 20K on a tested device, consistent with high-context memory pressure. Keep the context near 8K unless you have tested the selected model on your own device.

## 🧠 Backend Support

This app supports **ONNX-based Qwen models** and **LiteRT-based Qwen 3 and Gemma 4 models**.

### Backend overview

- **ONNX backend**: supports **Qwen2.5** and **Qwen3**
- **LiteRT backend**: supports **Qwen3** and **Gemma 4**

### Thinking Mode

- **Qwen3** and **Gemma 4** support **Thinking Mode**
- The toggle is shown only for models that support it

---

## 🚀 Why LiteRT

**LiteRT** is a strong fit for fast local Android chat because:

- It is designed for **high-performance on-device LLM deployment**
- It supports **hardware acceleration**, including **GPU and NPU acceleration** on supported devices
- It helps reduce startup and generation latency for local chat workloads
- It expands the range of practical Android model builds beyond a single backend path
- It fits well with a privacy-first app design focused on fully offline usage

> Note: model capability and performance still depend on the specific model build and the hardware of the target Android device.

---

## ⚙️ Requirements

- [Android Studio](https://developer.android.com/studio)
- A physical Android device for deployment and testing
- 4 GB or more RAM for smaller models
- More RAM is recommended for larger models such as **Gemma 4 E2B** and **Gemma 4 E4B**
- A temporary internet connection for downloading models inside the app
- Real hardware is preferred; emulators are mainly useful for UI checks

### Safe model loading and recovery

Before a model is loaded, Pocket LLM checks available storage and validates the downloaded files. Memory estimates are advisory only: they do not block loading or require confirmation because device memory reports cannot reliably predict LiteRT backend compatibility. The app automatically attempts initialization with the available backend fallbacks.

The app records when initialization starts and clears that record only after the model is ready. If initialization fails or the app stops during loading, the same model is not retried automatically on the next launch. A recovery prompt lets you choose another model, delete the failed files, retry manually, or continue without a loaded model. Saved chat history is kept independently and is not deleted by model-load recovery.

---

## 🚀 How to Build & Run

1. Clone this repository.
2. Install the latest **Android Studio**.
3. Open the Android project folder in Android Studio:

    ```text
    pocket_llm_src/
    ```
4. Build and install the app on your Android device.
5. Launch the app.
6. On first launch, choose a model from the built-in model picker.
7. Download the selected model directly inside the app.
8. Start chatting locally on device

---

## 📄 License Notice

### Gemma 4

Gemma 4 is provided by Google under the **Apache License 2.0**. Google's Gemma documentation also states that Gemma models are provided with open weights and support responsible commercial use.

- Gemma 4 license: https://ai.google.dev/gemma/apache_2
- Gemma 4 overview: https://ai.google.dev/gemma/docs/core

### Qwen models

Qwen model files follow the upstream Qwen license terms.  
Please review the original model license before redistribution or commercial use.
