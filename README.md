# 🤖 Pocket LLM for Android (Offline, Private & Fast)

Run private AI chat on your Android phone, with text, images, PDFs, audio, and offline speech-to-text.

Chat in the app, open its Web UI on your computer, or connect a compatible OpenAI client while the model runs on your phone.

Install the small base APK, then download a built-in model or import your own LiteRT model (beta).

---

[![Total APK downloads](https://img.shields.io/github/downloads/dineshsoudagar/local-llms-on-android/total?logo=github&label=Total%20APK%20downloads)](https://github.com/dineshsoudagar/local-llms-on-android/releases)

---

## 🆕 New in v1.6.0

Changes since v1.5.0:

### New features

- 🧩 **[Your own model (beta)](#use-your-own-model-beta):** Import a local `.litertlm` file for text chat.
- 📄 **[Documents](#document-and-audio-attachments):** Attach text files and PDFs, including scanned pages read with OCR.
- 🔎 **[Document answers](#document-and-audio-attachments):** Use simple BM25 retrieval for questions or process all source chunks for summaries.
- 🎙️ **[Speech-to-text](#document-and-audio-attachments):** Transcribe offline with multilingual Whisper, including German, and edit before sending.
- 🔊 **[Audio attachments](#document-and-audio-attachments):** Upload or record audio for native Gemma understanding or Whisper transcription.
- 🌐 **[LAN and browser chat](#lan-api-and-web-ui):** Open the phone's password-protected Web UI on your PC while inference stays on the phone.
- 🗂️ **[Browser history and uploads](#browser-web-ui):** Save, reopen, or delete chats on the phone, and upload PDFs or images from your PC.
- 🔌 **[OpenAI-compatible API](#openai-compatible-endpoints):** Connect Open WebUI, [example scripts](#python-client), or compatible harnesses; [OpenCode](#opencode-coding-agent-backend) tool-call round trips tested.
- ⚙️ **[Context settings](#context-length-setting):** Adjust context length per LiteRT model.

### Improvements

- ⏳ **Long inputs:** Added segmented audio processing and cancellable attachment jobs with progress.
- 🎨 **Browser controls:** Added light/dark themes, Enter to send, and Shift+Enter for a new line.
- ⚡ **Runtime:** Updated and pinned LiteRT-LM to **0.17.1**.

### Fixes and cleanup

- 🛡️ **Model loading:** Improved file validation, downloads, cancellation, and recovery after failed initialization.
- 🖼️ **Image input:** Removed FastVLM descriptions; images now use OCR or native Gemma vision.

### Limitations and next steps

- 🧩 **Custom models:** Beta compatibility varies by model and phone; native tool calling is unavailable.
- ⚠️ **Memory:** Large models and contexts can still fail to load or crash, including during long OpenCode sessions.
- 🖼️ **Browser images:** Direct image uploads require a compatible model.
- 🚀 **Coming next:** Improved document retrieval; this release uses lightweight BM25.

#### ➡️ [See all releases](https://github.com/dineshsoudagar/local-llms-on-android/releases)

---

### 🔗 Also Check Out

**[local-document-intelligence](https://github.com/dineshsoudagar/local-document-intelligence)**  
A privacy-first offline document intelligence system with persistent local RAG, hybrid retrieval, and source-grounded answers.

---

## ✨ Features

### Chat and privacy

- 📱 **Local inference:** Run models on your phone with ONNX or LiteRT, offline after installation.
- 💬 **Chat:** Stream replies, keep multi-turn history, render Markdown, and copy responses.
- 🗂️ **History:** Reopen or delete saved conversations in the Android app or browser.
- 🎨 **Appearance:** Choose light/dark mode, accent colors, and chat font size in the Android app.
- 🔐 **Privacy:** Keep inference and chat data on your phone, with no telemetry.

### Models and settings

- 🧠 **Built-in models:** Choose Qwen2.5, Qwen3, DeepSeek R1 Distill Qwen, or Gemma 4.
- 📦 **Model management:** Download, switch, and delete models inside the app.
- 🧩 **Custom models (beta):** Import your own `.litertlm` text model from device storage.
- 💭 **Thinking mode:** Toggle reasoning on supported Qwen3 and Gemma models.
- 🎛️ **Instructions:** Edit model instructions or choose a prompt preset.
- ⚙️ **Context length:** Set each LiteRT model's context size to balance history and memory use.
- 🛡️ **Recovery:** Validate model files and recover from interrupted loading without losing saved chats.

### Files and voice

- 🎙️ **Dictation:** Turn recordings into editable text with offline multilingual Whisper, including German.
- 🔊 **Audio:** Use native Gemma audio understanding or Whisper transcription for other models.
- 📄 **Documents:** Attach text files and PDFs, including scanned PDFs read with OCR.
- 🔎 **Retrieval:** Ask document questions with BM25 or summarize source chunks within the context budget.
- 📎 **References:** Keep document, page, or timestamp references with the owning chat.
- ⏳ **Progress:** Track or cancel long document extraction and audio transcription jobs.
- 🖼️ **Images:** Extract text with OCR or send images directly to supported Gemma models.
- 📷 **Camera:** Capture, retake, crop, review, and send photos.

### LAN and integrations

- 🌐 **LAN server:** Serve the phone's selected model to devices on the same private network.
- 💻 **Web UI:** Chat in a PC browser, upload PDFs or images, and save conversations on the phone.
- 🔌 **OpenAI-compatible API:** Connect compatible clients through model and chat-completion endpoints.
- 🛠️ **Agent tools:** Return native tool calls from supported LiteRT models for harnesses such as OpenCode to execute.

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
- **DeepSeek R1 Distill Qwen 1.5B LiteRT** - Optional reasoning model for **high-RAM mobiles**
- **Qwen3 0.6B LiteRT** - Best for **low-end mobiles**
- **Qwen3 0.6B Q4F16 ONNX** - Good for **low to mid-range mobiles**
- **Qwen2.5 0.5B ONNX** - Best for **mid to high-end mobiles**, **full precision**

### Use your own model (beta)

- Open **Manage Models → use your own model (beta)** and choose a local `.litertlm` file.
- The app copies the file into private storage and adds it to the model picker.
- Custom models support text chat; image input uses OCR and audio input uses Whisper transcription.
- Custom models do not expose native tool calling through the LAN API.
- Custom-model compatibility is experimental; larger models or contexts may fail to load or crash.

### Image input support

- **OCR mode** - Extract text from images
- **Gemma native image mode** - Send images directly to supported Gemma models
- **Camera capture** - Take a photo, retake, crop, review, and send it as input

> Note: internet is required only for downloading models. Chat, OCR, image input, camera workflows, and inference remain fully on-device after the required models are installed.

## Document and audio attachments

- **Text:** Attach UTF-8 or UTF-16 text files up to 16 MiB.
- **PDFs:** Attach embedded-text or scanned PDFs up to 64 MiB and 500 pages in the Android app.
- **PDF OCR:** Extract embedded text first and use on-device OCR only for pages that need it.
- **Questions:** Use lightweight BM25 to select relevant passages that fit the model context.
- **Summaries:** Process all source chunks for summaries and ordered transformations.
- **References:** Prompt answers to retain page, section, or timestamp markers.
- **Audio files:** Attach Android-decodable audio up to 512 MiB and two hours, or record audio in-app.
- **Gemma audio:** Use native audio understanding with overlapping segments for longer recordings.
- **Other models:** Transcribe audio with multilingual Whisper tiny int8 after a separate download of about 100 MB.
- **Dictation:** Tap the microphone, record, transcribe, and edit the recognized text before sending.
- **Attachment limits:** Keep one active document or audio source per Android chat; do not mix it with images in one send.
- **Storage:** Detaching keeps the source data; deleting its chat removes it from private app storage.
- **Progress:** Long extraction and transcription jobs show progress and can be cancelled.

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
- **LiteRT backend**: supports **Qwen3**, **DeepSeek R1 Distill Qwen**, **Gemma 4**, and imported `.litertlm` text models (beta)

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
