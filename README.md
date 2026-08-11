# 🤖 Pocket LLM for Android (Offline, Private & Fast)

An Android application that brings local LLM chat, voice input, image input, documents, PDF OCR, audio attachments, and camera-based prompting to your phone.

Pocket LLM runs fully on device after model download. It supports ONNX-based Qwen models, LiteRT-based Qwen 3 and Gemma 4 models, streaming responses, persistent local chat history, markdown-rendered replies, downloadable models, in-app model switching, editable model instructions, and multiple image input workflows.

The app ships as a small base APK. Users download only the models they want, switch between them inside the app, and delete unused models later to save device storage.

---

[![Total APK downloads](https://img.shields.io/github/downloads/dineshsoudagar/local-llms-on-android/total?logo=github&label=Total%20APK%20downloads)](https://github.com/dineshsoudagar/local-llms-on-android/releases)

---

## 🆕 New in v1.5.0

Pocket LLM now supports richer local input workflows beyond text chat.

- 🎙️ Added voice input for faster prompting
- 🖼️ Added image input with OCR and Gemma direct image input
- 📷 Added camera capture with retake, crop, and photo review
- 🗂️ Added a side panel for quick access to previous chats
- 🗑️ Added easier chat deletion from the history panel
- 💾 Added downloaded model deletion to free device storage
- ⚙️ Added editable model instructions with presets and custom prompts
- 🎨 Added dark mode, light mode, accent colors, and chat font-size control
- 📋 Added copy button for assistant responses

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

## 📦 Download APK - v1.5.0

The app ships as a **single smaller base APK**.

#### ➡️ [Download APK](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.5.0/pocket_llm_v1.5.0.apk)

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
