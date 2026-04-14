# 🤖 Local LLMs on Android (Offline, Private & Fast)

An Android application that brings local LLM chat to your phone, fully offline, private, and fast.

It supports **ONNX-based Qwen models** and **LiteRT-based Qwen 3 and Gemma 4 models**, with streaming responses, persistent local chat history, manual reopening of previous chats, markdown-rendered replies, and a cleaner chat-first UI.

---

[![Total APK downloads](https://img.shields.io/github/downloads/dineshsoudagar/local-llms-on-android/total?logo=github&label=Total%20APK%20downloads)](https://github.com/dineshsoudagar/local-llms-on-android/releases)

---

## 🆕 New in v1.3.0

- Added support for **Gemma 4 E2B**, **Gemma 4 E4B**, and **Qwen3-0.6B** with the **LiteRT backend**
- Added **persistent local chat history** with automatic on-device saving
- Added **Previous Chats** to reopen and continue saved sessions
- Added **Thinking Mode** for supported models
- Improved **markdown rendering** for assistant responses
- Added built-in **themes** and **chat font size** settings
- Refined the overall **chat UI** and usability

---

### 🔗 Also Check Out

**[local-document-intelligence](https://github.com/dineshsoudagar/local-document-intelligence)**  
A privacy-first offline document intelligence system with persistent local RAG, hybrid retrieval, and source-grounded answers.

---

## ✨ Features

- 📱 Fully on-device LLM inference for private offline use
- 🧠 Supports **Qwen2.5**, **Qwen3**, **Qwen3 LiteRT**, **Gemma 4 E2B**, and **Gemma 4 E4B**
- ⚡ Supports **ONNX** and **LiteRT** backends
- 🚀 Hardware-accelerated **LiteRT** inference on supported devices
- 🔤 Hugging Face-compatible tokenizer support for ONNX Qwen models
- 🧩 Configurable prompts, precision, KV cache, and backend setup
- 🧘‍♂️ **Thinking Mode** for **Qwen3** and **Gemma 4**
- 💬 Persistent multi-turn chat with local history and Previous Chats
- 📝 Markdown rendering for assistant replies, including table support
- 🎨 Built-in themes and adjustable chat font size
- 🛑 Stop-generation support
- 🔐 100% offline, with no network calls or telemetry

---

## 📸 Inference Preview

<table align="center">
  <tr>
    <td align="center">
      <img src="data/Gemma4chat.gif" alt="Model Output 1" width="260"/><br/>
      <sub><b>Chat inference</b></sub>
    </td>
    <td align="center">
      <img src="data/Themes.gif" alt="Model Output 2" width="260"/><br/>
      <sub><b>Theme switching</b></sub>
    </td>
    <td align="center">
      <img src="data/Gemma4 thinking.gif" alt="Chat UI Preview" width="270"/><br/>
      <sub><b>Thinking mode</b></sub>
    </td>
  </tr>
</table>

<p align="center">
  <em>Figure: App interface showing local LLM chat and streaming responses on Android.</em>
</p>

---

## 📦 Download Prebuilt APKs - V1.3.0

- 🚀 **Gemma 4 E4B LiteRT** - Best for **flagship mobiles**  
  [Download APK](https://huggingface.co/dineshdroid/local-llms-on-android-apks/resolve/main/pocket_llm_gemma4_e4b_litertlm_v1.3.0.apk) - `3.28 GB`

- ⚖️ **Gemma 4 E2B LiteRT** - Best for **decent to mid-range mobiles**  
  [Download APK](https://huggingface.co/dineshdroid/local-llms-on-android-apks/resolve/main/pocket_llm_gemma4_e2b_litertlm_v1.3.0.apk) - `2.37 GB`

- 📱 **Qwen3 0.6B LiteRT** - Best for **low-end mobiles**  
  [Download APK](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/1.3.0/pocket_llm_qwen3_0.6b_litertlm_v1.3.0.apk) - `654 MB`

- ⚡ **Qwen3 0.6B Q4FP16 ONNX** - Good for **low to mid-range mobiles**  
  [Download APK](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/1.3.0/pocket_llm_qwen3_0.6b_q4fp16_onnx_v1.3.0.apk) - `1 GB`

- 🧠 **Qwen2.5 0.5B ONNX** - Best for **mid to high-end mobiles**, **full precision**  
  [Download APK](https://huggingface.co/dineshdroid/local-llms-on-android-apks/resolve/main/pocket_llm_qwen2.5_0.5b_onnx_v1.3.0.apk) - `2.44 GB`

## 🧠 Model Support

This app supports **ONNX-based Qwen models** and **LiteRT-based Qwen 3 and Gemma 4 models**.

### Supported models

- **Qwen2.5-0.5B**
- **Qwen3-0.6B**
- **Gemma 4 E2B** 
- **Gemma 4 E4B**

### Backend overview

- **ONNX backend**: supports Qwen2.5 and Qwen3
- **LiteRT backend**: supports Qwen3 and Gemma 4

### ONNX model files

For ONNX builds, the app expects:

- `model.onnx`
- `tokenizer.json`

### LiteRT model files

For LiteRT builds, the app expects the matching `.litertlm` model file for the selected model.

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

This release uses LiteRT to broaden the app's supported local model lineup while keeping the experience fully on-device.

> Note: model capability and performance still depend on the specific model build and the hardware of the target Android device.

---

## ⚙️ Requirements

- [Android Studio](https://developer.android.com/studio)
- [ONNX Runtime for Android](https://github.com/microsoft/onnxruntime) for ONNX Qwen builds
- LiteRT dependencies for LiteRT Qwen and Gemma builds
- A physical Android device for deployment and testing
- 4 GB or more RAM for FP16 / Q4 models
- 6 GB or more RAM for FP32 models
- Real hardware is preferred; emulators are mainly useful for UI checks

---

## 🔁 Choose Which Model to Build With

The active model is selected in:

`Qwen_chat_style_app/app/src/main/java/com/example/local_llm/ModelDescriptor.kt`

Inside `ModelRegistry`, change:

```kotlin
private const val SELECTED_MODEL_ID = "qwen2_5"
```

to one of:

```kotlin
"qwen2_5"      // Qwen2.5
"qwen3"        // Qwen3
"qwen3_litert" // Qwen3 LiteRT
"gemma4_e2b"   // Gemma4-2B
"gemma4_e4b"   // Gemma4-4B
```

### Notes

- The app title uses the selected model display name
- **Thinking Mode** is available for **Qwen3**
- **Gemma 4** is displayed as **Gemma4** in the UI

---

## 🔁 Get or Convert Models

### Option 1: Use pre-converted ONNX Qwen models

Download the ONNX model files from Hugging Face:

- [Qwen2.5](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)
- [Qwen3](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX)

### Option 2: Use LiteRT model files

Use the LiteRT model file that matches the model you select in `ModelDescriptor.kt`:

- [gemma-4-E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main)
- [gemma-4-E2B](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/tree/main)
- [qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B/tree/main)

### Option 3: Convert a Qwen model to ONNX yourself

```bash
pip install optimum[onnxruntime]
# or
python -m pip install git+https://github.com/huggingface/optimum.git
```

Export the model:

```bash
optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct qwen2.5-0.5B-onnx/
```

You can also convert a fine-tuned Qwen variant by pointing Optimum to your model path.

---

## 🚀 How to Build & Run

1. Clone this repository.
2. Install the latest **Android Studio**.
3. Open the Android project folder in Android Studio:

   ```text
   pocket_llm_src/
   ```
4. Place the required model assets in:

   ```text
   app/src/main/assets/
   ```

   You can place `model.onnx` and `tokenizer.json` directly in that folder, or inside a single nested model folder.

5. Add the files that match the model build you selected:

   **For ONNX builds**
   - `model.onnx`
   - `tokenizer.json`

   **For LiteRT builds**
   - the matching `.litertlm` model file for that model

6. In `ModelDescriptor.kt`, set the active model using `SELECTED_MODEL_ID`.
7. Connect your Android phone using USB or wireless debugging.
8. Run the app from Android Studio, or generate a signed APK from:

   **Build → Generate Signed Bundle / APK**

9. Once installed, look for the **Pocket LLM** icon on your device.

**Note**: All Kotlin files are declared in the package `com.example.local_llm`, and the Gradle script uses the same `applicationId`.  
If you rename the app or package, you must also refactor the package declarations, folder structure, and Gradle `applicationId`.

## Customize Your App Experience

- Define the assistant's tone and role using the model's default system prompt
- Adjust `TEMPERATURE` to control response randomness
- Adjust `REPETITION_PENALTY` to reduce repetitive output
- Change `MAX_TOKENS` to control reply length
- Use built-in themes for a different look and feel
- Adjust chat font size from the Settings screen

---

## 📄 License Notice

### Gemma 4

Gemma 4 is provided by Google under the **Apache License 2.0**. Google's Gemma documentation also states that Gemma models are provided with open weights and support responsible commercial use.

- Gemma 4 license: https://ai.google.dev/gemma/apache_2
- Gemma 4 overview: https://ai.google.dev/gemma/docs/core

### Qwen models

Qwen model files follow the upstream Qwen license terms.  
Please review the original model license before redistribution or commercial use.
