# 🤖 Local LLMs on Android (Offline, Private & Fast)

An Android application that brings a large language model (LLM) to your phone — fully offline, no internet needed. Powered by ONNX Runtime and a Hugging Face-compatible tokenizer, it provides fast, private, on-device question answering with streaming responses.

---

## ✨ Features

- 📱 Fully on-device LLM inference with ONNX Runtime.
- 🔤 Hugging Face-compatible BPE tokenizer (`tokenizer.json`)  
- 🧠 Qwen2.5 & Qwen3 prompt formatting with streaming generation  
- 🧩 Custom `ModelConfig` for precision, prompt style, and KV cache  
- 🧘‍♂️ **Thinking Mode** toggle (enabled in Qwen3) for step-by-step reasoning  
- 🚀 Coroutine-based UI for smooth user experience.
- 🔐 Runs 100% offline — no network, no telemetry  

---

## 📸 Inference Preview

<p align="center">
  <img src="data/Demo.gif" alt="Model Output 2" width="25%" style="margin: 1%"/>
  <img src="data/Demo2.gif" alt="Input Prompt" width="25%" style="margin: 1%"/>
  <img src="data/Qwen3demo.gif" alt="Input Prompt" width="25%" style="margin: 1%"/>
</p>

<p align="center">
  <em>Figure: App interface showing prompt input and generated answers using the local LLM.</em>
</p>

---

## 📂 App Variants

This repo includes **two modes** of interaction:

### [Qwen_QA_style_app](Qwen_QA_style_app)
- Single-turn QA with minimal prompt.
- Fastest response time.
- Best for quick facts or instructions.

### [Qwen_chat_style_app](Qwen_chat_style_app)
- Multi-turn chat with short-term memory.
- Qwen-style prompt formatting with context compression.
- Best for reasoning, assistant-style dialogue, and follow-up questions.

---

## 🧠 Model Info

This app supports both **Qwen2.5-0.5B-Instruct** and **Qwen3-0.6B** — optimized for instruction-following, QA, and reasoning tasks.

### 🔁 Option 1: Use Preconverted ONNX Model

Download the `model.onnx` and `tokenizer.json` from Hugging Face:

- 🔹 [Qwen2.5](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)  
- 🔹 [Qwen3](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX)  

### ⚙️ Option 2: Convert Model Yourself

```bash
pip install optimum[onnxruntime]
# or
python -m pip install git+https://github.com/huggingface/optimum.git
```

Export the model:

```bash
optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct qwen2.5-0.5B-onnx/
```

- You can also convert any fine-tuned variant by specifying the model path.
- Learn more about [Optimum here](https://huggingface.co/docs/optimum/main/en/index).

---

## ⚙️ Requirements

- [Android Studio](https://developer.android.com/studio)
- [ONNX Runtime for Android](https://github.com/microsoft/onnxruntime-genai/releases) (already included in this repo).
- A physical Android device for deployment and testing, ≥ 4 GB RAM for FP16 / Q4 models, ≥ 6 GB RAM for FP32 models.
- Real hardware preferred—emulators are acceptable for UI checks only.

---
---
#### Choose which Qwen model to run

In[MainActivity.kt](app/src/main/java/com/example/local_llm/MainActivity.kt) you will find two pre-defined `ModelConfig` objects:

```kotlin
val modelconfigqwen25 = …   // Qwen 2.5-0.5B
val modelconfigqwen3  = …   // Qwen 3-0.6B
````
Right below them is a single line that tells the app which one to use:

````kotlin
val config = modelconfigqwen25      // ← change to modelconfigqwen3 for Qwen 3
````

----
## How to Build & Run

1. Open Android Studio and create a new project (Empty Activity).
2. Name your app `local_llm`.
3. Copy all the project files from [Qwen_QA_style_app](Qwen_QA_style_app) or [Qwen_chat_style_app](Qwen_chat_style_app) into the appropriate folders.
4. Place your `model.onnx` and `tokenizer.json` in:
   ```
   app/src/main/assets/
   ```
5. Connect your Android phone using wireless debugging or USB.
6. To install:
   - Press Run ▶️ in Android Studio, **or**
   - Go to **Build → Generate Signed Bundle / APK** to export the `.apk` file.
7. Once installed, look for the **Pocket LLM** icon&nbsp;
   <img src="data/pocket_llm_icon.png" alt="Pocket LLM icon" width="28" style="vertical-align:middle;border-radius:100%"/>
   on your home screen.

**Note**: All Kotlin files are declared in the package com.example.local_llm, and the Gradle script sets applicationId "com.example.local_llm".
If you name the app (or change the package) to anything other than local_llm, you must refactor:
- The directory structure in app/src/main/java/...,                     
- Every package com.example.local_llm line, and
- The applicationId in app/build.gradle.
- Otherwise, Android Studio will raise “package … does not exist” errors and the project will fail to compile.
----

## 📦 Download Prebuilt APKs

- ➡️ [pocket_llm_qwen2.5_0.5B_v1.1.0.apk](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.1.0/pocket_llm_qwen2.5_0.5B_v1.1.0.apk)  
  - Full precision (FP32). Best for high-end devices. Improved inference performance.

- ➡️ [pocket_llm_qwen2.5_0.5B_fp16_v1.1.0.apk](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.1.0/pocket_llm_qwen2.5_0.5B_fp16_v1.1.0.apk)  
  - Half-precision (FP16). Great balance of speed and accuracy for most devices.

- ➡️ [pocket_llm_qwen2.5_0.5B_q4fp16_v1.1.0.apk](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.1.0/pocket_llm_qwen2.5_0.5B_q4fp16_v1.1.0.apk)  
  - Quantized Q4 + FP16. Fastest and lightest version of Qwen2.5.

- ➡️ [pocket_llm_qwen3_0.6B_fp16_v1.1.0.apk](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.1.0/pocket_llm_qwen3_0.6B_fp16_v1.1.0.apk)  
  - 🔥 New! Qwen3-0.6B with improved reasoning and **Thinking Mode** support.

- ➡️ [pocket_llm_qwen3_0.6B_q4fp16_v1.1.0.apk](https://github.com/dineshsoudagar/local-llms-on-android/releases/download/v1.1.0/pocket_llm_qwen3_0.6B_q4fp16_v1.1.0.apk)  
  - 🔥 New! Qwen3 quantized version (Q4 + FP16). Compact and fast with Thinking Mode.

## Customize Your App Experience with These
- Define the assistant’s tone and role by setting defaultSystemPrompt (in your model config).
- Adjust TEMPERATURE to control response randomness — lower for accuracy, higher for creativity ([OnnxModel.kt](app/src/main/java/com/example/local_llm/OnnxModel.kt)).
- Use REPETITION_PENALTY to avoid repetitive answers and improve fluency ([OnnxModel.kt](app/src/main/java/com/example/local_llm/OnnxModel.kt)).
- Change MAX_TOKENS to limit or expand the length of generated replies ([OnnxModel.kt](app/src/main/java/com/example/local_llm/OnnxModel.kt)).

### 📄 License Notice
Note: These ONNX models are based on Qwen, which is licensed under the [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE).
