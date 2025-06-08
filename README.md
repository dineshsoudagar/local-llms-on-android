# 🤖 Local LLMs on Android (Offline, Private & Fast)

An Android application that brings a large language model (LLM) to your phone — fully offline, no internet needed. Powered by ONNX Runtime and a Hugging Face-compatible tokenizer, it provides fast, private, on-device question answering with streaming responses.

---

## ✨ Features

- 📱 Fully on-device LLM inference with ONNX Runtime  
- 🔤 Hugging Face-compatible BPE tokenizer (`tokenizer.json`)  
- 🧠 Qwen2.5 & Qwen3 prompt formatting with streaming generation  
- 🧩 Custom `ModelConfig` for precision, prompt style, and KV cache  
- 🧘‍♂️ **Thinking Mode** toggle (enabled in Qwen3) for step-by-step reasoning  
- 🚀 Coroutine-based UI for smooth user experience  
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
- [ONNX Runtime for Android](https://github.com/microsoft/onnxruntime-genai/releases) (already included in this repo)
- A physical Android device for deployment and testing

---

## 📲 How to Build & Run

1. Open Android Studio and create a new project (Empty Activity).
2. Name your app `local_llm`.
3. Copy all the project files from this repo into the appropriate folders.
4. Place your `model.onnx` and `tokenizer.json` in:
   ```
   app/src/main/assets/
   ```
5. Connect your Android phone using wireless debugging or USB.
6. To install:
   - Press Run ▶️ in Android Studio, **or**
   - Go to **Build → Generate Signed Bundle / APK** to export the `.apk` file.

---

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

## 🔐 Privacy First

This app performs all inference locally on your device. No data is sent to any server, ensuring full privacy and low latency.

---

## 🔮 Roadmap

- 🧠 **Qwen3-0.6B** — Added Qwen3 model support.
- 🔁 **Chat Memory** — Add multi-turn conversation with context retention.
- 🦙 **LLaMA 3 1B** — Support Meta’s new compact LLM.

## 📄 License

MIT License — use freely, modify locally, and deploy offline. Contributions welcome!
