
# 🤖 Local LLMs on Android (Offline, Private & Fast)

An Android application that brings a large language model (LLM) to your phone — fully offline, no internet needed. Powered by ONNX Runtime and a Hugging Face-compatible tokenizer, it provides fast, private, on-device question answering with streaming responses.

---

## ✨ Features

- 📱 On-device, offline question answering with ONNX Runtime
- 🔤 Hugging Face-compatible BPE tokenizer (`tokenizer.json`)
- 🧠 Qwen-compatible prompt formatting with streaming token generation
- 🚀 Coroutine-based inference for responsive performance
- 🔐 Runs fully offline — your data stays on your device

---

## 📸 Inference Preview

<p align="center">
  <img src="data/Demo.gif" alt="Model Output 2" width="25%" style="margin: 1%"/>
  <img src="data/Demo2.gif" alt="Input Prompt" width="25%" style="margin: 1%"/>
  </p>

<p align="center">
  <em>Figure: App interface showing prompt input and generated answers using the local LLM.</em>
</p>

---

## 🧠 Model Info

This app uses the **Qwen2.5-0.5B-Instruct** model optimized for instruction-following and QA tasks.

### 🔁 Option 1: Use Preconverted ONNX Model

- Download [ONNX model](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/tree/main/onnx) and [tokenizer.json](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/tree/main) from Hugging Face.

### ⚙️ Option 2: Convert Model Yourself

Install Optimum with ONNX export support:

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

## 📦 Download Prebuilt APK

- ➡️ [pocket_llm_qwen2.5_0.5B_v1.0.0.apk](https://github.com/dineshsoudagar/Local-LLM-On-Andriod-Qwen-QA/releases/download/v1.0.0/pocket_llm_qwen2.5_0.5B_v1.0.0.apk)
  - Best for high-end devices. Full precision (FP32).
- ➡️ [pocket_llm_qwen2.5_0.5B_fp16_v1.0.0.apk](https://github.com/dineshsoudagar/Local-LLM-On-Andriod-Qwen-QA/releases/download/v1.0.0/pocket_llm_qwen2.5_0.5B_fp16_v1.0.0.apk)
  - Optimized for mid to high-end phones. Uses half-precision (FP16).
- ➡️ [pocket_llm_qwen2.5_0.5B_q4fp16_v1.0.0.apk](https://github.com/dineshsoudagar/Local-LLM-On-Andriod-Qwen-QA/releases/download/v1.0.0/pocket_llm_qwen2.5_0.5B_q4fp16_v1.0.0.apk)
  - Best overall balance of speed and size. Quantized (Q4 + FP16).
---

## 🔐 Privacy First

This app performs all inference locally on your device. No data is sent to any server, ensuring full privacy and low latency.

---

## 🔮 Roadmap

- 🔁 **Chat Memory** — Add multi-turn conversation with context retention.
- 🧠 **Qwen3-0.6B** — Update to newer, more capable Qwen model.
- 🦙 **LLaMA 3 1B** — Support Meta’s new compact LLM.

## 📄 License

MIT License — use freely, modify locally, and deploy offline. Contributions welcome!
