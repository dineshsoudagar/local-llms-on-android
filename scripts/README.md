# ONNX LLM Toolkit 🧠⚡

This repo provides lightweight tools for running and managing ONNX-exported large language models (LLMs) on local hardware — with support for:

- 🧠 Efficient autoregressive generation with past key/value (KV) cache
- 📦 Merging ONNX external tensor data into a single portable file
- 🐍 Clean Python CLI tools with no unnecessary dependencies

---

## 📂 File Overview

### 1. `infer_onnx_with_kv_cache.py`

Run fast, incremental inference with ONNX-based LLMs (e.g., Qwen, LLaMA).

**Features:**
- Supports multiple architectures (`qwen2.5`, `qwen3`, `llama3`)
- Uses KV caching to avoid recomputing previous tokens
- CLI interface for model choice, prompt input, and output control

#### ✅ Example usage

```bash
# Default model paths
python infer_onnx_with_kv_cache.py --model qwen3 --prompt "What is the capital of Germany?"

# Custom paths
python infer_onnx_with_kv_cache.py \
  --model llama3 \
  --prompt "Explain relativity in simple terms." \
  --model_path ./models/llama/model.onnx \
  --tokenizer_path ./models/llama/tokenizer.json
```

---

### 2. `merge_onnx_external_data.py`

Merge `.onnx` models that use external tensor storage (e.g., `model.onnx_data`) into a single, self-contained file.

**Why?**
Some ONNX exports store large weights in external files. This tool merges them for easier deployment.

#### ✅ Example usage

```bash
python merge_onnx_external_data.py \
  --model_path qwen_2.5_05B_Instruct_onnx_w_optim/model.onnx

# Optional: specify output file
python merge_onnx_external_data.py \
  --model_path ./big_model/model.onnx \
  --output_path ./big_model/merged_model.onnx
```

---

## 📦 Dependencies

Install minimal required packages:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```txt
onnx
onnxruntime
tokenizers
numpy
rich
tqdm
requests
PyYAML
```

GPU or Hugging Face ecosystem users may also include:
```txt
onnxruntime-gpu
transformers
optimum @ git+https://github.com/huggingface/optimum.git
```

---

## 🚀 Notes

- Ensure that model/tokenizer paths are correct and accessible.
- For GPU use, install `onnxruntime-gpu` and ensure your CUDA environment is configured.
- The `optimum` dependency is only needed if you're converting/exporting models from Hugging Face.

---

## 📜 License

MIT License. Use at your own risk. No warranties, just fast inference 💨
