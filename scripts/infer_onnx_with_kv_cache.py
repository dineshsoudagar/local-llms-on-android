import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import argparse
import os
import sys

# ===============================
# Default model configurations
# ===============================

DEFAULT_MODELS = {
    "qwen2.5": {
        "model_path": "qwen_2.5_05B_Instruct_onnx/qwen25_05B_fp16.onnx",
        "tokenizer_path": "qwen_2.5_05B_Instruct_onnx/tokenizer.json",
        "num_layers": 24,
        "num_kv_heads": 2,
        "head_dim": 64,
        "end_token_id": 151645,
    },
    "qwen3": {
        "model_path": "qweb_3_0.6B_onnx_from_HF/model.onnx",
        "tokenizer_path": "qweb_3_0.6B_onnx_from_HF/tokenizer.json",
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
        "end_token_id": 151645,
    },
    "llama3": {
        "model_path": "meta-llama_Llama-3.2-1B_onnx/model.onnx",
        "tokenizer_path": "meta-llama_Llama-3.2-1B_onnx/tokenizer.json",
        "num_layers": 16,
        "num_kv_heads": 8,
        "head_dim": 64,
        "end_token_id": 128001,
    }
}

DEFAULT_MAX_TOKENS = 50


# ===============================
# Helpers
# ===============================

def get_providers():
    available = ort.get_available_providers()
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']


def initialize_past_kv(batch_size, num_layers, num_kv_heads, head_dim):
    return {
        f"past_key_values.{layer}.{kv}": np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)
        for layer in range(num_layers)
        for kv in ["key", "value"]
    }


def validate_file(path, label):
    if not os.path.isfile(path):
        print(f"[ERROR] {label} not found: {path}")
        print(f"Either specify it via --{label.lower().replace('_', '')} or set it as default in the script.")
        sys.exit(1)


# ===============================
# Inference
# ===============================

def generate_text(model_key, prompt, max_new_tokens, model_path=None, tokenizer_path=None):
    config = DEFAULT_MODELS[model_key]

    model_path = model_path or config["model_path"]
    tokenizer_path = tokenizer_path or config["tokenizer_path"]
    validate_file(model_path, "model_path")
    validate_file(tokenizer_path, "tokenizer_path")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    session = ort.InferenceSession(model_path, providers=get_providers())

    encoded = tokenizer.encode(prompt)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]

    past_kv = initialize_past_kv(
        batch_size=input_ids.shape[0],
        num_layers=config["num_layers"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"]
    )

    generated = []
    for _ in range(max_new_tokens):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past_kv
        }

        try:
            outputs = session.run(None, inputs)
            logits, *present_kvs = outputs

            next_token = np.argmax(logits[:, -1], axis=-1, keepdims=True)
            token_id = int(next_token[0, 0])
            generated.append(token_id)

            if token_id == config["end_token_id"]:
                break

            input_ids = next_token
            attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
            position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)

            for i, key in enumerate(past_kv):
                past_kv[key] = present_kvs[i]

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            break

    output = tokenizer.decode(generated)
    print("\n=== Generated Text ===")
    print(output)


# ===============================
# CLI Entry
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX LLM inference")

    parser.add_argument("--model", type=str, required=True, choices=DEFAULT_MODELS.keys(),
                        help="Choose a model preset: qwen2.5, qwen3, llama3")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to generate from")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens to generate")
    parser.add_argument("--model_path", type=str, help="Optional: override ONNX model path")
    parser.add_argument("--tokenizer_path", type=str, help="Optional: override tokenizer.json path")

    args = parser.parse_args()

    generate_text(
        model_key=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
