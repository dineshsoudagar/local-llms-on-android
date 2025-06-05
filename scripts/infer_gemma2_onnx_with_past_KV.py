import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# === Config ===
MODEL_PATH = "gemma2_decoder_with_past_latest_2/gemma2_decoder_with_past_latest_2.onnx"
BOS_TOKEN_ID = 2
EOS_TOKEN_IDS = [1, 107]
MAX_NEW_TOKENS = 50
NUM_LAYERS = 26
NUM_KV_HEADS = 4
HEAD_DIM = 256

# === Load model ===
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# === Encode prompt ===
prompt = "User: Translate this sentence into german language: I want to go to office early today.  \nAssistant:"
encoded = tokenizer.encode(prompt)
input_ids = np.array([encoded], dtype=np.int64)
attention_mask = np.ones_like(input_ids, dtype=np.int64)
position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
cur_pos = input_ids.shape[1]

print("Prompt tokens:", input_ids.tolist())
print("Decoded:", tokenizer.decode(input_ids[0]))

# === Initialize empty past_key_values before loop
past_key_values = {
    f"past_key_values.{layer}.{kind}": np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
    for layer in range(NUM_LAYERS)
    for kind in ["key", "value"]
}

# === Inference loop
generated = []
cur_pos = input_ids.shape[1]

for step in range(MAX_NEW_TOKENS):
    position_ids = np.array([[cur_pos]], dtype=np.int64)

    inputs_ = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **past_key_values,  # ✅ always include
    }

    try:
        outputs = session.run(None, inputs_)
        logits, *present_values = outputs
        print(f"[DEBUG] Step {step}, logits shape: {logits.shape}")

        if logits.ndim == 3:
            next_token_id = np.argmax(logits[:, -1], axis=-1, keepdims=True)
        elif logits.ndim == 2:
            next_token_id = np.argmax(logits, axis=-1, keepdims=True).reshape(1, 1)
        else:
            raise ValueError("Unexpected logits shape")

        token = int(next_token_id[0, 0])
        generated.append(token)

        if token in EOS_TOKEN_IDS:
            break

        # Prepare next step
        input_ids = next_token_id
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        cur_pos += 1

        for i, key in enumerate(past_key_values):
            past_key_values[key] = present_values[i]

    except Exception as e:
        print("Error during inference:", e)
        break

# === Decode
output_text = tokenizer.decode(generated)
print("Generated text:", output_text)
