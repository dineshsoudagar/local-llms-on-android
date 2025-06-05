import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

# === CONFIGURATION ===
MODEL_PATH = "gemma2_onnx_no_past_kv/model.onnx"
TOKENIZER_PATH = "gemma2_onnx_no_past_kv/tokenizer.json"
EOS_TOKEN_IDS = {1, 107}
MAX_NEW_TOKENS = 50

# Model arch params (no caching)
NUM_LAYERS = 26
NUM_KV_HEADS = 4
HEAD_DIM = 256

# === 1. Load model ===
available_providers = ort.get_available_providers()
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# === 2. Load tokenizer ===
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# === 3. Encode input ===
prompt = "What is the capital of France?"
encoded = tokenizer.encode(prompt)
input_ids = encoded.ids
generated = input_ids.copy()

for _ in range(MAX_NEW_TOKENS):
    input_array = np.array([generated], dtype=np.int64)
    attention_mask = np.ones_like(input_array, dtype=np.int64)
    position_ids = np.arange(len(generated), dtype=np.int64)[None, :]

    # Prepare inputs (no past key/values in this model)
    inputs = {
        "input_ids": input_array,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    try:
        outputs = session.run(None, inputs)
        logits = outputs[0]
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)

        if next_token in EOS_TOKEN_IDS:
            break
    except Exception as e:
        print("Inference error:", e)
        break

# === 4. Decode output ===
output_text = tokenizer.decode(generated[len(input_ids):])
print("Generated:", output_text)
