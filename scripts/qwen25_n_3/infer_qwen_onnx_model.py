"""
Infer QWEN ONNX model
"""

import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

# Configuration
MODEL_PATH = "qwen_2.5_05B_Instruct_onnx_w_optim/model.onnx"
TOKENIZER_PATH = "qwen_2.5_05B_Instruct_onnx_w_optim/tokenizer.json"
END_TOKEN_ID = 151645  # <|im_end|>
MAX_NEW_TOKENS = 50
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64

# 1. Load the ONNX model with CUDA fallback to CPU
available_providers = ort.get_available_providers()
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# 2. Load the tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# 3. Encode input prompt
prompt = "What is the capital of France?"
encoded = tokenizer.encode(prompt)
input_ids = np.array([encoded.ids], dtype=np.int64)
attention_mask = np.ones_like(input_ids, dtype=np.int64)
position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]

# 4. Initialize empty past key values
batch_size = input_ids.shape[0]
past_key_values = {
    f"past_key_values.{layer}.{kv}": np.zeros((batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
    for layer in range(NUM_LAYERS)
    for kv in ["key", "value"]
}

# 5. Generation loop
generated = []

for _ in range(MAX_NEW_TOKENS):
    # Prepare model inputs
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **past_key_values,
    }

    try:
        # Run inference
        outputs = session.run(None, inputs)
        logits, *present_values = outputs

        # Get next token (greedy decoding)
        next_token_id = np.argmax(logits[:, -1], axis=-1, keepdims=True)
        token_id = int(next_token_id[0, 0])
        generated.append(token_id)

        # Early stopping if end token is generated
        if token_id == END_TOKEN_ID:
            break

        # Update inputs for next step
        input_ids = next_token_id
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)

        # Update past key values
        for i, key in enumerate(past_key_values):
            past_key_values[key] = present_values[i]

    except Exception as e:
        print(f"Error during inference: {e}")
        break

# 6. Decode and print result
output_text = tokenizer.decode(generated)
print("Generated text:", output_text)
