import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# -------- CONFIG --------
MODEL_PATH = "gemma2_with_past_KV_onnx_final_15jun_2/decoder_with_past_model.onnx"
TOKENIZER_PATH = "gemma2_with_past_KV_onnx_final_15jun_2/tokenizer.json"
MAX_NEW_TOKENS = 50
NUM_LAYERS = 26
NUM_KV_HEADS = 4
HEAD_DIM = 256
EOS_TOKEN_IDS = [1, 107]

# -------- LOAD MODEL & TOKENIZER --------
providers = ['CUDAExecutionProvider',
             'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else [
    'CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

prompt = "Translate to German: I want to go to the office early."
encoded = tokenizer.encode(prompt)
input_ids = np.array([encoded.ids], dtype=np.int64)
attention_mask = np.ones_like(input_ids, dtype=np.int64)
position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
batch_size = input_ids.shape[0]
past_key_values = {}

for layer in range(NUM_LAYERS):
    for kv in ["key", "value"]:
        name = f"past_key_values.{layer}.{kv}"
        past_key_values[name] = np.zeros((batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

# 5. Generation loop
generated = []
max_new_tokens = 50
for i in range(max_new_tokens):
    print(f"Generating new token {i}")
    # Dynamically create the inputs dictionary (including past_key_values)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    inputs.update(past_key_values)  # Add past_key_values to inputs

    try:
        # Run inference
        outputs = session.run(None, inputs)
        logits, *present_values = outputs

        # Get next token
        next_token_id = np.argmax(logits[:, -1], axis=-1, keepdims=True)
        token_id = int(next_token_id[0, 0])
        generated.append(token_id)

        # Update inputs for the next token
        input_ids = next_token_id
        #attention_mask = np.ones_like(input_ids, dtype=np.int64)
        #position_ids = position_ids[:, -1:] + 1

        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)
        # Update past_key_values for the next step
        for i, key in enumerate(past_key_values.keys()):
            #print(key)
            #print(present_values[i].shape)
            past_key_values[key] = present_values[i]

        if token_id in EOS_TOKEN_IDS:  # End token <|im_end|>
            break

    except Exception as e:
        print(f"Error during inference: {e}")
        break

# 6. Decode and print the result
output_text = tokenizer.decode(generated)
print("Generated text:", output_text)
