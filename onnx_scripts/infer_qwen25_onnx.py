import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

# 1. Load the ONNX model with a fallback to CPU if CUDA is unavailable
providers = ['CUDAExecutionProvider',
             'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else [
    'CPUExecutionProvider']
session = ort.InferenceSession("qwen_2.5_05B_onnx_converted_w_optim/model.onnx", providers=providers)

# 2. Load the tokenizer from JSON
tokenizer = Tokenizer.from_file("Qwen2.5-0.5B-Instruct_from_HF/tokenizer.json")

# 3. Prepare the prompt
prompt = "What is the capital of France?"
encoded = tokenizer.encode(prompt)
input_ids = np.array([encoded.ids], dtype=np.int64)
attention_mask = np.ones_like(input_ids, dtype=np.int64)
position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]

# 4. Initialize past key values (dynamically based on model input signature)
num_layers = 24
num_kv_heads = 2
head_dim = 64  # Correct head_dim to match model's expectations
batch_size = input_ids.shape[0]

# Check the input shapes expected by the model
print("Model input shapes:")
for input_meta in session.get_inputs():
    print(f"Input: {input_meta.name}, Shape: {input_meta.shape}")

# Initialize past_key_values with correct shape (based on model's input)
past_key_values = {}
for layer in range(num_layers):
    for kv in ["key", "value"]:
        name = f"past_key_values.{layer}.{kv}"
        past_key_values[name] = np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=np.float32)

# 5. Generation loop
generated = []
max_new_tokens = 50
for _ in range(max_new_tokens):
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
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        position_ids = position_ids[:, -1:] + 1

        # Update past_key_values for the next step
        for i, key in enumerate(past_key_values.keys()):
            past_key_values[key] = present_values[i]

        if token_id == 151645:  # End token <|im_end|>
            break

    except Exception as e:
        print(f"Error during inference: {e}")
        break

# 6. Decode and print the result
output_text = tokenizer.decode(generated)
print("Generated text:", output_text)
