import torch
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
def create_position_ids(input_ids, past_length=0):
    """
    Generate position_ids based on input_ids and past_key_values length.
    Args:
        input_ids (torch.Tensor): shape [batch_size, seq_len]
        past_length (int): length of past_key_values (i.e., number of previously generated tokens)

    Returns:
        position_ids (torch.Tensor): shape [batch_size, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    return torch.arange(past_length, past_length + seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def generate(onnx_model_path, tokenizer_dir, prompt_text, max_new_tokens=50):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

    # Format chat input
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="np", add_generation_prompt=True)
    print("=== Initial input_ids ===")
    print(input_ids)
    print("Decoded prompt:", tokenizer.decode(input_ids[0]))

    attention_mask = np.ones_like(input_ids)

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    eos_token_id = tokenizer.eos_token_id
    generated_ids = input_ids.tolist()[0]

    # KV shape detection
    kv_input = [i for i in session.get_inputs() if "past_key_values.0.key" in i.name][0]
    shape = kv_input.shape
    num_heads = shape[1]
    head_dim = shape[3]
    batch_size = 1
    num_layers = 28

    empty_kv = np.zeros((batch_size, num_heads, 0, head_dim), dtype=np.float16)
    past_key_values = {
        f"past_key_values.{i}.key": empty_kv.copy()
        for i in range(num_layers)
    }
    past_key_values.update({
        f"past_key_values.{i}.value": empty_kv.copy()
        for i in range(num_layers)
    })

    print("\n=== Starting generation ===\n")
    for step in range(max_new_tokens):
        if step == 0:
            current_input_ids = input_ids
            current_attention_mask = attention_mask
            position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64).reshape(1, -1)
        else:
            current_input_ids = np.array([[generated_ids[-1]]], dtype=np.int64)
            current_attention_mask = np.ones_like(current_input_ids)
            position_ids = np.array([[input_ids.shape[1] + step - 1]], dtype=np.int64)

        print(f"\n--- Step {step} ---")
        print("Current input_ids:", current_input_ids)
        print("Position_ids:", position_ids)
        print("Decoded so far:", tokenizer.decode(generated_ids, skip_special_tokens=True))

        inputs = {
            "input_ids": current_input_ids,
            "attention_mask": current_attention_mask,
            "position_ids": position_ids,
            **past_key_values,
        }

        outputs = session.run(None, inputs)
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        print("Next token id:", next_token_id, "->", tokenizer.decode([next_token_id]))

        generated_ids.append(next_token_id)

        if next_token_id == eos_token_id:
            print("[STOP] Reached EOS token")
            break

        offset = 1
        for i in range(num_layers):
            past_key_values[f"past_key_values.{i}.key"] = outputs[offset + i * 2]
            past_key_values[f"past_key_values.{i}.value"] = outputs[offset + i * 2 + 1]

    print("\n=== Final Output ===")
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(decoded)

# Load tokenizer (for external use too)
tokenizer = AutoTokenizer.from_pretrained("qwen_3_0.6B", trust_remote_code=True, local_files_only=True)

# Run generation
generate(
    "qwen_3_0.6B/model_fp16.onnx",
    "qwen_3_0.6B",
    "Hi"
)
