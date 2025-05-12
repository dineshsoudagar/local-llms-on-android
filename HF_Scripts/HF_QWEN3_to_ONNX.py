import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define model name and output path
model_name = "Qwen/Qwen3-0.6B"
onnx_output_path = "qwen3_0.6b_non_thinking.onnx"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Prepare a sample input in non-thinking mode
prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Disable thinking mode
)
inputs = tokenizer(text, return_tensors="pt")

# Export the model to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    onnx_output_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=17,
    do_constant_folding=True
)

print(f"Model successfully exported to {onnx_output_path}")
