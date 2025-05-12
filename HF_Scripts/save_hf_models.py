from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pretrained Qwen3-0.6B model and tokenizer
model_name = "Qwen/Qwen1.5-0.5B"  # change to your local or fine-tuned path if needed

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer in FP16 format
output_dir = "../checkpoints/HF_pretrained/qwen3_0_6b_fp16_textgen"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
