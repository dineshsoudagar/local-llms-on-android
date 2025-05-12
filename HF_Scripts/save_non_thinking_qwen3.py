from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Prepare prompt without "thinking" mode
messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # ❌ disable thinking mode
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)
output = tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("content:", output)

model.save_pretrained("./qwen3_no_thinking")
tokenizer.save_pretrained("./qwen3_no_thinking")