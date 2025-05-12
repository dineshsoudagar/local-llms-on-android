import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

checkpoint_dir = (r"C:\Others\Projects\LLM-ENGLISH-GERMAN-Small-Translator\out\QWEN_Translator_A1_explanation_hf_model"
                  r"\converted2")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to_onnx()
# Prompt (no Alpaca formatting)
prompt = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Translate to German: Who are you?
"""



# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate with greedy decoding (equivalent to temperature=0)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,        # <- greedy
        temperature=0,        # ignored because do_sample=False
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(">> Prompt:", prompt)
print(">> Reply:", output_text.replace(prompt, "").strip())  # Remove prompt from output