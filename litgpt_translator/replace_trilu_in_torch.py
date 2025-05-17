import torch
from litgpt import LLM, Tokenizer


import torch
import torch.nn as nn

# 1. Backup original tril
torch_tril_original = torch.tril

# 2. Define custom tril (ONNX-friendly version)
def custom_tril(input, diagonal=0):
    size = input.size(-1)
    row = torch.arange(size, device=input.device).unsqueeze(1)
    col = torch.arange(size, device=input.device).unsqueeze(0)
    mask = (row >= col).to(dtype=input.dtype)
    return input * mask

# 3. Monkey patch
torch.tril = custom_tril

# 4. Load your model
model = LLM.load(r"C:\Others\Projects\LLM-ENGLISH-GERMAN-Small-Translator\checkpoints\meta-llama\Llama-3.2-1B-Instruct")
print(model)
tokenizer = Tokenizer(r"C:\Others\Projects\LLM-ENGLISH-GERMAN-Small-Translator\checkpoints\meta-llama\Llama-3.2-1B-Instruct")
model.eval()

# 5. Run dummy forward and trace it
# Replace `dummy_input` with the actual input shape your model expects
dummy_input = torch.randint(0, 1000, (1, 128), dtype=torch.long, device="cuda")  # e.g., token ids for a GPT model
traced_model = torch.jit.trace(model, dummy_input)

# 6. Save traced model
traced_model.save("model_without_trilu.pth")

# 7. Optional: Restore torch.tril
torch.tril = torch_tril_original
