import json
from tqdm import trange, tqdm
from litgpt import LLM
import random

dataset_file = r"de_en_dataset_with_A1_explanations_for20k_llama_3_8B.json"
with open(dataset_file, "r") as file:
    translator_dataset = json.load(file)
with open("de_en_next5K_dataset_with_A1_explanations_llama_3_8B.json", "r") as file2:
    explanation_dataset = json.load(file2)
print(len(translator_dataset))
modified_explanation_dataset = []
for entry in tqdm(explanation_dataset):
    try:
        new_entry = {"instruction": f"Explain the German sentence for A1 level: {entry['output']}",
                     "output": f"{entry['A1_response']}"}
        modified_explanation_dataset.append(new_entry)
    except:
        continue

final_dataset = translator_dataset + modified_explanation_dataset
print(len(final_dataset))
print(len(modified_explanation_dataset))
random.shuffle(final_dataset)

with open("../datasets/Version1_de_en_dataset_with_A1_explanations_for25k_llama_3_8B.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(final_dataset))
