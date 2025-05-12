import json
from tqdm import tqdm

dataset_file = (r"C:\Others\Projects\LLM-ENGLISH-GERMAN-Small-Translator"
                r"\de_en_dataset_with_word_meanings_grammar_tip_llama_3_8B_up_to_30000.json")

with open(dataset_file,  encoding="utf-8") as file:
    dataset = json.load(file)

print(len(dataset))

HF_dataset = []
litgpt_dataset = []
for entry in tqdm(dataset):
    # Translation entry
    user_content = f"Translate to English: {entry['output']}"
    assistant_content = f"{entry['instruction'].rsplit(':')[-1]}"
    HF_dataset.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    })
    litgpt_dataset.append({
        "instruction": user_content,
        "output": assistant_content
    })

    # Optional A1 explanation entry
    if "A1_response" in entry:
        user_content = f"Explain the German sentence: {entry['output']}, Level:A1"
        assistant_content = f"{entry['A1_response']}"
        litgpt_dataset.append({
            "instruction": user_content,
            "output": assistant_content
        })
        HF_dataset.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

with open("../datasets/Version2_en_de_dataset_A1_level_for30k_llama_3_8B_HF_format.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(HF_dataset))

with open("../datasets/Version3_en_de_dataset_A1_level_for30k_llama_3_8B_litgpt_format.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(litgpt_dataset))
