# Re-run the data preparation now that both files are uploaded

# File paths
de_path = "C:\Others\Projects\de-en-txt-data\de-en_v2.txt\Tatoeba.de-en.de"
en_path = "C:\Others\Projects\de-en-txt-data\de-en_v2.txt\Tatoeba.de-en.en"
output_path = "C:\Others\Projects\de-en-txt-data\litgpt_translation_test_dataset.json"

# Load up to 1000 sentence pairs
num_samples = 1000

with open(en_path, "r", encoding="utf-8") as f_en, \
        open(de_path, "r", encoding="utf-8") as f_de:
    en_sentences = [line.strip() for line in f_en][:num_samples]
    de_sentences = [line.strip() for line in f_de][:num_samples]

# Prepare Lit-GPT instruction-tuning format
dataset = []
for en, de in zip(en_sentences, de_sentences):
    dataset.append({
        "instruction": f"Translate to German: {en}",
        "output": f"{de}"
    })

# Save to JSONL file
import json

with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset))
