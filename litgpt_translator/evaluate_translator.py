"""
Evaluate an LLM model response based on another good model
"""
import json
from tqdm import trange, tqdm
from litgpt import LLM

test_data_file = r"C:\Others\Projects\de-en-txt-data\litgpt_translation_test_dataset.json"
with open(test_data_file, "r") as file:
    test_data = json.load(file)


def generate_model_scores(data_dict, model, response_field="response", target_field="output"):
    scores = []
    for entry in tqdm(data_dict, desc="Scoring entries"):
        prompt = (
            f"Given the input f{entry['instruction']} "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = model.generate(prompt, max_new_tokens=50)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores


# Load the model to be tested
llm = LLM.load(r"C:\Others\Projects\LLM-ENGLISH-GERMAN-Small-Translator\out\finetune\lora\QWEN_Translator_plus_A1_explanation_2\final")

# Record the response
for i in trange(len(test_data)):
    response = llm.generate(test_data[i]["instruction"], temperature=0)
    test_data[i]["response"] = response

with open("../extra_files/test_data_response_2.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(test_data))

with open("../extra_files/test_data_response_2.json", "r") as response_file:
    test_data_response = json.load(response_file)

## Loader a scorer
#del llm  # delete previous `llm` to free up GPU memory
scorer = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
scores = generate_model_scores(test_data_response, model=scorer)
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")