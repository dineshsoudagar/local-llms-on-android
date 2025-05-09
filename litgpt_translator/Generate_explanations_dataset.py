"""
Create dataset with explanations
"""
import json
from tqdm import trange, tqdm
from litgpt import LLM

dataset_file = r"C:\Others\Projects\de-en-txt-data\litgpt_translation_dataset.json"
#dataset_file = "/mnt/c/Others/Projects/de-en-txt-data/litgpt_translation_dataset.json"
with open(dataset_file, "r") as file:
    dataset = json.load(file)

#levels = ["Provide A2 Level german sentence below. Dont add anything else in your "
#          "response, Format your answer exactly like this below. \nword meanings: \n word1: word1 explanation, \n word2: word2 explanation. \n Some additional points to take care in your response. \n Point-1 always "
#          "provide the absolute meaning of the word. \n Point-2 if the meaning of the word changes in the current context. \n Point-3 Avoid repeating words. \n Point-4 Explain only those words which you think explanation is needed for a beginner level. \n Point-5 If a word is a name, mark it as [Name]. \n Point-6  Use simple and accurate meanings. \n Point-7 Do not add any introductions or conclusions."]
levels = ["""
You are a German teacher for A1 learners.

Task: Given a German sentence, provide:
1. Word meanings for key words in the sentence (only if needed for A1 level).
2. One simple grammar explanation related to the sentence.

Format your output exactly like this:

word meanings:
word1: <simple A1-level meaning>
word2: <simple A1-level meaning>
...

grammar tip:
<One short grammar explanation related to this sentence>

Rules:
- Do not include any extra text before or after.
- Use only the required format.
- Use simple and clear explanations.
- Mark names as [Name].
- Explain only what is necessary for a beginner.
"""]




# Load the model to be tested
model_path = r"/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM.load(model_path)

# Record the response
Total_samples_needed = 30000


def save_dataset(dataset, index):
    """
    :param dataset:
    :param index:
    """
    filename = f"de_en_dataset_with_word_meanings_grammar_tip_llama_3_8B_up_to_{index}.json"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset, ensure_ascii=False, indent=4))


# Begin processing the dataset
for i in trange(Total_samples_needed):
    for level in levels:
        new_input = level + f"\n German Sentence: {dataset[i]['output']}"
        response = llm.generate(new_input, temperature=0, max_new_tokens=100)
        dataset[i]["A1_response"] = response

    # Save progress every 5000 samples to avoid data loss
    if (i + 1) % 5000 == 0 or (i + 1) == Total_samples_needed:
        save_dataset(dataset, i + 1)  # Save after every 5000 samples

# Final save if the last sample wasn't saved in the loop
save_dataset(dataset, Total_samples_needed)
