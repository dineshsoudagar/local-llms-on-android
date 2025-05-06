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

levels = ["Provide some meaning of all the words in the german sentence below. Dont add anything else in your "
          "response, Format your answer exactly like this below. \nword meanings: \n word1: word1 explanation, \n word2: word2 explanation. \n Some additional points to take care in your response. \n Point-1 always "
          "provide the absolute meaning of the word. \n Point-2 if the meaning of the word changes in the current context. \n Point-3 Avoid repeating words. \n Point-4 Explain only those words which you think explanation is needed for a beginner level. \n Point-5 If a word is a name, mark it as [Name]. \n Point-6  Use simple and accurate meanings. \n Point-7 Do not add any introductions or conclusions."]





# Load the model to be tested
model_path = r"C:\Others\Projects\LLM-ENGLISH-GERMAN-Small-Translator\checkpoints\meta-llama\Meta-Llama-3-8B-Instruct"
llm = LLM.load(model_path)

# Record the response
Total_samples_needed = 5000
dataset = dataset[20000:]
for i in trange(Total_samples_needed):
    # print(dataset[i])
    for level in levels:
        new_input = level + f"\n German Sentence: {dataset[i]['output']}"
        response = llm.generate(new_input, temperature=0)
        #print(new_input)
        #print(response)
        dataset[i]["A1_response"] = response

with open("de_en_next5K_dataset_with_A1_explanations_llama_3_8B.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(dataset))
