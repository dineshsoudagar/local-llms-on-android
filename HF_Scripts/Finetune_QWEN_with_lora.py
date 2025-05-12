from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import torch

model_name = "Qwen/Qwen2.5-0.5B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Load tokenizer and apply chat template
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model and prepare for LoRA fine-tuning
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # Or load_in_8bit=True
    #quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True  # Don't use quantization on Windows
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
print("trainable_parameters", model.get_nb_trainable_parameters())
# Load dataset
dataset = load_dataset("json", data_files="Version2_en_de_dataset_A1_level_for30k_llama_3_8B_HF_format.json")


# Tokenize using chat template
def format_and_tokenize(example):
    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    tokenized_ = tokenizer(
        formatted_text,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized_["input_ids"][0],
        "attention_mask": tokenized_["attention_mask"][0]
    }


tokenized = dataset.map(format_and_tokenize)
print(tokenized.keys())
# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen3-lora-checkpoints",
    eval_steps=100,  # Evaluate every 100 steps
    logging_steps=50,
    save_steps=500,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=6e-5,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=5,
    report_to="none",
    warmup_steps=100,
    dataloader_num_workers=4
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.args.gradient_checkpointing = False
trainer.train()
