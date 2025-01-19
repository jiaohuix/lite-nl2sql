import os
import json
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# --- Configuration ---
# Model and Data Paths
model_name = "unsloth/Qwen2.5-1.5B"  # Or any other unsloth model
train_file = "data/example_train.json"
dev_file = "data/example_dev.json"
# output_dir = "dbgpt_hub_sql/output/adapter/Qwen2.5-1.5B-sql-lora"
output_dir = "dbgpt_hub_sql/output/adapter/Qwen2.5-1.5B-sql-rslora"


# Training Parameters
max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True
lora_rank = 64
lora_alpha = 32
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 8
warmup_steps = 5
weight_decay = 0.01
seed = 3407

# exp2
save_steps = 200 # 540 steps/epoch
eval_steps = 200
logging_steps = 10
learning_rate = 5e-6

# --- Data Loading and Preprocessing ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
EOS_TOKEN = tokenizer.eos_token

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data



def formatting_prompts_func(examples):
    instructions = [item["instruction"] for item in examples]
    inputs = [item["input"] for item in examples]
    outputs = [item["output"] for item in examples]
    texts = []
    for instruction, ipt, output in zip(instructions, inputs, outputs):
        # text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        text = instruction + ipt + output + EOS_TOKEN
        texts.append(text)
    print("text:",text)
    return {"text": texts}

# def formatting_prompts_func(examples):
#     instructions = [item["instruction"] for item in examples]
#     inputs = [item["input"] for item in examples]
#     outputs = [item["output"] for item in examples]

#     instruct_start_token = "##Instruction"
#     texts = []
#     for instruction, ipt, output in zip(instructions, inputs, outputs):
#         instruct_start_idx = instruction.find(instruct_start_token)
#         system_prompt = instruction[:instruct_start_idx]
#         instruction = instruction[instruct_start_idx:] + f". {ipt}"
#         text = tokenizer.apply_chat_template(
#             [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": instruction},
#                 {'role': 'assistant', 'content': output}
#             ],
#             tokenize=False,
#             add_generation_prompt=False
#         )
#         texts.append(text)
        
#     print("text:",text)
#     return {"text": texts}


train_data = read_json_file(train_file)
train_dataset = Dataset.from_dict({"data": train_data})
train_dataset = train_dataset.map(lambda examples: formatting_prompts_func(examples["data"]), batched=True)

dev_data = read_json_file(dev_file)
dev_dataset = Dataset.from_dict({"data": dev_data})
dev_dataset = dev_dataset.map(lambda examples: formatting_prompts_func(examples["data"]), batched=True)

print("Processed Train Dataset:", train_dataset)
print("Processed Dev Dataset:", dev_dataset)

# --- Model Loading and LoRA Setup ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_alpha,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = seed,
    use_rslora = True, # exp2
    loftq_config = None,
)

# --- Training Setup ---
training_args = TrainingArguments(
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    warmup_steps = warmup_steps,
    num_train_epochs = num_train_epochs,
    eval_strategy ="steps",
    eval_steps = eval_steps,
    save_steps = save_steps,
    learning_rate = learning_rate,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = logging_steps,
    optim = "adamw_8bit",
    weight_decay = weight_decay,
    lr_scheduler_type = "cosine_with_restarts", # Changed to match your original config
    seed = seed,
    output_dir = output_dir,
    report_to = "none",
    overwrite_output_dir = True,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = dev_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = training_args,
)

# --- Memory Stats and Training ---
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 8bit
model.save_pretrained_gguf("Qwen2.5-1.5B-sql-rslora-q8", tokenizer,)
# model.push_to_hub_gguf("hf/model", tokenizer, token = "")
    