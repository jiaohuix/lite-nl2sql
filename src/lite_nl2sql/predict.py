import os
import sys
import json

import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from peft import PeftModel

# --- Configuration ---
# Model and Data Paths
model_name = "unsloth/Qwen2.5-1.5B"  # Or any other unsloth model
predict_file = "data/example_dev.json"
lora_dir = sys.argv[1]
output_file = sys.argv[2]
# output_file = "dbgpt_hub_sql/output/pred_sql.sql"

# lora_dir = "dbgpt_hub_sql/output/adapter/Qwen2.5-1.5B-sql-rslora/checkpoint-200/"
# lora_dir = "dbgpt_hub_sql/output/adapter/Qwen2.5-1.5B-sql-lora/checkpoint-2000/"

# --- Model Loading ---
max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True

tokenizer = AutoTokenizer.from_pretrained(model_name)
EOS_TOKEN = tokenizer.eos_token

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

print("model loader...")

# --- Data Loading and Preprocessing ---
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# exp1
def formatting_prompts_func(examples):
    instructions = [item["instruction"] for item in examples]
    inputs = [item["input"] for item in examples]
    texts = []
    for instruction, ipt in zip(instructions, inputs):
        text = instruction + ipt
        texts.append(text)
    return {"text": texts}


predict_data = read_json_file(predict_file)
predict_dataset = Dataset.from_dict({"data": predict_data})
predict_dataset = predict_dataset.map(lambda examples: formatting_prompts_func(examples["data"]), batched=True)

print("Processed Predict Dataset:", predict_dataset)

# --- Prediction ---
# def generate_sql(model, tokenizer, prompt):
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=max_seq_length, truncation=True).to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=512)
#     sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("prompt：",prompt)
#     print("sql：",sql)
    
#     return sql

def generate_sql(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_seq_length, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    
    # 计算 inputs 的长度
    input_length = inputs["input_ids"].shape[1]
    
    # 去掉 outputs 中对应于 inputs 的部分
    trimmed_outputs = outputs[:, input_length:]
    
    # 解码剩余部分
    sql = tokenizer.decode(trimmed_outputs[0], skip_special_tokens=True)
    
    return sql.strip()
    
def predict_and_save(model, tokenizer, dataset, output_file):
    generated_sqls = []
    for item in tqdm(dataset):
        prompt = item["text"]
        sql = generate_sql(model, tokenizer, prompt)
        generated_sqls.append(sql)

    with open(output_file, "w", encoding="utf-8") as f:
        for sql in generated_sqls:
            f.write(sql.strip() + "\n")

# --- Run Prediction ---
predict_and_save(model, tokenizer, predict_dataset, output_file)
print(f"SQL queries saved to {output_file}")
