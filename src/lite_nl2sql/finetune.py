import os
import json
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from trainer import ModelTrainer,initialize_model, setup_peft_model


def main():
    # Initialize model
    model, tokenizer = initialize_model()
    model = setup_peft_model(model)

    # Train model
    trainer = ModelTrainer(model, tokenizer)
    trainer_instance = trainer.setup_trainer()

    # Memory Stats and Training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer_instance.train()

    # save 8bit
    # model.save_pretrained_gguf("Qwen2.5-1.5B-sql-rslora-q8", tokenizer,)
    # model.push_to_hub_gguf("hf/model", tokenizer, token = "")
        

    # Push to hub
    # model.push_to_hub_gguf(
    #     "theneuralmaze/RickLLama-3.1-8B",
    #     tokenizer,
    #     token=os.getenv("HUGGINGFACE_TOKEN"),
    # )

if __name__ == "__main__":
    main()
