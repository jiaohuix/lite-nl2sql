DATASET_CONFIG = {
    "train": "data/example_train.json",
    "dev": "data/example_dev.json",
}

MAX_SEQ_LENGTH = 2048

MODEL_CONFIG = {
    "model_name": "unsloth/Qwen2.5-1.5B",
    "load_in_4bit": False,
    "dtype": None,
}

PEFT_CONFIG = {
    "r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],
    "use_rslora": True,
    "use_gradient_checkpointing": "unsloth",
}

TRAINING_ARGS = {
    "learning_rate": 5e-6,
    "lr_scheduler_type": "linear",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 8,
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "output_dir": "output",
    "seed": 3407,
    "report_to": "none",
}