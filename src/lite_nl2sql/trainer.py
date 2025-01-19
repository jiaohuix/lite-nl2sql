import json
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from config import DATASET_CONFIG, MAX_SEQ_LENGTH, TRAINING_ARGS, MODEL_CONFIG, PEFT_CONFIG
from exceptions import UnslothNotInstalledError

try:
    from unsloth import standardize_sharegpt, apply_chat_template, is_bfloat16_supported, FastLanguageModel    # type: ignore
except ImportError:
    raise UnslothNotInstalledError

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _formatting_prompts_func(self, examples):
        EOS_TOKEN = self.tokenizer.eos_token

        instructions = [item["instruction"] for item in examples]
        inputs = [item["input"] for item in examples]
        outputs = [item["output"] for item in examples]
        texts = []
        for instruction, ipt, output in zip(instructions, inputs, outputs):
            text = instruction + ipt + output + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def _load_dataset(self, file_path):
        dataset = self._load_json(file_path)
        dataset = Dataset.from_dict({"data": dataset})
        dataset = dataset.map(lambda examples: self._formatting_prompts_func(examples["data"]), batched=True)

        return dataset
    
    def setup_trainer(self):
        train_dataset = self._load_dataset(DATASET_CONFIG["train"])
        dev_dataset = self._load_dataset(DATASET_CONFIG["dev"])
        training_args = TrainingArguments(
            **TRAINING_ARGS,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
        )

        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset  = dev_dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=True,
            args=training_args,
        )
    

def initialize_model():
    """Initialize and return the base model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        dtype=MODEL_CONFIG["dtype"],
    )
    return model, tokenizer


def setup_peft_model(model):
    """Apply PEFT configuration to the model."""
    return FastLanguageModel.get_peft_model(model, **PEFT_CONFIG)


# todo: use qwen template
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
