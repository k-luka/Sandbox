from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from datasets import load_dataset

# ------------------------------------ Argument Parsing ------------------------------------ #


@dataclass
class ModelArguments:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"

@dataclass
class TrainingArguments:
    dataset_path: str = "HPAI-BSC/medqa-cot-llama31"


# ------------------------------------ Load dataset ------------------------------------ #
ds = load_dataset(TrainingArguments.dataset_path)
ds = ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
test_ds = ds["test"]

# ------------------------------------ Load model ------------------------------------ #
model = AutoModelForCausalLM.from_pretrained(ModelArguments.model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(ModelArguments.model_id)

# ------------------------------------ Map ------------------------------------ #
def to_messages(example):
    return {
        "messages": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }

train_ds = train_ds.map(to_messages, remove_columns=["system_prompt", "question", "response"])
test_ds = test_ds.map(to_messages, remove_columns=["system_prompt", "question", "response"])

rendered_train_ds = tokenizer.apply_chat_template(
    train_ds[0]["messages"],
    tokenize=False,
    add_generation_prompt=False
)
print(rendered_train_ds[:500])

# TODO:
rendered_test_ds = tokenizer.apply_chat_template(
    test_ds[0]["messages"],
    tokenize=False,
    add_generation_prompt=False
)


