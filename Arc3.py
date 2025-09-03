"""Minimal SFT training example with TRL."""

from dataclasses import dataclass

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


@dataclass
class ModelArguments:
    """Configuration for the model to fine-tune."""

    model_id: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class DataArguments:
    """Dataset location and options."""

    dataset_path: str = "HPAI-BSC/medqa-cot-llama31"


def to_messages(example: dict) -> dict:
    """Convert raw dataset fields into chat messages."""

    return {
        "messages": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def main() -> None:
    # ------------------------------------ Load dataset ------------------------------------ #
    ds = load_dataset(DataArguments.dataset_path)
    ds = ds["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = ds["train"].map(
        to_messages, remove_columns=["system_prompt", "question", "response"]
    )
    test_ds = ds["test"].map(
        to_messages, remove_columns=["system_prompt", "question", "response"]
    )

    # ------------------------------------ Load model ------------------------------------ #
    model = AutoModelForCausalLM.from_pretrained(
        ModelArguments.model_id, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(ModelArguments.model_id)

    # ------------------------------------ Formatting ------------------------------------ #
    def formatting_prompts_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # ------------------------------------ SFT trainer ------------------------------------ #
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    # Example rendering of a test record; remove or adapt as needed.
    tokenizer.apply_chat_template(
        test_ds[0]["messages"], tokenize=False, add_generation_prompt=False
    )

    # Uncomment the line below to run training in an environment with adequate resources.
    # trainer.train()


if __name__ == "__main__":
    main()

