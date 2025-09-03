# Sandbox

This repository explores minimal data preparation and supervised fine-tuning
for chat-style large language models using
[Hugging Face TRL](https://huggingface.co/docs/trl).

## Quickstart

```bash
pip install -r requirements.txt  # include `trl`, `transformers`, and `datasets`
python Arc3.py  # starts a tiny SFT training loop
```

`Arc3.py` downloads the MedQA dataset, formats each example into chat
messages, and wires up a basic `SFTTrainer` configuration. Uncomment the
`trainer.train()` line inside the script to launch actual training once you
have the necessary hardware resources.

