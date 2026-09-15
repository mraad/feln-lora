"""FELN dataset for NeMo AutoModel recipes.

Reuses src.prompt so tokens match the scoring and GGUF prompts byte-for-byte, then
re-packages each example in AutoModel's shifted next-token layout
(see formatting_utils._package_tokenized_example).
"""

from pathlib import Path

from src.feln_data import Schema
from src.prompt import build_dataset, load_split


class FELNDataset:
    def __init__(self, data_dir, split, tokenizer, max_seq_length=2048):
        data_dir = Path(data_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        rows = build_dataset(
            load_split(data_dir, split),
            tokenizer,
            max_seq_length,
            Schema(data_dir / "Layers.json").context(),
        )
        self.rows = [
            {
                "input_ids": row["input_ids"][:-1],
                "labels": row["labels"][1:],
                "attention_mask": row["attention_mask"][:-1],
                "___PAD_TOKEN_IDS___": {
                    "input_ids": tokenizer.pad_token_id,
                    "labels": -100,
                    "attention_mask": 0,
                },
            }
            for row in rows
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.rows[index])
