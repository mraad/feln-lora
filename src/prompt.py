"""The trained prompt contract: text in, FELN JSON out.

Training (scripts/automodel_feln.py), HF scoring/export (src.infer_feln) and the
llama-server bundle (`inference_config.json`) all derive their prompts from here, so
tokens match byte-for-byte across the three.
"""

from __future__ import annotations

import json
from pathlib import Path

from feln import FELN
from pydantic import ValidationError

PROMPT_TEMPLATE = (
    "You convert natural language geospatial queries into FELN JSON.\n"
    'FELN JSON has three keys: "layers" (list of layer names), "where" '
    '(list of SQL WHERE clauses, one per layer), and "relations" (list of '
    "spatial relations from layers[0] to each layers[i+1], never between consecutive layers). "
    "Preserve the requested primary layer and order of spatial filters. Return only JSON. "
    "Use exact schema field names and enum codes. SQL values are case sensitive; "
    "preserve name spelling. Empty string is not NULL.\n\n"
    "Query: {text}\nFELN:"
)


def make_prompt(text: str, tokenizer=None, schema_context: str = "") -> str:
    instruction = PROMPT_TEMPLATE.split("\n\nQuery:")[0]
    if schema_context:
        instruction += "\nSchema:\n" + schema_context
    if tokenizer is not None and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return instruction + "\n\nQuery: " + text + "\nFELN:"


def load_split(data_dir: Path, name: str) -> list[dict]:
    records = json.loads((Path(data_dir) / f"{name}.json").read_text(encoding="utf-8"))
    for record in records:  # fail fast on bad data, not mid-training at first eval
        FELN.model_validate(record["meta"])
    return records


def build_dataset(
    records: list[dict], tokenizer, max_length: int, schema_context: str = ""
) -> list[dict]:
    """Tokenize records into input_ids/labels, masking the prompt portion of labels."""
    rows = []
    for record in records:
        prompt = make_prompt(record["text"], tokenizer, schema_context)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion = tokenizer(
            json.dumps(record["meta"], ensure_ascii=False), add_special_tokens=False
        )["input_ids"] + [tokenizer.eos_token_id]
        ids = prompt_ids + completion
        if len(ids) > max_length:
            raise ValueError(
                f"Example requires {len(ids)} tokens, max_seq_length={max_length}; refusing truncated labels"
            )
        rows.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": [-100] * len(prompt_ids) + completion,
            }
        )
    return rows


def parse_feln_json(text: str) -> FELN | None:
    """Strict JSON-only contract; prose, fenced JSON and extra keys are generation errors."""
    try:
        meta = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict) or set(meta) != {"layers", "where", "relations"}:
        return None
    try:
        return FELN.model_validate(meta)
    except ValidationError:
        return None
