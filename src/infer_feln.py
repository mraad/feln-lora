"""Score a LoRA checkpoint or merged model on labeled records, export the merged bundle,
or query either. Runs on the GPU box inside the AutoModel venv (torch, transformers, peft).
GGUFs are scored through llama-server with `src.edge_client --records` instead.

Examples:
  python -m src.infer_feln --model checkpoints/epoch_1_step_443/model --schema data/Layers.json --records data/val.json --output eval-val.json
  python -m src.infer_feln --model checkpoints/epoch_1_step_443/model --schema data/Layers.json --export exports/nemotron-4b-step443
  python -m src.infer_feln --model exports/nemotron-4b-step443 --schema data/Layers.json --text 'Show gas wells'
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from feln import FELN, FELNCompare

from .edge_client import output_schema
from .feln_data import Schema, fingerprint, write_json
from .prompt import make_prompt, parse_feln_json


@dataclass
class EvalResult:
    exact: list[bool] = field(default_factory=list)
    raw: list[bool] = field(default_factory=list)
    structural: list[float] = field(default_factory=list)
    invalid: int = 0
    predictions: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def score_prediction(result: EvalResult, record: dict, generated: str, schema: Schema):
    """Exact = feln's strict comparator (normalized SQL); raw = byte-identical meta.

    A prediction that parses but fails the catalog (unknown field, bad enum code) counts
    as invalid: the deployment client would reject it the same way.
    """
    expected = FELN.model_validate(record["meta"])
    predicted = parse_feln_json(generated)
    if predicted is not None:
        try:
            schema.validate(predicted.model_dump())
        except ValueError:
            predicted = None
    result.invalid += predicted is None
    result.exact.append(predicted is not None and predicted.same(expected))
    result.raw.append(predicted is not None and predicted.model_dump() == expected.model_dump())
    result.structural.append(
        FELNCompare.structural(predicted, expected) if predicted is not None else 0.0
    )
    result.predictions.append(
        {
            "text": record["text"],
            "expected": record["meta"],
            "generated": generated,
            "exact": result.exact[-1],
            "origin": record.get("origin", "unknown"),
        }
    )


def result_summary(result: EvalResult) -> dict:
    n = len(result.exact)
    slices = {}
    for name, indices in {
        **{
            f"{count}_layers": [
                i for i, p in enumerate(result.predictions) if len(p["expected"]["layers"]) == count
            ]
            for count in (1, 2, 3)
        },
        "nonempty_relations": [
            i for i, p in enumerate(result.predictions) if any(p["expected"]["relations"])
        ],
    }.items():
        if indices:
            slices[name] = {
                "n": len(indices),
                "exact_match": sum(result.exact[i] for i in indices) / len(indices),
            }
    return {
        "n": n,
        "exact_match": sum(result.exact) / n if n else 0.0,
        "raw_exact_match": sum(result.raw) / n if n else 0.0,
        "mean_structural": sum(result.structural) / n if n else 0.0,
        "parse_error_rate": result.invalid / n if n else 0.0,
        "elapsed_seconds": result.elapsed_seconds,
        "slices": slices,
    }


def progress(result: EvalResult, total: int):
    print(
        f"{len(result.exact)}/{total} exact={sum(result.exact) / len(result.exact):.3f}", flush=True
    )


def evaluate(model, tokenizer, records, max_new_tokens, schema, batch_size) -> EvalResult:
    import torch

    result = EvalResult()
    started = time.monotonic()
    tokenizer.padding_side = "left"
    context = schema.context()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            prompts = [make_prompt(r["text"], tokenizer, context) for r in batch]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            generated = tokenizer.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for record, text in zip(batch, generated):
                score_prediction(result, record, text, schema)
            progress(result, len(records))
    result.elapsed_seconds = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--text")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument(
        "--compile-sql",
        action="store_true",
        help="Apply schema typing/casing; report raw model accuracy separately",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not (args.text or args.records or args.export):
        parser.error("Provide --text, --records, or --export")
    if args.records and not args.output:
        parser.error("--records requires --output")

    schema = Schema(args.schema)
    records = json.loads(args.records.read_text()) if args.records else []
    for record in records:
        schema.validate(record["meta"])
    provenance = {
        "schema_sha256": fingerprint(args.schema),
        "model": str(args.model),
        "records_sha256": fingerprint(args.records) if args.records else None,
        "compiled_sql": args.compile_sql,
    }
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # same as training
    if (args.model / "adapter_config.json").exists():
        config = PeftConfig.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            revision=config.revision,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        model, adapter = PeftModel.from_pretrained(base, args.model), True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": 0}
        )
        adapter = False
    model.eval()
    if args.export:
        if args.export.exists() and any(args.export.iterdir()):
            raise SystemExit("Export directory is not empty; choose a new path")
        if adapter:
            # BF16 merging can round away small LoRA updates. Accumulate in
            # FP32, then store FP16 for GGUF export's finer weight precision.
            model = model.float().merge_and_unload().to(torch.float16)
        generation = model.generation_config
        if generation is not None and not generation.do_sample:
            # Some bases ship sampling knobs without do_sample; transformers 5
            # refuses to save that, and FELN decoding is greedy anyway.
            generation.temperature = None
            generation.top_p = None
            generation.top_k = None
        model.save_pretrained(args.export, safe_serialization=True)
        tokenizer.save_pretrained(args.export)
        sentinel = "FELN_QUERY_PLACEHOLDER_9b41"
        prefix, suffix = make_prompt(sentinel, tokenizer, schema.context()).split(sentinel)
        write_json(
            args.export / "inference_config.json",
            {
                "prompt_prefix": prefix,
                "prompt_suffix": suffix,
                "max_new_tokens": args.max_new_tokens,
                "json_schema": output_schema(list(schema.layers)),
            },
        )
        write_json(args.export / "Layers.json", schema.raw)
        write_json(
            args.export / "export_manifest.json",
            {
                **provenance,
                "merge_dtype": "float32" if adapter else None,
                "stored_dtype": str(model.dtype),
            },
        )
        print(f"Exported merged {model.dtype} weights to {args.export}")
    if records:
        result = evaluate(model, tokenizer, records, args.max_new_tokens, schema, args.batch_size)
        raw_summary = result_summary(result)
        if args.compile_sql:
            raw_result = result
            result = EvalResult()
            for record, prediction in zip(records, raw_result.predictions):
                expected = schema.compile(record["meta"])
                try:
                    parsed = parse_feln_json(prediction["generated"])
                    compiled = schema.compile(parsed.model_dump()) if parsed else None
                except ValueError:
                    compiled = None
                score_prediction(result, {**record, "meta": expected}, json.dumps(compiled), schema)
                result.predictions[-1]["raw_generated"] = prediction["generated"]
                result.predictions[-1]["raw_exact"] = prediction["exact"]
                result.predictions[-1]["original_expected"] = record["meta"]
            result.elapsed_seconds = raw_result.elapsed_seconds
        summary = {**result_summary(result), **provenance}
        if args.compile_sql:
            summary["raw_model"] = raw_summary
        write_json(args.output.with_suffix(".json"), summary)
        write_json(
            args.output.parent / (args.output.name + "-predictions.json"), result.predictions
        )
        print(json.dumps(summary, indent=2))
    if args.text:
        inputs = tokenizer(
            make_prompt(args.text, tokenizer, schema.context()),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        predicted = parse_feln_json(generated)
        if predicted is None:
            raise SystemExit(
                "Model did not produce valid FELN; clarification or review is required"
            )
        output = (
            schema.compile(predicted.model_dump())
            if args.compile_sql
            else schema.validate(predicted.model_dump()).model_dump()
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
