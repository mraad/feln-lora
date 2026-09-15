"""llama-server client: no torch or transformers required.

Start llama-server with the exported GGUF (see src.studio.serve_gguf), then:
python -m src.edge_client --bundle runs/nemotron-mac-20260915/merged --text 'Show gas wells' --url http://127.0.0.1:8092
This validates structure/schema, not whether the model understood the question.
"""

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from feln import FELN, FELNCompare

from .feln_data import Schema, write_json

LLAMA_SERVER_ARGS = ("-c", "2048", "-np", "1", "-ngl", "all", "--host", "127.0.0.1")


def output_schema(layer_names: list[str]) -> dict:
    """Constrain aligned array lengths and relation syntax during decoding."""
    relation_pattern = (
        r"^(intersects|inside|contains|within|"
        r"(withinDistance|notWithinDistance) [0-9]+(\.[0-9]+)? "
        r"(meters|kilometers|feet|miles|yards))?$"
    )
    variants = []
    for count in (1, 2, 3):
        variants.append(
            {
                "type": "object",
                "properties": {
                    "layers": {
                        "type": "array",
                        "items": {"type": "string", "enum": layer_names},
                        "minItems": count,
                        "maxItems": count,
                    },
                    "where": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": count,
                        "maxItems": count,
                    },
                    "relations": {
                        "type": "array",
                        "items": {"type": "string", "pattern": relation_pattern},
                        "minItems": count - 1,
                        "maxItems": count - 1,
                    },
                },
                "required": ["layers", "where", "relations"],
                "additionalProperties": False,
            }
        )
    return {"oneOf": variants}


def validate_text(text: str):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("Provide a nonempty question of at most 2000 characters")
    if any(token in text for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError("Question contains reserved model control tokens")


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as response:
            return b'"ok"' in response.read()
    except (urllib.error.URLError, OSError):
        return False


def compile_raw(schema: Schema, raw: str) -> dict:
    """Schema-compile model output; a rejection keeps the raw text in the message."""
    try:
        return schema.compile(json.loads(raw))
    except ValueError as exc:
        raise ValueError(f"{exc}\nRaw output: {raw}") from exc


def complete(config: dict, text: str, url: str, prompt_prefix: str | None = None) -> dict:
    """One llama-server /completion call with the bundle's prompt and grammar."""
    validate_text(text)
    prefix = config["prompt_prefix"] if prompt_prefix is None else prompt_prefix
    payload = {
        "prompt": prefix + text + config["prompt_suffix"],
        "n_predict": config["max_new_tokens"],
        "temperature": 0,
        "json_schema": config["json_schema"],
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if result.get("truncated") or result.get("stopped_limit"):
        raise ValueError(
            "Model reached its context/output limit; ask a shorter question\n"
            f"Raw output: {result.get('content', '')}"
        )
    return result


def generate(text: str, bundle: Path, url: str = "http://127.0.0.1:8080") -> dict:
    config = json.loads((bundle / "inference_config.json").read_text())
    result = complete(config, text, url)
    return compile_raw(Schema(bundle / "Layers.json"), result["content"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument(
        "--records", type=Path, help="Evaluate labeled questions on the actual device"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.text:
        print(json.dumps(generate(args.text, args.bundle, args.url), ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error("--records requires --output")
    records = json.loads(args.records.read_text())
    if not records:
        parser.error("--records must contain at least one question")
    schema = Schema(args.bundle / "Layers.json")
    expected = [FELN.model_validate(schema.compile(row["meta"])) for row in records]
    predictions, times = [], []
    for row, reference in zip(records, expected):
        started = time.monotonic()
        try:
            actual = generate(row["text"], args.bundle, args.url)
            parsed = FELN.model_validate(actual)
            exact = parsed.same(reference)
            structural = FELNCompare.structural(parsed, reference)
            error = None
        except ValueError as exc:
            actual, exact, structural, error = None, False, 0.0, str(exc)
        times.append(time.monotonic() - started)
        predictions.append(
            {
                "text": row["text"],
                "expected": reference.model_dump(),
                "generated": actual,
                "exact": exact,
                "error": error,
                "structural": structural,
                "seconds": times[-1],
            }
        )
        if len(predictions) % 25 == 0:
            write_json(args.output.with_suffix(".predictions.json"), predictions)
            print(f"{len(predictions)}/{len(records)}", flush=True)
    warm = sorted(times[1:] or times)
    report = {
        "n": len(records),
        "exact_match": sum(p["exact"] for p in predictions) / len(records),
        "invalid_output_rate": sum(p["error"] is not None for p in predictions) / len(records),
        "mean_structural": sum(p["structural"] for p in predictions) / len(records),
        "first_request_seconds": times[0],
        "warm_p50_seconds": statistics.median(warm),
        "warm_p95_seconds": warm[math.ceil(0.95 * len(warm)) - 1],
        "timing_scope": "sequential HTTP round-trip plus validation/compilation; first request excluded from warm statistics",
        "url": args.url,
        "records": str(args.records),
    }
    write_json(args.output.with_suffix(".predictions.json"), predictions)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
