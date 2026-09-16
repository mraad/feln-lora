"""Immutable North Sea run inputs from the current ArcGIS catalog, execute-filtered.

Run from the repo root:
  uv run --no-sync python -m scripts.prepare_northsea --output runs/<run>
Gold records are never dropped; LIKE->ILIKE rewrites and execution failures are
recorded in the manifest. Synthetic records that fail to execute are dropped.
"""

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import duckdb

from src.feln_data import (
    Schema,
    fingerprint,
    grouped_split,
    render,
    sample_meta,
    target_key,
    write_json,
)
from src.spatial_query import SpatialQuery

SOURCE = Path.home() / "Documents/ArcGIS/Projects/NorthSea"


def ilike_columns(schema):
    columns = {}
    for (layer, column), operator in schema.like.items():
        if operator == "ILIKE":
            columns.setdefault(layer, []).append(column)
    return columns


def to_ilike(meta, ilike):
    """Textual LIKE->ILIKE on hinted columns only; everything else stays byte-identical."""
    where = []
    for layer, clause in zip(meta["layers"], meta["where"]):
        for column in ilike.get(layer, ()):
            clause = re.sub(
                rf'("?{re.escape(column)}"?\s+(?:NOT\s+)?)LIKE\b',
                r"\1ILIKE",
                clause,
                flags=re.IGNORECASE,
            )
        where.append(clause)
    return {**meta, "where": where}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--generated", type=int, default=3600)
    parser.add_argument("--empty-cap", type=float, default=0.10)
    args = parser.parse_args()
    started = time.monotonic()
    data = args.output / "data"
    data.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    for name in ("FELN.json", "Layers.json"):
        shutil.copy2(args.source / name, data / name)
    schema = Schema(data / "Layers.json")
    excluded = sorted({layer["name"] for layer in schema.raw["layers"]} - set(schema.layers))
    ilike = ilike_columns(schema)
    gold = json.loads((data / "FELN.json").read_text())
    assert len(gold) == 1000
    records, corrections = [], []
    for i, record in enumerate(gold):
        meta = to_ilike(record["meta"], ilike)
        if meta != record["meta"]:
            corrections.append(
                {
                    "source_index": i,
                    "before": record["meta"]["where"],
                    "after": meta["where"],
                }
            )
        records.append(
            {
                "text": record["text"],
                "meta": meta,
                "origin": "corrected_source",
                "source_index": i,
            }
        )
    rng = random.Random(args.seed)
    for i in range(args.generated):
        meta = sample_meta(schema, rng, i)
        records.append(
            {
                "text": render(meta, schema, i),
                "meta": meta,
                "origin": "schema_generated",
            }
        )
    for record in records:
        schema.validate(record["meta"])

    database = SpatialQuery(args.source / "NorthSea.ddb", schema)
    connection = database.connect()
    kept, dropped, gold_failures = [], [], []
    for n, record in enumerate(records, 1):
        try:
            _, sql, params, _ = database.plan(record["meta"], 1)
            record["empty"] = not connection.execute(
                f"SELECT 1 FROM ({sql}) q LIMIT 1", params
            ).fetchall()
        except (ValueError, duckdb.Error) as exc:
            if record["origin"] == "corrected_source":
                gold_failures.append({"source_index": record["source_index"], "error": str(exc)})
                record["empty"] = None
            else:
                dropped.append({**record, "error": str(exc)})
                continue
        kept.append(record)
        if n % 250 == 0:
            print(
                f"{n}/{len(records)} executed, {len(dropped)} dropped, {time.monotonic() - started:.0f}s",
                flush=True,
            )

    synthetic = [r for r in kept if r["origin"] == "schema_generated"]
    empties = [r for r in synthetic if r["empty"]]
    allowed = int(args.empty_cap * len(synthetic))
    over_cap = {id(record) for record in empties[allowed:]}
    dropped += [{**r, "error": "empty result beyond cap"} for r in kept if id(r) in over_cap]
    kept = [r for r in kept if id(r) not in over_cap]
    connection.close()

    splits = grouped_split(kept, args.seed)
    groups = {name: {target_key(r["meta"]) for r in rows} for name, rows in splits.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    for name, rows in splits.items():
        assert rows
        write_json(data / f"{name}.json", rows)
    write_json(data / "dropped.json", dropped)
    manifest = {
        "seed": args.seed,
        "source_records": len(gold),
        "generated_requested": args.generated,
        "catalog_sha256": fingerprint(data / "Layers.json"),
        "source_sha256": fingerprint(data / "FELN.json"),
        "database_sha256": fingerprint(args.source / "NorthSea.ddb"),
        "excluded_table_layers": excluded,
        "ilike_columns": ilike,
        "like_to_ilike_corrections": len(corrections),
        "corrections": corrections,
        "gold_execution_failures": gold_failures,
        "synthetic_dropped": len(dropped),
        "synthetic_empty_kept": len(empties) - len(over_cap),
        "empty_cap": args.empty_cap,
        "gold_empty": sum(1 for r in kept if r["origin"] == "corrected_source" and r["empty"]),
        "counts": {k: len(v) for k, v in splits.items()},
        "sha256": {p.name: fingerprint(p) for p in sorted(data.glob("*.json"))},
        "target_groups_disjoint": True,
        "seconds": round(time.monotonic() - started),
        "limitations": [
            "Synthetic language, not a blind real-user benchmark",
            "Execution probe uses the planar EPSG:32632 SQL compiler, limit 1",
            "Previous challenge is regression-only",
            "Validation-only model selection; test never used for tuning",
        ],
    }
    write_json(data / "manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"corrections", "sha256"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
