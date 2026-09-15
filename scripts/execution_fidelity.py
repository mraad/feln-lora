"""Execution fidelity: do gold and predicted FELN return the same OBJECTIDs?

Reported next to, never instead of, the strict comparator. Run from the repo root:
  uv run --no-sync python -m scripts.execution_fidelity \
    --predictions <predictions.json> --schema <Layers.json> --output <report.json>
"""

import argparse
import json
from pathlib import Path

import duckdb

from src.feln_data import Schema, fingerprint, write_json
from src.prompt import parse_feln_json
from src.spatial_query import DATABASE, SpatialQuery

LIMIT = 5000


def objectids(database, connection, meta):
    """OBJECTIDs only: the planned query's projection is wrapped, not materialized."""
    _, sql, params, _ = database.plan(meta, LIMIT)
    rows = connection.execute(f"SELECT OBJECTID FROM ({sql}) q", params).fetchall()
    return {row[0] for row in rows[:LIMIT]}, len(rows) > LIMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = Schema(args.schema)
    database = SpatialQuery(args.database, schema)
    connection = database.connect()
    rows, counts = (
        [],
        {
            "n": 0,
            "strict_exact": 0,
            "execution_equal": 0,
            "execution_equal_nonempty": 0,
            "execution_equal_strict_mismatch": 0,
            "strict_exact_execution_differs": 0,
            "prediction_invalid": 0,
            "prediction_execution_failed": 0,
            "truncated": 0,
        },
    )
    jaccard, precision, recall = [], [], []
    hits = predicted_total = expected_total = 0
    for prediction in json.loads(args.predictions.read_text()):
        counts["n"] += 1
        counts["strict_exact"] += bool(prediction["exact"])
        expected, expected_truncated = objectids(database, connection, prediction["expected"])
        row = {"text": prediction["text"], "strict_exact": prediction["exact"]}
        predicted = parse_feln_json(prediction["generated"])
        if predicted is None:
            counts["prediction_invalid"] += 1
            row["status"] = "invalid"
        else:
            try:
                got, got_truncated = objectids(database, connection, predicted.model_dump())
            except (ValueError, duckdb.Error) as exc:
                counts["prediction_execution_failed"] += 1
                row["status"] = "execution_failed"
                row["error"] = str(exc)
            else:
                union = expected | got
                common = len(expected & got)
                score = common / len(union) if union else 1.0
                jaccard.append(score)
                # Empty gold and empty prediction: perfect retrieval, not undefined.
                precision.append(common / len(got) if got else float(not expected))
                recall.append(common / len(expected) if expected else float(not got))
                hits += common
                predicted_total += len(got)
                expected_total += len(expected)
                equal = expected == got
                counts["truncated"] += expected_truncated or got_truncated
                counts["execution_equal"] += equal
                counts["execution_equal_nonempty"] += equal and bool(expected)
                counts["execution_equal_strict_mismatch"] += equal and not prediction["exact"]
                counts["strict_exact_execution_differs"] += prediction["exact"] and not equal
                row.update(
                    status="ok",
                    execution_equal=equal,
                    jaccard=score,
                    expected_rows=len(expected),
                    predicted_rows=len(got),
                    predicted=predicted.model_dump(),
                )
        rows.append(row)
    n = counts["n"] or 1
    report = {
        **counts,
        "strict_exact_rate": counts["strict_exact"] / n,
        "execution_equal_rate": counts["execution_equal"] / n,
        "mean_jaccard_over_executed": sum(jaccard) / len(jaccard) if jaccard else 0.0,
        "macro_precision_over_executed": sum(precision) / len(precision) if precision else 0.0,
        "macro_recall_over_executed": sum(recall) / len(recall) if recall else 0.0,
        "micro_precision_rows": hits / predicted_total if predicted_total else 0.0,
        "micro_recall_rows": hits / expected_total if expected_total else 0.0,
        "predictions_sha256": fingerprint(args.predictions),
        "limit": LIMIT,
        "note": "Execution equality is a diagnostic; strict comparator remains the selection metric. "
        "Empty-result questions match trivially; see execution_equal_nonempty.",
        "rows": rows,
    }
    connection.close()
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
