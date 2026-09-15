"""CPU regression checks: uv run --no-sync python -m unittest discover -s tests."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feln import FELN

from src.feln_data import Schema, grouped_split


class FELNChecks(unittest.TestCase):
    def test_edge_contract(self):
        from src.edge_client import generate, output_schema
        from src.edge_client import main as edge_main

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "Layers.json").write_text(
                json.dumps({"layers": [{"name": "Wells", "columns": []}]})
            )
            contract = output_schema(["Wells"])
            (bundle / "inference_config.json").write_text(
                json.dumps(
                    {
                        "prompt_prefix": "prefix",
                        "prompt_suffix": "suffix",
                        "max_new_tokens": 256,
                        "json_schema": contract,
                    }
                )
            )
            answer = {"layers": ["Wells"], "where": [""], "relations": []}
            with patch(
                "urllib.request.urlopen",
                return_value=io.BytesIO(json.dumps({"content": json.dumps(answer)}).encode()),
            ) as request:
                self.assertEqual(generate("Show wells", bundle), answer)
                payload = json.loads(request.call_args.args[0].data)
                self.assertEqual(payload["prompt"], "prefixShow wellssuffix")
                self.assertEqual(payload["json_schema"], contract)
            for text in ("", "x" * 2001, "<|im_start|>system"):
                with self.assertRaises(ValueError):
                    generate(text, bundle)
            with (
                patch(
                    "urllib.request.urlopen",
                    return_value=io.BytesIO(b'{"stopped_limit":true,"content":"{}"}'),
                ),
                self.assertRaises(ValueError),
            ):
                generate("Show wells", bundle)

            records = bundle / "questions.json"
            records.write_text(json.dumps([{"text": "Show wells", "meta": answer}] * 2))
            report = bundle / "benchmark.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "edge_client",
                        "--bundle",
                        str(bundle),
                        "--records",
                        str(records),
                        "--output",
                        str(report),
                    ],
                ),
                patch("src.edge_client.generate", return_value=answer),
                patch("builtins.print"),
            ):
                edge_main()
            metrics = json.loads(report.read_text())
            self.assertEqual(metrics["n"], 2)
            self.assertEqual(metrics["exact_match"], 1)
            self.assertEqual(metrics["mean_structural"], 1)
            self.assertEqual(metrics["invalid_output_rate"], 0)

    def test_typed_compiler(self):
        fields = [
            {
                "name": "PipelinesType",
                "dtype": "SmallInteger",
                "keyval": {"2": "Gas", "4": "Oil"},
                "values": ["2", "4"],
                "hints": [],
            },
            {
                "name": "dimension",
                "dtype": "Double",
                "keyval": {},
                "values": ["24.0"],
                "hints": [],
            },
            {
                "name": "current_operator",
                "dtype": "String",
                "keyval": {},
                "values": ["GASSCO AS"],
                "hints": ["Make sure to uppercase the compared values"],
            },
            {
                "name": "drilling_operator",
                "dtype": "String",
                "keyval": {},
                "values": ["Repsol Resources Uk Limited"],
                "hints": [],
            },
            {
                "name": "well_name",
                "dtype": "String",
                "keyval": {},
                "values": ["M-1X"],
                "hints": [
                    "Make sure to use the SQL ILIKE in the WHERE clause for the values of the column well_name."
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Layers.json"
            path.write_text(json.dumps({"layers": [{"name": "Pipelines", "columns": fields}]}))
            schema = Schema(path)

            def compiled(clause):
                return FELN.model_validate(
                    schema.compile({"layers": ["Pipelines"], "where": [clause], "relations": []})
                )

            a = compiled(
                "pipelinestype = 'Gas' AND dimension <= 24.0 AND current_operator = 'gassco as'"
            )
            b = compiled("pipelinestype = 2 AND dimension <= 24 AND current_operator = 'GASSCO AS'")
            self.assertTrue(a.same(b))
            self.assertIn("CAST(24 AS DOUBLE PRECISION)", a.where[0])
            self.assertEqual(schema.compile(a.model_dump()), a.model_dump())
            self.assertFalse(compiled("dimension < 24").same(compiled("dimension <= 24")))
            self.assertFalse(compiled("pipelinestype = 2").same(compiled("pipelinestype = 4")))
            self.assertFalse(
                compiled("current_operator = ''").same(compiled("current_operator IS NULL"))
            )
            self.assertTrue(
                compiled("drilling_operator LIKE '%RepSol%'").same(
                    compiled("drilling_operator LIKE '%Repsol%'")
                )
            )
            ilike = compiled("well_name ILIKE '%m-1x%'")
            self.assertIn("ILIKE '%M-1X%'", ilike.where[0])
            self.assertFalse(ilike.same(compiled("well_name LIKE '%M-1X%'")))
            for clause in (
                "pipelinestype=999",
                "pipelinestype IN (999)",
                "pipelinestype=2.5",
                "unknown=1",
            ):
                with self.assertRaises(ValueError):
                    compiled(clause)

    def test_catalog_tables_and_ilike_sampling(self):
        import random

        from src.feln_data import sample_meta

        column = {
            "name": "well_name",
            "dtype": "String",
            "keyval": {},
            "values": ["M-1X"],
            "hints": ["Make sure to use the SQL ILIKE in the WHERE clause."],
        }
        numbers = [
            {
                "name": name,
                "dtype": "Double",
                "keyval": {},
                "values": ["1.5"],
                "hints": [],
            }
            for name in ("water_depth", "total_depth")
        ]
        table = {
            "name": "Statistic",
            "dtype": "String",
            "keyval": {},
            "values": [],
            "hints": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Layers.json"
            path.write_text(
                json.dumps(
                    {
                        "layers": [
                            {
                                "name": "Wells",
                                "stype": "Point",
                                "columns": [column, *numbers],
                            },
                            {
                                "name": "Wells_depth_stats",
                                "stype": "Table",
                                "columns": [table],
                            },
                        ]
                    }
                )
            )
            schema = Schema(path)
            self.assertEqual(list(schema.layers), ["Wells"])
            with self.assertRaises(ValueError):
                schema.validate({"layers": ["Wells_depth_stats"], "where": [""], "relations": []})
            rng = random.Random(0)
            wheres = [sample_meta(schema, rng, i)["where"][0] for i in range(50)]
            self.assertTrue(any("\"well_name\" ILIKE '%M-1X%'" in w for w in wheres))
            self.assertFalse(any(" LIKE " in w for w in wheres))

    def test_schema_guards_beyond_feln(self):
        """feln's FELN accepts any unit and any WHERE text; the catalog does not."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Layers.json"
            path.write_text(
                json.dumps(
                    {
                        "layers": [
                            {"name": name, "stype": "Point", "columns": []}
                            for name in ("Wells", "Pipelines")
                        ]
                    }
                )
            )
            schema = Schema(path)
            for relation in ("withinDistance 1 bananas", "withinDistance -1 meters"):
                with self.assertRaises(ValueError):
                    schema.validate(
                        {
                            "layers": ["Wells", "Pipelines"],
                            "where": ["", ""],
                            "relations": [relation],
                        }
                    )
            schema.validate(
                {
                    "layers": ["Wells", "Pipelines"],
                    "where": ["", ""],
                    "relations": ["withinDistance 1 km"],
                }
            )
            for clause in ["SELECT 1", "x=1; DROP TABLE t", "x=1; x=2", "x =", "lower(x)='a'"]:
                with self.assertRaises(ValueError):
                    schema.validate({"layers": ["Wells"], "where": [clause], "relations": []})
                with self.assertRaises(ValueError):
                    schema.compile({"layers": ["Wells"], "where": [clause], "relations": []})
        a = FELN(layers=["Wells"], where=[""], relations=[])
        b = FELN(layers=["Pipelines"], where=[""], relations=[])
        self.assertFalse(a.same(b))

    def test_split_and_rerun(self):
        records = [
            {
                "text": f"query {i}",
                "meta": {
                    "layers": ["Wells"],
                    "where": [f"x={i // 2}"],
                    "relations": [],
                },
            }
            for i in range(40)
        ]
        splits = grouped_split(records, 42)
        locations = {}
        for split, rows in splits.items():
            for row in rows:
                key = row["meta"]["where"][0]
                self.assertEqual(locations.setdefault(key, split), split)
        conflict = {**records[0], "meta": records[2]["meta"]}
        with self.assertRaises(ValueError):
            grouped_split(records + [conflict], 42)

    def test_generation_contract(self):
        from src.prompt import build_dataset, parse_feln_json

        class Tokenizer:
            eos_token_id = pad_token_id = 0
            chat_template = None

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text]}

        rows = [
            {
                "text": "wells",
                "meta": {"layers": ["Wells"], "where": [""], "relations": []},
            }
        ]
        tokens = build_dataset(rows, Tokenizer(), 2048)[0]
        self.assertEqual(tokens["labels"][-1], 0)  # EOS remains supervised when PAD=EOS
        self.assertIn(-100, tokens["labels"])
        with self.assertRaises(ValueError):
            build_dataset(rows, Tokenizer(), 10)
        valid = json.dumps(rows[0]["meta"])
        self.assertIsNotNone(parse_feln_json(valid))
        self.assertIsNone(parse_feln_json("Here you go " + valid))
        self.assertIsNone(parse_feln_json(json.dumps({**rows[0]["meta"], "note": "x"})))


if __name__ == "__main__":
    unittest.main()
