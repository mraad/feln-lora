"""Catalog schema (validate/compile FELN against Layers.json), synthetic question
generation, and leakage-resistant splits. Data is built by scripts/prepare_northsea.py.
Generated questions are synthetic, not human-validated or real user logs.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import sqlglot
import sqlglot.errors
from feln import FELN, normalize_where, parse_relation, to_meters
from sqlglot import exp

LIKE_TYPES = (exp.Like, exp.ILike)
COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, *LIKE_TYPES)


def parse_predicate(clause: str):
    """One SQL predicate as an AST; malformed SQL is a ValueError like every other rejection."""
    try:
        expressions = sqlglot.parse(clause)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"Invalid SQL predicate: {clause}") from exc
    if len(expressions) != 1 or not isinstance(
        expressions[0], (exp.Predicate, exp.Connector, exp.Not, exp.Paren)
    ):
        raise ValueError(f"Expected one SQL predicate: {clause}")
    return expressions[0]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Schema:
    def __init__(self, path: Path):
        self.path = path
        self.raw = json.loads(path.read_text())
        # Standalone tables carry no geometry, so FELN can never target them.
        self.layers = {
            layer["name"]: layer for layer in self.raw["layers"] if layer.get("stype") != "Table"
        }
        self.columns = {
            name: {c["name"].lower(): c for c in layer["columns"]}
            for name, layer in self.layers.items()
        }
        # The catalog says how a text column is matched ("use the SQL LIKE/ILIKE").
        self.like = {
            (name, column["name"]): match.group(1)
            for name, columns in self.columns.items()
            for column in columns.values()
            for hint in column["hints"]
            if (match := re.search(r"use the SQL (I?LIKE)\b", hint))
        }

    def validate(self, meta: dict) -> FELN:
        feln = FELN.model_validate(meta)
        for rel in feln.relations:
            relation = parse_relation(rel)  # feln accepts any unit; the SQL compiler does not
            if relation.unit and (
                relation.distance < 0 or to_meters(relation.distance, relation.unit) is None
            ):
                raise ValueError(f"Unsupported distance relation: {rel}")
        for name, clause in zip(feln.layers, feln.where):
            if name not in self.layers:
                raise ValueError(f"Unknown layer {name}")
            if not clause:
                continue
            tree = parse_predicate(clause)
            if any(
                isinstance(node, exp.Func) and not isinstance(node, (exp.Cast, exp.And, exp.Or))
                for node in tree.walk()
            ):
                raise ValueError(
                    "SQL functions other than type casts are outside the supported filter grammar"
                )
            for column in tree.find_all(exp.Column):
                if column.table or column.name.lower() not in self.columns[name]:
                    raise ValueError(f"Unknown field {name}.{column.sql()}")
            # Reject hallucinated enum codes without treating sampled string values
            # as exhaustive domains: new facilities/operators are legitimate.
            checks = []
            for predicate in tree.walk():
                if isinstance(predicate, COMPARISONS):
                    checks.append((predicate, predicate.expression))
                elif isinstance(predicate, exp.In):
                    checks.extend((predicate, value) for value in predicate.expressions)
                elif isinstance(predicate, exp.Between):
                    checks.extend((predicate, predicate.args[key]) for key in ("low", "high"))
            for predicate, value in checks:
                column = predicate.this
                if isinstance(column, exp.Column):
                    spec = self.columns[name][column.name.lower()]
                    if isinstance(value, exp.Cast):
                        value = value.this
                    if (
                        spec["keyval"]
                        and spec["dtype"] != "String"
                        and value.sql() not in spec["keyval"]
                    ):
                        raise ValueError(f"Invalid enum code: {predicate.sql()}")
                    if isinstance(value, exp.Literal):
                        string_type = spec["dtype"] in {"String", "Date"}
                        if value.is_string != string_type:
                            raise ValueError(f"Wrong literal type: {predicate.sql()}")
                        if spec["keyval"] and string_type:
                            known = set(spec["keyval"]) | set(spec["values"])
                            if value.this and value.this not in known:
                                raise ValueError(f"Unknown coded value: {predicate.sql()}")
        return feln

    def compile(self, meta: dict) -> dict:
        """Emit schema-typed SQL, preserving predicates and spatial joins.

        Resolve exact enum labels/codes, explicit uppercase policies, and uniquely
        matched spelling in the supplied value catalog. Unknown names are never
        invented. Keep raw-model accuracy separate from compiled-output accuracy.
        """
        result = FELN.model_validate(meta).model_dump()
        names = {name.casefold(): name for name in self.layers}
        types = {
            "SmallInteger": "SMALLINT",
            "Integer": "INTEGER",
            "Double": "DOUBLE PRECISION",
        }
        for i, name in enumerate(result["layers"]):
            if name.casefold() not in names:
                raise ValueError(f"Unknown layer {name}")
            name = result["layers"][i] = names[name.casefold()]
            if not result["where"][i]:
                continue
            tree = parse_predicate(result["where"][i])
            for column in tree.find_all(exp.Column):
                if column.table or column.name.lower() not in self.columns[name]:
                    raise ValueError(f"Unknown field {name}.{column.sql()}")
                column.set(
                    "this",
                    exp.to_identifier(self.columns[name][column.name.lower()]["name"], quoted=True),
                )
            for predicate in list(tree.walk()):
                if not isinstance(predicate, (*COMPARISONS, exp.In, exp.Between)):
                    continue
                if not isinstance(predicate.this, exp.Column):
                    continue
                spec = self.columns[name][predicate.this.name.lower()]
                if isinstance(predicate, exp.In):
                    values = list(predicate.expressions)
                elif isinstance(predicate, exp.Between):
                    values = [predicate.args["low"], predicate.args["high"]]
                else:
                    values = [predicate.expression]
                for value in values:
                    # Preserve existing casts: replacing a cast could change truncation.
                    if isinstance(value, exp.Cast):
                        continue
                    literal_node = value.this if isinstance(value, exp.Neg) else value
                    if not isinstance(literal_node, exp.Literal):
                        continue
                    text = literal_node.this if literal_node.is_string else value.sql()
                    mapped = {
                        key
                        for key, label in spec["keyval"].items()
                        if str(text).casefold() in (key.casefold(), label.casefold())
                    }
                    if len(mapped) == 1:
                        text = mapped.pop()
                    if spec["dtype"] in types:
                        try:
                            number = Decimal(text)
                        except InvalidOperation as exc:
                            raise ValueError(f"Invalid numeric value: {text}") from exc
                        if not number.is_finite() or abs(number.adjusted()) > 308:
                            raise ValueError("Numeric value is outside supported range")
                        if spec["dtype"] != "Double":
                            bits = 16 if spec["dtype"] == "SmallInteger" else 32
                            if number != number.to_integral_value() or not -(
                                2 ** (bits - 1)
                            ) <= number < 2 ** (bits - 1):
                                raise ValueError(
                                    "Integer value is fractional or overflows its schema type"
                                )
                        text = format(number, "f")
                        text = text.rstrip("0").rstrip(".") if "." in text else text
                        value.replace(
                            exp.Cast(
                                this=sqlglot.parse_one(text),
                                to=exp.DataType.build(types[spec["dtype"]]),
                            )
                        )
                    elif literal_node.is_string:
                        if any("uppercase the compared values" in hint for hint in spec["hints"]):
                            text = text.upper()
                        elif not spec["keyval"]:
                            needle = text.strip("%") if isinstance(predicate, LIKE_TYPES) else text
                            if needle and not any(c in needle for c in "%_"):
                                matches = {
                                    match.group()
                                    for candidate in spec["values"]
                                    for match in re.finditer(
                                        re.escape(needle), str(candidate), re.IGNORECASE
                                    )
                                    if isinstance(predicate, LIKE_TYPES)
                                    or match.group().casefold() == str(candidate).casefold()
                                }
                                if len(matches) == 1:
                                    text = text.replace(needle, matches.pop())
                        value.replace(exp.Literal.string(text))
            result["where"][i] = tree.sql(dialect="postgres")
        self.validate(result)
        return result

    def context(self) -> str:
        lines = []
        for name, layer in self.layers.items():
            fields = []
            for c in layer["columns"]:
                field = c["name"].lower()
                if c["keyval"]:
                    field += "=" + json.dumps(
                        c["keyval"], ensure_ascii=False, separators=(",", ":")
                    )
                fields.append(field)
            lines.append(f"{name}: " + "; ".join(fields))
        return "\n".join(lines)


FIELD_PHRASES = {
    "current_operator": "operated by {v}",
    "from_facility": "starting at {v}",
    "to_facility": "ending at {v}",
    "current_phase": "whose current phase is {v}",
    "country": "in {v}",
    "source": "reported by {v}",
    "field_name": "in the {v} field",
    "field_label": "with the field label {v}",
    "purpose": "used for {v}",
    "status": "with status {v}",
    "production_facility": "using production facility {v}",
    "field_current_activity_status": "whose field activity status is {v}",
    "discovery_current_activity_sta": "whose discovery activity status is {v}",
}
BOOL_PHRASES = {
    "formation_tops": "formation tops",
    "geochem_info": "geochemical information",
    "log": "well logs",
    "mud": "mud records",
    "oil_samples": "oil samples",
    "old_wdss": "old WDSS records",
    "paly_slides": "palynology slides",
    "wellbore_history": "wellbore history",
    "npd_papers": "NPD papers",
    "doc_by_licensee": "licensee documents",
    "core_sample": "core samples",
    "cuttings_sample": "cuttings samples",
}


def describe_predicate(node, columns: dict, variant: int = 0) -> str:
    if isinstance(node, exp.Paren):
        return "(" + describe_predicate(node.this, columns, variant) + ")"
    if isinstance(node, (exp.And, exp.Or)):
        join = " and " if isinstance(node, exp.And) else " or "
        left, right = (
            describe_predicate(node.this, columns, variant),
            describe_predicate(node.expression, columns, variant),
        )
        return f"either ({left}) or ({right})" if isinstance(node, exp.Or) else left + join + right
    if isinstance(node, exp.Not) and isinstance(node.this, exp.Is):
        return f"whose {node.this.this.name.replace('_', ' ')} is not NULL"
    if not isinstance(node.this, exp.Column):
        raise ValueError(f"Unsupported predicate {node}")  # noqa: TRY004 - valid AST outside renderer vocabulary
    name = node.this.name.lower()
    spec = columns[name]
    right = node.expression
    if isinstance(right, exp.Cast):
        right = right.this
    value = right.this if isinstance(right, exp.Literal) else right.sql()
    display = spec["keyval"].get(str(value), str(value))
    label = name.replace("_", " ")
    if isinstance(node, exp.Is):
        return f"whose {label} is NULL"
    if isinstance(node, exp.EQ):
        if name in BOOL_PHRASES and str(value) in {"0", "1", "YES", "NO"}:
            return ("with " if str(value) in {"1", "YES"} else "without ") + BOOL_PHRASES[name]
        if value == "":
            return f"whose {label} is an empty string"
        if name in {"pipelinestype", "content_type", "discovery_type"}:
            return f"classified as {display.lower()}"
        if name == "current_phase" and display == "IN SERVICE":
            return "still in service"
        if name == "country" and display == "Netherland":
            display = "the Netherlands"
        return FIELD_PHRASES.get(name, "whose " + label + " is {v}").format(v=display)
    if isinstance(node, LIKE_TYPES):
        text = str(value)
        if text.startswith("%") and text.endswith("%"):
            return f"whose {label} contains {text[1:-1]}"
        if text.endswith("%"):
            return f"whose {label} starts with {text[:-1]}"
        if text.startswith("%"):
            return f"whose {label} ends with {text[1:]}"
        return f"whose {label} matches {text}"
    operators = {
        exp.NEQ: "is not",
        exp.GT: "is greater than",
        exp.GTE: "is at least",
        exp.LT: "is less than",
        exp.LTE: "is at most",
    }
    op = operators[type(node)]
    if spec["dtype"] == "Date":
        op = {
            exp.GT: "is after",
            exp.GTE: "is on or after",
            exp.LT: "is before",
            exp.LTE: "is on or before",
            exp.NEQ: "is not",
        }[type(node)]
    return f"whose {label} {op} {display if value != '' else 'an empty string'}"


def render(meta: dict, schema: Schema, variant: int) -> str:
    def noun(i):
        name = meta["layers"][i]
        clause = meta["where"][i]
        if not clause:
            return name.lower()
        tree = sqlglot.parse_one(clause)
        if variant % 2 == 0:
            terms = list(tree.flatten()) if isinstance(tree, exp.And) else [tree]
            subtype = schema.layers[name]["subtype"].lower()
            for term in terms:
                if (
                    isinstance(term, exp.EQ)
                    and isinstance(term.this, exp.Column)
                    and term.this.name.lower() == subtype
                ):
                    code = term.expression.sql()
                    prefix = schema.columns[name][subtype]["keyval"][code].lower() + " "
                    others = [t for t in terms if t is not term]
                    suffix = " and ".join(
                        describe_predicate(t, schema.columns[name], variant) for t in others
                    )
                    return prefix + name.lower() + (" " + suffix if suffix else "")
        return name.lower() + " " + describe_predicate(tree, schema.columns[name], variant)

    text = noun(0)
    for i, relation in enumerate(meta["relations"], 1):
        parts = relation.split()
        kind = parts[0]
        secondary = noun(i)
        prefix = " that " if i == 1 else " and also "
        if kind in {"withinDistance", "notWithinDistance"}:
            distance, unit = parts[1:]
            if variant % 3 == 1:
                unit = {
                    "kilometers": "km",
                    "meters": "m",
                    "miles": "mi",
                    "feet": "ft",
                }.get(unit, unit)
            phrase = (
                f"are within {distance} {unit} of {secondary}"
                if kind == "withinDistance"
                else f"are more than {distance} {unit} from any {secondary}"
            )
        else:
            phrase = {
                "within": "are within ",
                "inside": "are inside ",
                "contains": "contain ",
                "intersects": "intersect ",
            }[kind] + secondary
        # Explicitly restart the primary subject in three-layer questions, avoiding
        # accidentally describing a secondary-to-secondary relation in English.
        text += prefix + phrase if i == 1 else f"; these {meta['layers'][0].lower()} also " + phrase
    frames = [
        "Show me {q}.",
        "Which {q}?",
        "Can you find {q}?",
        "I need {q} on the map.",
        "Please list {q}.",
        "Find {q}.",
        "I’m looking for {q}.",
        "Bring up {q}.",
        "Could you pull up {q}?",
        "For my review, show {q}.",
    ]
    return frames[variant % len(frames)].format(q=text)


def literal(value: str, dtype: str) -> str:
    if dtype in {"SmallInteger", "Integer", "Double"}:
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def sample_meta(schema: Schema, rng: random.Random, index: int) -> dict:
    names = list(schema.layers)
    rng.shuffle(names)
    count = rng.choices([1, 2, 3], [0.35, 0.45, 0.20])[0]
    names = names[:count]
    clauses = []
    for name in names:
        if rng.random() < 0.10:
            clauses.append("")
            continue
        columns = list(schema.columns[name].values())
        predicates = []
        for column in rng.sample(columns, rng.choices([1, 2, 3], [0.55, 0.35, 0.10])[0]):
            values = list(column["keyval"]) or column["values"]
            if not values:
                continue
            value = rng.choice(values)
            field = column["name"].lower()
            dtype = column["dtype"]
            operator = "="
            if rng.random() < 0.05:
                predicates.append(
                    f'"{field}" IS ' + ("NOT " if rng.random() < 0.5 else "") + "NULL"
                )
                continue
            if dtype in {"Double", "Date"}:
                operator = rng.choice(["=", ">", "<", ">=", "<="])
            elif not column["keyval"] and rng.random() < 0.12:
                operator = "<>"
            like = schema.like.get((name, column["name"]))
            if dtype == "String" and like:
                operator = like
                # Keep the original spelling of database strings.
                value = "%" + str(value) + "%"
            predicates.append(f'"{field}" {operator} {literal(value, dtype)}')
        clauses.append((" OR " if rng.random() < 0.10 else " AND ").join(predicates))
    relations = []
    for name in names[1:]:
        kinds = ["withinDistance", "notWithinDistance", "intersects"]
        if name == "Discoveries":
            kinds += ["within"]
        if names[0] == "Discoveries":
            kinds += ["contains"]
        kind = rng.choice(kinds)
        if "Distance" in kind:
            distance = rng.choice(
                [
                    "0.5",
                    "1",
                    "2",
                    "3",
                    "5",
                    "7.5",
                    "10",
                    "15",
                    "20",
                    "25",
                    "50",
                    "100",
                    "500",
                    "1000",
                ]
            )
            kind += f" {distance} {rng.choice(['kilometers', 'meters', 'miles', 'feet'])}"
        relations.append(kind)
    return {"layers": names, "where": clauses, "relations": relations}


def target_key(meta: dict) -> str:
    return json.dumps(
        {**meta, "where": [normalize_where(w) for w in meta["where"]]}, sort_keys=True
    )


def grouped_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """Identical targets and their paraphrases stay together across all files."""
    groups = defaultdict(list)
    texts = {}
    for record in records:
        key = target_key(record["meta"])
        text = " ".join(record["text"].casefold().split())
        if text in texts:
            if texts[text] != key:
                raise ValueError(f"Conflicting labels: {record['text']}")
            continue
        texts[text] = key
        groups[key].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    result = {name: [] for name in ("train", "val", "test")}
    for i, key in enumerate(keys):
        split = "train" if i < int(n * 0.8) else "val" if i < int(n * 0.9) else "test"
        result[split].extend(groups[key])
    return result
