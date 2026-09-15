"""Read-only, parameterized FELN execution. Never execute model SQL verbatim."""

import json
import math
import threading
import time
from pathlib import Path

import duckdb
import sqlglot
from feln import to_meters
from sqlglot import exp

from .feln_data import Schema

DATABASE = Path("/Users/mraad/Documents/ArcGIS/Projects/NorthSea/NorthSea.ddb")
DISTANCE_NOTE = "Distances use WGS84 / UTM zone 32N (EPSG:32632), a regional planar approximation—not geodesic distance."


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def predicate(clause, columns, alias, parameters):
    if not clause:
        return "TRUE"
    trees = sqlglot.parse(clause)
    if len(trees) != 1 or trees[0] is None:
        raise ValueError("Exactly one filter expression is allowed")

    def value(node):
        if isinstance(node, exp.Column) and not node.table and node.name.lower() in columns:
            return f"{alias}.{quote(columns[node.name.lower()]['name'])}"
        if isinstance(node, exp.Literal):
            item = (
                node.this
                if node.is_string
                else float(node.this)
                if any(c in node.this.lower() for c in ".e")
                else int(node.this)
            )
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite numeric literal")
            parameters.append(item)
            return "?"
        if (
            isinstance(node, exp.Neg)
            and isinstance(node.this, exp.Literal)
            and not node.this.is_string
        ):
            return "(-" + value(node.this) + ")"
        if isinstance(node, exp.Cast):
            dtype = node.args["to"].sql(dialect="duckdb")
            if dtype not in {
                "SMALLINT",
                "INT",
                "INTEGER",
                "BIGINT",
                "DOUBLE",
                "FLOAT",
                "DATE",
                "TIMESTAMP",
                "TEXT",
                "VARCHAR",
            }:
                raise ValueError("Unsupported cast")
            return f"CAST({value(node.this)} AS {dtype})"
        raise ValueError("Filter operands must be known fields or literal values")

    def walk(node):
        if isinstance(node, exp.Paren):
            return "(" + walk(node.this) + ")"
        if isinstance(node, (exp.And, exp.Or)):
            return f"({walk(node.this)} {'AND' if isinstance(node, exp.And) else 'OR'} {walk(node.expression)})"
        if isinstance(node, exp.Not):
            return "(NOT " + walk(node.this) + ")"
        operators = {
            exp.EQ: "=",
            exp.NEQ: "<>",
            exp.GT: ">",
            exp.GTE: ">=",
            exp.LT: "<",
            exp.LTE: "<=",
            exp.Like: "LIKE",
            exp.ILike: "ILIKE",
        }
        if type(node) in operators:
            return f"({value(node.this)} {operators[type(node)]} {value(node.expression)})"
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
            return f"({value(node.this)} IS NULL)"
        if isinstance(node, exp.In) and node.expressions and not node.args.get("query"):
            return f"({value(node.this)} IN ({', '.join(value(v) for v in node.expressions)}))"
        if isinstance(node, exp.Between):
            return f"({value(node.this)} BETWEEN {value(node.args['low'])} AND {value(node.args['high'])})"
        raise ValueError(
            "Unsupported filter expression; queries, functions, and SQL statements are forbidden"
        )

    return walk(trees[0])


class SpatialQuery:
    def __init__(self, database: Path, schema: Schema):
        self.database, self.schema = Path(database), schema
        if not self.database.is_file():
            raise ValueError("NorthSea database is missing")

    def connect(self):
        connection = duckdb.connect(str(self.database), read_only=True)
        try:
            connection.execute("LOAD spatial")
            connection.execute("SET enable_external_access=false")
            connection.execute("SET threads=4")
            connection.execute("SET memory_limit='2GB'")
            if connection.execute("SELECT DISTINCT wkid FROM sp_ref").fetchall() != [(4326,)]:
                raise ValueError("This application requires the verified WGS84 database")
            return connection
        except Exception:
            connection.close()
            raise

    def plan(self, meta, limit=2000):
        # Check raw clauses before Schema.compile's parse_one normalization.
        validated = self.schema.validate(meta)
        for name, clause in zip(validated.layers, validated.where):
            predicate(clause, self.schema.columns[name], "check", [])
        feln = self.schema.compile(meta)
        if not 1 <= limit <= 5000:
            raise ValueError("Result limit must be between 1 and 5000")
        params, ctes = [], []
        distance = any("Distance" in r for r in feln["relations"])
        for i, (name, clause) in enumerate(zip(feln["layers"], feln["where"])):
            table = quote(self.schema.layers[name]["table_name"])
            where = predicate(clause, self.schema.columns[name], "t", params)
            projected = (
                ", ST_Transform(geometry, 'EPSG:4326', 'EPSG:32632', always_xy := true) AS metric_geom"
                if distance
                else ""
            )
            ctes.append(
                f"l{i} AS MATERIALIZED (SELECT t.*{projected} FROM {table} t WHERE ({where}) AND geometry IS NOT NULL AND NOT ST_IsEmpty(geometry))"
            )
        conditions = []
        for i, relation in enumerate(feln["relations"], 1):
            parts = relation.split()
            if not parts:
                raise ValueError("A secondary layer requires an explicit spatial relation")
            kind = parts[0]
            if kind in {"withinDistance", "notWithinDistance"}:
                meters = to_meters(float(parts[1]), parts[2])
                if meters is None or not math.isfinite(meters) or meters < 0 or meters > 2_000_000:
                    raise ValueError("Distance must be between 0 and 2000 km")
                params.append(meters)
                condition = "ST_DWithin(p.metric_geom, s.metric_geom, ?)"
            else:
                function = {
                    "intersects": "ST_Intersects",
                    "within": "ST_Within",
                    "inside": "ST_Within",
                    "contains": "ST_Contains",
                }[kind]
                condition = f"{function}(p.geometry, s.geometry)"
            negation = "NOT " if kind == "notWithinDistance" else ""
            conditions.append(f"{negation}EXISTS (SELECT 1 FROM l{i} s WHERE {condition})")
        name = feln["layers"][0]
        cols = ["OBJECTID"] + [
            v["name"] for v in self.schema.columns[name].values() if v["name"].lower() != "objectid"
        ]
        fields = ", ".join(f"p.{quote(c)}" for c in cols)
        sql = (
            "WITH "
            + ",\n".join(ctes)
            + f"\nSELECT {fields}, ST_AsGeoJSON(p.geometry) AS __geojson FROM l0 p WHERE "
            + (" AND ".join(conditions) or "TRUE")
            + " ORDER BY p.OBJECTID LIMIT ?"
        )
        params.append(limit + 1)
        return feln, sql, params, distance

    def execute(self, meta, limit=2000):
        feln, sql, params, distance = self.plan(meta, limit)
        started = time.monotonic()
        with self.connect() as connection:
            timer = threading.Timer(30, connection.interrupt)
            timer.start()
            try:
                cursor = connection.execute(sql, params)
                names = [v[0] for v in cursor.description]
                rows = cursor.fetchall()
            finally:
                timer.cancel()
                timer.join()
        truncated = len(rows) > limit
        features = []
        for row in rows[:limit]:
            props = dict(zip(names[:-1], row[:-1]))
            for key, value in props.items():
                if hasattr(value, "isoformat"):
                    props[key] = value.isoformat()
                elif isinstance(value, float) and not math.isfinite(value):
                    props[key] = None
            features.append(
                {
                    "type": "Feature",
                    "id": props["OBJECTID"],
                    "properties": props,
                    "geometry": json.loads(row[-1]),
                }
            )
        layer = self.schema.layers[feln["layers"][0]]
        return {
            "feln": feln,
            "geojson": {"type": "FeatureCollection", "features": features},
            "layer": {k: layer[k] for k in ("name", "stype", "display", "subtype")},
            "count": len(features),
            "truncated": truncated,
            "limit": limit,
            "sql": sql,
            "parameters": params,
            "query_seconds": time.monotonic() - started,
            "warnings": [DISTANCE_NOTE] if distance else [],
        }
