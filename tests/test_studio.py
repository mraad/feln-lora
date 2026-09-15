"""FELN Studio server contract; no llama-server required."""

import json
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from src import studio

LAYERS = {
    "layers": [
        {
            "name": "Wells",
            "columns": [
                {
                    "name": "content_type",
                    "dtype": "SmallInteger",
                    "keyval": {"2": "Gas"},
                    "values": ["2"],
                    "hints": [],
                }
            ],
        }
    ]
}
GOLD = {"layers": ["Wells"], "where": ["content_type = 2"], "relations": []}


class FakeRuntime(studio.LlamaFELN):
    def __init__(self, bundle, url):
        super().__init__(bundle, url)
        self.prefixes = []

    def generate(self, text, prompt_prefix=None):
        self.prefixes.append(prompt_prefix)
        if text == "fail":
            raise ValueError("Model reached its output limit")
        # LlamaFELN returns schema-compiled FELN; mirror that contract.
        return {
            "feln": {
                "layers": ["Wells"],
                "where": ['"content_type" = CAST(2 AS SMALLINT)'],
                "relations": [],
            },
            "raw": '{"layers": ["wells"], "where": ["content_type = \'Gas\'"], "relations": []}',
            "backend": "fake",
            "seconds": 0.1,
            "first_token_seconds": 0.05,
            "prompt_tokens": 10,
            "generated_tokens": 5,
        }


class StudioChecks(unittest.TestCase):
    def test_api_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            bundle.mkdir()
            (bundle / "Layers.json").write_text(json.dumps(LAYERS))
            (bundle / "inference_config.json").write_text(
                json.dumps({"prompt_prefix": "P:", "prompt_suffix": ":S", "max_new_tokens": 8})
            )
            records = Path(directory) / "gold.json"
            records.write_text(json.dumps([{"text": "Show  gas wells", "meta": GOLD}]))
            app = studio.Studio({"fake": (bundle, "http://127.0.0.1:1")}, [records])
            fake = app.models["fake"]["runtime"] = FakeRuntime(bundle, "http://127.0.0.1:1")
            with ThreadingHTTPServer(("127.0.0.1", 0), studio.handler_for(app)) as server:
                Thread(target=server.serve_forever, daemon=True).start()
                port = server.server_port

                def call(path, data=None, headers=None):
                    connection = HTTPConnection("127.0.0.1", port)
                    connection.request(
                        "GET" if data is None else "POST",
                        path,
                        None if data is None else json.dumps(data),
                        {"Content-Type": "application/json", **(headers or {})},
                    )
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())

                status, config = call("/api/config")
                self.assertEqual(status, 200)
                self.assertEqual(config["models"][0]["prompt_prefix"], "P:")
                self.assertEqual(config["questions"], ["show gas wells"])

                status, result = call(
                    "/api/generate", {"query": "show gas   wells", "model": "fake"}
                )
                self.assertEqual(status, 200)
                self.assertTrue(result["valid"])
                self.assertTrue(result["exact"])
                # Gold is compiled before comparison: 2 becomes a typed cast.
                self.assertIn("CAST(2 AS SMALLINT)", result["expected"]["where"][0])
                self.assertIn("'Gas'", result["raw"])
                self.assertEqual(fake.prefixes[-1], None)

                status, result = call(
                    "/api/generate",
                    {
                        "query": "Unknown question",
                        "model": "fake",
                        "prompt_prefix": "X:",
                    },
                )
                self.assertEqual(status, 200)
                self.assertIsNone(result["expected"])
                self.assertEqual(fake.prefixes[-1], "X:")

                status, result = call("/api/generate", {"query": "fail", "model": "fake"})
                self.assertEqual((status, result["valid"]), (200, False))
                self.assertIn("output limit", result["error"])

                for body in (
                    {"query": "x", "model": "missing"},
                    {"query": "", "model": "fake"},
                    {"query": "x", "model": "fake", "prompt_prefix": ""},
                ):
                    self.assertEqual(call("/api/generate", body)[0], 400)
                self.assertEqual(
                    call(
                        "/api/generate",
                        {"query": "x", "model": "fake"},
                        {"Origin": "http://evil"},
                    )[0],
                    403,
                )
                self.assertEqual(call("/api/config", headers={"Host": "example.com"})[0], 403)
                self.assertEqual(call("/nope")[0], 404)
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
