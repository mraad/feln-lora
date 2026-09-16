"""FELN Studio: local playground for the fine-tuned Nemotron-3-Nano-4B (Q8_0 GGUF, Metal).

  uv run --no-sync python -m src.studio
Then open http://127.0.0.1:8766/. The Studio starts its own llama-server (Homebrew build,
port 8092) unless one is already healthy there, and stops it on exit. Other GGUF bundles
served elsewhere can be added with --llama LABEL=BUNDLE=URL.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import cast

from feln import FELN

from .edge_client import LLAMA_SERVER_ARGS, compile_raw, complete, healthy
from .feln_data import Schema

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web/studio"
NEMOTRON = {
    "label": "Nemotron-3-Nano-4B · Q8_0 · 2026-09-14",
    "bundle": ROOT / "runs/nemotron-mac-20260915/merged",
    "gguf": ROOT / "runs/nemotron-mac-20260915/gguf/nemotron-4b-step443-q8_0.gguf",
    "port": 8092,
}


def serve_gguf(gguf: Path, port: int, wait_seconds: int = 120) -> str:
    """Start llama-server for the GGUF unless one already answers on the port."""
    url = f"http://127.0.0.1:{port}"
    if healthy(url):
        return url
    log = gguf.with_suffix(".server.log").open("ab")
    process = subprocess.Popen(
        ["llama-server", "-m", str(gguf), *LLAMA_SERVER_ARGS, "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
    )
    atexit.register(process.terminate)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(1)
        if healthy(url):
            return url
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited; inspect {log.name}")
    raise TimeoutError("llama-server did not become healthy in time")


def key(text: str) -> str:
    return " ".join(text.casefold().split())


class LlamaFELN:
    """The edge client's request, keeping raw output and timings for the UI."""

    def __init__(self, bundle: Path, url: str):
        self.prompt = json.loads((bundle / "inference_config.json").read_text())
        self.schema = Schema(bundle / "Layers.json")
        self.url = url

    def generate(self, text: str, prompt_prefix: str | None = None) -> dict:
        started = time.monotonic()
        try:
            result = complete(self.prompt, text, self.url, prompt_prefix)
        except OSError as exc:
            raise ValueError(f"llama-server unavailable at {self.url}: {exc}") from exc
        raw = result.get("content", "")
        timings = result.get("timings", {})
        return {
            "feln": compile_raw(self.schema, raw),
            "raw": raw,
            "backend": "llama-server",
            "seconds": time.monotonic() - started,
            "first_token_seconds": timings.get("prompt_ms", 0) / 1000,
            "prompt_tokens": timings.get("prompt_n", 0),
            "generated_tokens": timings.get("predicted_n", 0),
        }


def shown(path: Path) -> Path:
    """Repository-relative when possible: the UI must not leak the home directory."""
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


class Studio:
    def __init__(self, models: dict[str, tuple[Path, str]], records: list[Path]):
        """models: label -> (bundle directory, llama-server URL serving its GGUF)."""
        if not models:
            raise ValueError("Register at least one model bundle")
        self.models = {
            label: {
                "bundle": bundle,
                "url": url,
                "runtime": LlamaFELN(bundle, url),
            }
            for label, (bundle, url) in models.items()
        }
        self.gold = {}
        for path in records:
            for record in json.loads(path.read_text()):
                self.gold[key(record["text"])] = record["meta"]
        self.lock = Lock()

    def config(self) -> dict:
        return {
            "models": [
                {
                    "label": label,
                    "bundle": f"{shown(entry['bundle'])} → {entry['url']}",
                    "prompt_prefix": entry["runtime"].prompt["prompt_prefix"],
                }
                for label, entry in self.models.items()
            ],
            "questions": sorted(self.gold, key=len),
        }

    def generate(self, label: str, query: str, prompt_prefix: str | None) -> dict:
        runtime = self.models[label]["runtime"]
        # ponytail: one request at a time; llama-server runs a single slot.
        with self.lock:
            try:
                result = runtime.generate(query, prompt_prefix)
            except ValueError as exc:
                return {"valid": False, "error": str(exc)}
        gold = self.gold.get(key(query))
        expected = exact = None
        if gold is not None:
            expected = runtime.schema.compile(gold)
            exact = FELN.model_validate(result["feln"]).same(FELN.model_validate(expected))
        return {
            "valid": True,
            "feln": result["feln"],
            "raw": result["raw"],
            "seconds": result["seconds"],
            "first_token_seconds": result["first_token_seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "generated_tokens": result["generated_tokens"],
            "backend": result["backend"],
            "expected": expected,
            "exact": exact,
        }


def handler_for(app: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Questions and model output stay out of access logs.

        def send(self, status: int, content: bytes, mime: str):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def json(self, status: int, payload: dict):
            self.send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

        def local_request(self) -> bool:
            port = cast(ThreadingHTTPServer, self.server).server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin") in {
                None,
                *(f"http://{host}" for host in hosts),
            }

        def do_GET(self):
            if not self.local_request():
                return self.json(403, {"error": "Local requests only"})
            if self.path == "/api/config":
                return self.json(200, app.config())
            files = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if self.path not in files:
                return self.json(404, {"error": "Not found"})
            name, mime = files[self.path]
            self.send(200, (STATIC / name).read_bytes(), mime + "; charset=utf-8")

        def do_POST(self):
            if not self.local_request():
                return self.json(403, {"error": "Local requests only"})
            if self.path != "/api/generate":
                return self.json(404, {"error": "Not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.json(415, {"error": "Expected application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 200_000:
                    raise ValueError("Request must be between 1 and 200,000 bytes")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")  # noqa: TRY004 - client error, not a type bug
                query, label = data.get("query"), data.get("model")
                prefix = data.get("prompt_prefix")
                if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                    raise ValueError("Enter a question of 1–2,000 characters")
                if label not in app.models:
                    raise ValueError("Choose one of the registered models")
                if prefix is not None and (
                    not isinstance(prefix, str) or not 0 < len(prefix) <= 100_000
                ):
                    raise ValueError("Prompt prefix must be 1–100,000 characters")
            except (ValueError, UnicodeDecodeError) as exc:
                message = "Invalid JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
                return self.json(400, {"error": message})
            try:
                result = app.generate(label, query.strip(), prefix)
            except Exception:
                logging.getLogger(__name__).exception("Inference failed")
                return self.json(502, {"error": "Inference failed; inspect the server terminal."})
            self.json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--no-nemotron",
        action="store_true",
        help="Do not register (or start) the default Nemotron GGUF.",
    )
    parser.add_argument(
        "--llama",
        action="append",
        metavar="LABEL=BUNDLE=URL",
        help="GGUF bundle (inference_config.json + Layers.json) served by llama-server at URL.",
    )
    parser.add_argument(
        "--records",
        action="append",
        type=Path,
        help="Question/FELN files used as gold; default tests/challenge.json.",
    )
    args = parser.parse_args()
    models = {}
    if not args.no_nemotron:
        models[NEMOTRON["label"]] = (
            NEMOTRON["bundle"],
            serve_gguf(NEMOTRON["gguf"], NEMOTRON["port"]),
        )
    for item in args.llama or []:
        label, bundle, url = item.split("=", 2)
        models[label] = (Path(bundle), url)
    app = Studio(models, args.records or [ROOT / "tests/challenge.json"])
    # A plain kill must still run atexit, which stops the llama-server we started.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app)) as server:
        print(f"FELN Studio → http://127.0.0.1:{server.server_port}", flush=True)
        for label, entry in app.models.items():
            print(f"  {label}: {entry['bundle']} {entry['url']}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
