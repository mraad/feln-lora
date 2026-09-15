"""GPU delegation wiring without AWS, SSH, or a GPU: fake runner, local fake llama-server."""

import json
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import ClassVar

from src import gpu_server

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


class FakeLlama(BaseHTTPRequestHandler):
    prompts: ClassVar[list] = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlama.prompts.append(payload)
        body = json.dumps(
            {"content": '{"layers": ["wells"], "where": ["content_type = 2"], "relations": []}'}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GpuServerChecks(unittest.TestCase):
    def test_machine_server_and_generate(self):
        calls = []

        def runner(command, timeout=60):
            calls.append(command)
            if command[0] == "aws":
                return json.dumps(
                    {
                        "Reservations": [
                            {
                                "Instances": [
                                    {
                                        "InstanceId": "i-1",
                                        "InstanceType": "g7e.12xlarge",
                                        "State": {"Name": "running"},
                                        "PublicIpAddress": "1.2.3.4",
                                    }
                                ]
                            }
                        ]
                    }
                )
            script = command[-1]
            if script.startswith("tmux has-session"):
                # Reflects the state after a start request: sessions up, healthy, right alias.
                started = any("tmux new-session" in c[-1] for c in calls)
                block = (
                    ("session" if started else "nosession")
                    + "\n"
                    + ('{"status":"ok"}' if started else "nohealth")
                    + "\n"
                    + ('{"data":[{"id":"alias-x"}]}' if started else "nomodels")
                    + "\n"
                )
                return block * 2 + "0, 10 MiB, 0 %\n1, 10 MiB, 0 %\n"
            if script.startswith("sha256sum"):
                return "abc  /remote/model.gguf\n"
            return ""

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            bundle.mkdir()
            (bundle / "Layers.json").write_text(json.dumps(LAYERS))
            (bundle / "inference_config.json").write_text(
                json.dumps(
                    {
                        "prompt_prefix": "P:",
                        "prompt_suffix": ":S",
                        "max_new_tokens": 8,
                        "json_schema": {},
                    }
                )
            )
            with ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama) as llama:
                Thread(target=llama.serve_forever, daemon=True).start()
                config = {
                    "machine": {
                        "profile": "p",
                        "region": "r",
                        "instance_id": "i-1",
                        "name": "box",
                    },
                    "ssh": {"host": "h", "user": "u", "key": "/k"},
                    "server": {
                        "llama_server": "/remote/llama-server",
                        "gguf": "/remote/model.gguf",
                        "gguf_sha256": "abc",
                        "alias": "alias-x",
                        "port": 8090,
                        "tmux": "sess",
                        "gpus": ["0", "1"],
                        "slots": 2,
                        "context": 2048,
                    },
                    "bundle": str(bundle),
                    "local_port": llama.server_port,
                }
                gpu = gpu_server.GpuServer(config, runner)
                gpu.urls = lambda: [f"http://127.0.0.1:{llama.server_port}"] * 2
                self.assertEqual(gpu.machine_status()["state"], "running")
                self.assertEqual(calls[-1][:4], ["aws", "--profile", "p", "--region"])

                status = gpu.server_start()
                self.assertTrue(status["healthy"])
                self.assertEqual([i["port"] for i in status["instances"]], [8090, 8091])
                starts = [c[-1] for c in calls if "tmux new-session" in c[-1]]
                self.assertEqual(len(starts), 1)
                for expected in (
                    "CUDA_VISIBLE_DEVICES=0",
                    "sess-0",
                    "--port 8090",
                    "CUDA_VISIBLE_DEVICES=1",
                    "sess-1",
                    "--alias alias-x",
                    "--port 8091",
                    "-c 4096 -np 2",
                    "-ngl all",
                    "--host 127.0.0.1",
                ):
                    self.assertIn(expected, starts[0])
                self.assertTrue(any(c[-1].startswith("sha256sum") for c in calls))

                # The fake llama endpoint already answers on local_port: no ssh tunnel is spawned.
                result = gpu.generate("Show gas wells")
                self.assertEqual(result["feln"]["layers"], ["Wells"])
                self.assertIn("CAST(2 AS SMALLINT)", result["feln"]["where"][0])
                self.assertEqual(FakeLlama.prompts[-1]["prompt"], "P:Show gas wells:S")
                self.assertIsNone(gpu.tunnel_process)

                gpu.config["server"]["alias"] = "other"
                with self.assertRaises(RuntimeError):
                    gpu.server_start()
                llama.shutdown()


if __name__ == "__main__":
    unittest.main()
