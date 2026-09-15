"""Remote GPU inference: EC2 machine control, llama-server lifecycle over SSH, local tunnel.

Configuration: gpu_server.json at the repository root (no secrets; the SSH key is a path).
  python -m src.gpu_server machine status|start|stop
  python -m src.gpu_server server status|start|stop
  python -m src.gpu_server feln 'Show gas wells in Norway'
The tunnel forwards 127.0.0.1:<local_port> to the remote server's localhost port; nothing
listens publicly on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from .edge_client import healthy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "gpu_server.json"


def run(command: list[str], timeout: int = 60) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} failed ({result.returncode}): {result.stderr.strip()[-600:]}"
        )
    return result.stdout


class GpuServer:
    def __init__(self, config: dict, runner=run):
        self.config = config
        self.run = runner
        self.tunnel_process = None
        self._next = -1

    @classmethod
    def load(cls, path: Path = CONFIG) -> GpuServer:
        return cls(json.loads(path.read_text()))

    # --- EC2 machine -----------------------------------------------------------------

    def aws(self, *args: str) -> dict:
        machine = self.config["machine"]
        command = [
            "aws",
            "--profile",
            machine["profile"],
            "--region",
            machine["region"],
            "--output",
            "json",
            "ec2",
            *args,
        ]
        return json.loads(self.run(command) or "{}")

    def machine_status(self) -> dict:
        instance = self.aws(
            "describe-instances",
            "--instance-ids",
            self.config["machine"]["instance_id"],
        )["Reservations"][0]["Instances"][0]
        return {
            "instance_id": instance["InstanceId"],
            "type": instance["InstanceType"],
            "state": instance["State"]["Name"],
            "public_ip": instance.get("PublicIpAddress"),
            "name": self.config["machine"]["name"],
        }

    def machine_start(self, wait_seconds: int = 300) -> dict:
        self.aws("start-instances", "--instance-ids", self.config["machine"]["instance_id"])
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            status = self.machine_status()
            if status["state"] == "running" and self.ssh_ok():
                return {**status, "ssh": True}
            time.sleep(10)
        raise TimeoutError("Machine did not become reachable over SSH in time")

    def machine_stop(self) -> dict:
        # Stopping loses tmux sessions on the host; the EBS volume and the elastic IP survive.
        self.stop_tunnel()
        self.aws("stop-instances", "--instance-ids", self.config["machine"]["instance_id"])
        return self.machine_status()

    # --- SSH and the remote llama-server ----------------------------------------------

    def ssh_command(self, *extra: str, multiplex: bool = True) -> list[str]:
        ssh = self.config["ssh"]
        # Commands share one authenticated connection for a minute; the tunnel must
        # not, or the persisted master would inherit its port forwards.
        control = (
            [
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPath=/tmp/feln-gpu-%C",
                "-o",
                "ControlPersist=60",
            ]
            if multiplex
            else ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
        )
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            *control,
            "-i",
            ssh["key"],
            *extra,
            f"{ssh['user']}@{ssh['host']}",
        ]

    def ssh(self, script: str, timeout: int = 60) -> str:
        return self.run([*self.ssh_command("-T"), script], timeout=timeout)

    def ssh_ok(self) -> bool:
        try:
            return self.ssh("echo ok").strip() == "ok"
        except (RuntimeError, subprocess.TimeoutExpired):
            return False

    # One llama-server per GPU: a 4B model does not saturate one Blackwell GPU, and
    # splitting a single instance across GPUs was measured slower; independent
    # instances double throughput instead. Instance i: GPU gpus[i], port base+i.
    def instances(self) -> list[dict]:
        server = self.config["server"]
        return [
            {
                "gpu": gpu,
                "port": server["port"] + i,
                "local_port": self.config["local_port"] + i,
                "tmux": f"{server['tmux']}-{gpu}",
            }
            for i, gpu in enumerate(server["gpus"])
        ]

    def server_status(self) -> dict:
        server = self.config["server"]
        script = ""
        for inst in self.instances():
            script += (
                f"tmux has-session -t {shlex.quote(inst['tmux'])} 2>/dev/null && echo session || echo nosession; "
                # printf, not echo: a failed curl must still yield exactly one line.
                f"curl -s -m 3 http://127.0.0.1:{inst['port']}/health || printf nohealth; echo; "
                f"curl -s -m 3 http://127.0.0.1:{inst['port']}/v1/models || printf nomodels; echo; "
            )
        script += "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true"
        try:
            lines = self.ssh(script).splitlines()
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            return {"reachable": False, "error": str(exc)[-300:]}
        instances = []
        for i, inst in enumerate(self.instances()):
            session, health, models = lines[3 * i : 3 * i + 3]
            aliases = (
                [m["id"] for m in json.loads(models).get("data", [])]
                if models.startswith("{")
                else []
            )
            instances.append(
                {
                    **inst,
                    "tmux_session": session == "session",
                    "healthy": '"ok"' in health,
                    "models": aliases,
                    "serving_expected_model": server["alias"] in aliases,
                }
            )
        return {
            "reachable": True,
            "healthy": all(i["healthy"] for i in instances),
            "instances": instances,
            "gpus": [line for line in lines[3 * len(instances) :] if line],
        }

    def server_start(self, wait_seconds: int = 180) -> dict:
        server = self.config["server"]
        status = self.server_status()
        if not status.get("reachable"):
            raise RuntimeError("GPU host is not reachable over SSH; start the machine first")
        for inst in status["instances"]:
            if inst["healthy"] and not inst["serving_expected_model"]:
                raise RuntimeError(
                    f"Port {inst['port']} serves {inst['models']}, not {server['alias']}; refusing to replace it"
                )
        if status["healthy"]:
            return status
        if server.get("gguf_sha256"):
            digest = self.ssh(f"sha256sum {shlex.quote(server['gguf'])}", timeout=300).split()[0]
            if digest != server["gguf_sha256"]:
                raise RuntimeError("Remote GGUF checksum mismatch; not starting the server")
        log = shlex.quote(str(Path(server["gguf"]).parent / "server.log"))
        starts = []
        for inst in status["instances"]:
            if inst["healthy"] or inst["tmux_session"]:
                continue
            command = shlex.join(
                [
                    server["llama_server"],
                    "-m",
                    server["gguf"],
                    "--alias",
                    server["alias"],
                    "-c",
                    str(server["context"] * server["slots"]),
                    "-np",
                    str(server["slots"]),
                    "-ngl",
                    "all",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(inst["port"]),
                ]
            )
            starts.append(
                f"tmux new-session -d -s {shlex.quote(inst['tmux'])} "
                + shlex.quote(
                    f"CUDA_VISIBLE_DEVICES={inst['gpu']} {command} >> {log} 2>&1; echo EXIT=$? >> {log}"
                )
            )
        self.ssh("; ".join(starts))
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(5)
            status = self.server_status()
            if status.get("healthy"):
                return status
            if any(not i["tmux_session"] and not i["healthy"] for i in status["instances"]):
                raise RuntimeError(f"a llama-server exited; inspect {log} on the GPU host")
        raise TimeoutError("llama-server did not become healthy in time")

    def server_stop(self) -> dict:
        self.stop_tunnel()
        self.ssh(
            "; ".join(
                f"tmux kill-session -t {shlex.quote(i['tmux'])} 2>/dev/null || true"
                for i in self.instances()
            )
        )
        return self.server_status()

    # --- Tunnel and inference ------------------------------------------------------------

    def urls(self) -> list[str]:
        return [f"http://127.0.0.1:{i['local_port']}" for i in self.instances()]

    def url(self) -> str:
        # Round-robin across the per-GPU instances.
        self._next += 1
        return self.urls()[self._next % len(self.urls())]

    def tunnel_ok(self) -> bool:
        return all(healthy(url) for url in self.urls())

    def tunnel(self, wait_seconds: int = 20) -> str:
        if self.tunnel_ok():
            return self.url()
        self.stop_tunnel()
        forwards = []
        for inst in self.instances():
            forwards += ["-L", f"{inst['local_port']}:127.0.0.1:{inst['port']}"]
        # Keepalives make a tunnel whose peer died exit instead of squatting the ports.
        self.tunnel_process = subprocess.Popen(
            self.ssh_command(
                "-N",
                *forwards,
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=2",
                multiplex=False,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(1)
            if self.tunnel_ok():
                return self.url()
            if self.tunnel_process.poll() is not None:
                _, stderr = self.tunnel_process.communicate()
                error = (stderr or b"").decode()[-300:]
                raise RuntimeError(f"SSH tunnel exited: {error}")
        raise TimeoutError(
            "Tunnel is up but a remote instance is not healthy; start the server first"
        )

    def stop_tunnel(self):
        if self.tunnel_process is not None and self.tunnel_process.poll() is None:
            self.tunnel_process.terminate()
        self.tunnel_process = None

    def generate(self, text: str) -> dict:
        from .edge_client import generate

        bundle = ROOT / self.config["bundle"]  # absolute paths pass through unchanged
        started = time.monotonic()
        feln = generate(text, bundle, self.tunnel())
        return {
            "feln": feln,
            "backend": "remote-gpu-llama-server",
            "seconds": time.monotonic() - started,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("machine", "server", "feln"))
    parser.add_argument("action", help="status|start|stop, or the question for feln")
    args = parser.parse_args()
    gpu = GpuServer.load()
    if args.target == "feln":
        result = gpu.generate(args.action)
        gpu.stop_tunnel()
    elif args.action not in {"status", "start", "stop"}:
        parser.error("action must be status, start, or stop")
    else:
        result = getattr(gpu, f"{args.target}_{args.action}")()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
