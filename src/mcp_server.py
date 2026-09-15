"""FELN MCP server: local tools, remote GPU inference.

Register with an MCP client over stdio, e.g. for Claude Code:
  claude mcp add feln-gpu -- $PWD/.venv/bin/python -m src.mcp_server
`feln` starts the SSH tunnel (and the remote llama-server if the machine is up) on demand;
the EC2 machine itself is only started or stopped by the explicit tools.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .feln_data import Schema
from .gpu_server import ROOT, GpuServer
from .spatial_query import DATABASE as DEFAULT_DATABASE
from .spatial_query import SpatialQuery

gpu = GpuServer.load()
atexit.register(gpu.stop_tunnel)
database = SpatialQuery(
    Path(os.environ.get("FELN_DATABASE", DEFAULT_DATABASE)),
    Schema(ROOT / gpu.config["bundle"] / "Layers.json"),
)
mcp = MCPServer(
    "feln-gpu",
    instructions=(
        "Translate North Sea oil-and-gas questions (Wells, Pipelines, Discoveries) into "
        "FELN with `feln`, then run the result with `execute_feln` to get actual rows. "
        "Use `status` before starting or stopping the GPU machine."
    ),
)


@mcp.tool()
def feln(text: str) -> dict:
    """Translate a natural-language geospatial question into FELN JSON on the remote GPU."""
    # Steady state is local HTTP through the tunnel; ssh only when that is down.
    if not gpu.tunnel_ok():
        status = gpu.server_status()
        if status.get("reachable") and not status.get("healthy"):
            gpu.server_start()
    return gpu.generate(text)


@mcp.tool()
def execute_feln(feln: dict, limit: int = 50) -> dict:
    """Run a FELN query read-only against the local North Sea DuckDB; returns rows without geometry."""
    result = database.execute(feln, limit)
    return {
        "layer": result["layer"]["name"],
        "count": result["count"],
        "truncated": result["truncated"],
        "rows": [f["properties"] for f in result["geojson"]["features"]],
        "warnings": result["warnings"],
        "query_seconds": result["query_seconds"],
    }


@mcp.tool()
def status() -> dict:
    """EC2 machine state, remote llama-server health, and local tunnel state."""
    machine = gpu.machine_status()
    server = gpu.server_status() if machine["state"] == "running" else {"reachable": False}
    return {"machine": machine, "server": server, "tunnel": gpu.tunnel_ok()}


@mcp.tool()
def machine_start() -> dict:
    """Start the GPU EC2 instance (billable) and wait until SSH answers."""
    return gpu.machine_start()


@mcp.tool()
def machine_stop() -> dict:
    """Stop the GPU EC2 instance; kills the remote server and any tmux jobs on it."""
    return gpu.machine_stop()


@mcp.tool()
def server_stop() -> dict:
    """Stop the remote llama-server and the local tunnel; leaves the machine running."""
    return gpu.server_stop()


if __name__ == "__main__":
    mcp.run()
