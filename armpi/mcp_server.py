"""MCP server for the arm (stdio, stdlib only), so an LLM can drive it with tools.

    python -m armpi.mcp_server --url http://10.0.0.133:8000     # real arm via the Pi server
    python -m armpi.mcp_server --sim                            # MuJoCo arm + a cube

Claude Code picks it up from .mcp.json at the repo root (start `claude` there), or add it:
    claude mcp add armpi -- .venv/bin/python -m armpi.mcp_server --url http://10.0.0.133:8000

Tools: look, pick, place, move_to, gripper, home, status (see armpi/tools.py).
Logs go to stderr; stdout carries only JSON-RPC.
"""
import argparse
import json
import sys
import traceback

from .tools import ArmTools

SERVER = {"name": "armpi", "version": "0.1.0"}
DEFAULT_PROTOCOL = "2025-06-18"


def handle(tools, msg):
    """One JSON-RPC message -> response dict, or None for notifications."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return None  # notification (e.g. notifications/initialized)

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        return ok({"protocolVersion": params.get("protocolVersion", DEFAULT_PROTOCOL),
                   "capabilities": {"tools": {}}, "serverInfo": SERVER,
                   "instructions": "Robot arm on a table. Coordinates are mm in the arm's base "
                                   "frame: +x forward, +y to the arm's left. Look before and "
                                   "after picking; the arm can't feel what it holds."})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": tools.specs()})
    if method == "tools/call":
        try:
            return ok({"content": tools.call(params["name"], params.get("arguments")),
                       "isError": False})
        except Exception as e:  # tool errors go back to the model, not the protocol
            print(traceback.format_exc(), file=sys.stderr)
            return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}


def serve(tools, inp=sys.stdin, out=sys.stdout):
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(tools, msg)
        if resp is not None:
            out.write(json.dumps(resp) + "\n")
            out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="Pi server, e.g. http://10.0.0.133:8000")
    g.add_argument("--sim", action="store_true", help="MuJoCo arm with one cube in the left square")
    ap.add_argument("--obj-h", type=float, default=15.0, help="object height, mm")
    a = ap.parse_args(argv)
    tools = ArmTools.sim(obj_h=a.obj_h) if a.sim else ArmTools(a.url, obj_h=a.obj_h)
    print(f"armpi mcp server ready ({'sim' if a.sim else a.url})", file=sys.stderr)
    serve(tools)


if __name__ == "__main__":
    main()
