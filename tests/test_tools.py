"""LLM tool registry + the stdio MCP server, driven against the MuJoCo arm."""
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from armpi import mcp_server
from armpi.tools import REGIONS, ArmTools


def _rpc(tools, *msgs):
    inp = io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n")
    out = io.StringIO()
    mcp_server.serve(tools, inp, out)
    return [json.loads(l) for l in out.getvalue().splitlines()]


class FakeSkills:
    """Records calls; enough of Skills + Arm for ArmTools."""
    def __init__(self):
        self.calls = []
        outer = self

        class A:
            angles = {"base": 0, "shoulder": 90, "elbow": -90, "wrist": -45}
            def fk(self):
                return {"fingertip": (100.0, 0.0, 50.0)}
            def battery_mv(self):
                return 4400
        self.arm = A()
        for name in ("pick", "place", "move_to", "gripper", "home"):
            setattr(self, name, lambda *a, _n=name: outer.calls.append((_n, a)) or -80.0)


def test_protocol_and_specs():
    t = ArmTools(skills=FakeSkills(), snapshot=lambda: b"\xff\xd8fakejpeg")
    r = _rpc(t, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18"}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
             {"jsonrpc": "2.0", "id": 3, "method": "nope"})
    assert [m["id"] for m in r] == [1, 2, 3]  # notification got no reply
    assert r[0]["result"]["protocolVersion"] == "2025-06-18"
    names = {s["name"] for s in r[1]["result"]["tools"]}
    assert names == {"look", "pick", "place", "move_to", "gripper", "home", "status"}
    for s in r[1]["result"]["tools"]:
        assert s["inputSchema"]["type"] == "object"
    assert r[2]["error"]["code"] == -32601


def test_look_returns_image_and_moves_to_region():
    sk = FakeSkills()
    t = ArmTools(skills=sk, snapshot=lambda: b"\xff\xd8fakejpeg")
    content = t.call("look", {"where": "right"})
    assert sk.calls[0] == ("move_to", (*REGIONS["right"]["center"], 90.0))
    img = [c for c in content if c["type"] == "image"][0]
    assert base64.b64decode(img["data"]) == b"\xff\xd8fakejpeg" and img["mimeType"] == "image/jpeg"


def test_tool_errors_go_back_as_content():
    sk = FakeSkills()
    def boom(*a):
        raise ValueError("(400, 0, 6) unreachable within joint limits")
    sk.pick = boom
    t = ArmTools(skills=sk, snapshot=None)
    r = _rpc(t, {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                 "params": {"name": "pick", "arguments": {"x": 400, "y": 0}}})
    assert r[0]["result"]["isError"] is True
    assert "unreachable" in r[0]["result"]["content"][0]["text"]


def test_status_warns_on_low_battery():
    t = ArmTools(skills=FakeSkills(), snapshot=None)
    assert "LOW" in t.call("status")[0]["text"]


def test_sim_llm_style_episode():
    """What an LLM would do: look, pick where the object is, place in the other square."""
    pytest.importorskip("mujoco")
    t = ArmTools.sim(n=1, seed=2)
    obj = t.truth()[0]["xyz"]
    t.call("look", {"where": "left"})
    t.call("pick", {"x": obj[0], "y": obj[1]})
    rx, ry = REGIONS["right"]["center"]
    t.call("place", {"x": rx, "y": ry})
    t.call("home")
    x, y, _ = t.truth()[0]["xyz"]
    assert abs(x - rx) < 15 and abs(y - ry) < 15, (x, y)
