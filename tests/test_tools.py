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
        self.wait = lambda s: None
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
    assert names == {"look", "pick", "place", "move_to", "gripper", "home", "status",
                     "move_ball", "calibrate_look"}
    for s in r[1]["result"]["tools"]:
        assert s["inputSchema"]["type"] == "object"
    assert r[2]["error"]["code"] == -32601


def test_look_returns_image_and_moves_to_region():
    sk = FakeSkills()
    t = ArmTools(skills=sk, snapshot=lambda: b"\xff\xd8fakejpeg")
    content = t.call("look", {"where": "right"})
    cx, cy = REGIONS["right"]["center"]
    name, (x, y, z) = sk.calls[0]
    assert name == "move_to" and abs(x - 0.7 * cx) < 1e-9 and abs(y - 0.7 * cy) < 1e-9 and z == 80.0
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


def test_sim_move_ball_one_call():
    pytest.importorskip("mujoco")
    t = ArmTools.sim(n=1, seed=3, obj_h=20)  # 15 mm sim cubes slip in the sim fingers
    out = t.call("move_ball", {"from": "left", "to": "right"})
    log = out[0]["text"]
    assert "DONE" in log, log
    x, y, _ = t.truth()[0]["xyz"]
    rx, ry = REGIONS["right"]["center"]
    assert abs(x - rx) < 15 and abs(y - ry) < 15, (x, y, log)


def test_look_calib_fit_roundtrip():
    pytest.importorskip("cv2")
    from armpi.ball import LookCalib
    lc = LookCalib()
    px = [(100, 100), (500, 110), (480, 400), (120, 390), (300, 250)]
    tb = [(110, -50), (110, -80), (80, -80), (80, -50), (95, -65)]
    err = lc.fit("right", px, tb)
    assert err < 3
    x, y = lc.to_table("right", 300, 250)
    assert abs(x - 95) < 3 and abs(y + 65) < 3
