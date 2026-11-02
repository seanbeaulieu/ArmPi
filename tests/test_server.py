"""End-to-end: armpi.server on the sim backend, driven over HTTP by RemoteBackend.
Run: python tests/test_server.py   (no hardware, no camera needed)"""
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armpi import Arm, Calibration
from armpi.backends import SimBackend
from armpi.backends.remote import RemoteBackend
from armpi.calibration import DEFAULT_PATH
from armpi.server import App, make_handler


class NoCamera:
    error = "no camera in tests"

    def frame(self, timeout=0, after=None):
        return None


def start_server():
    tmp = tempfile.mkdtemp()
    cal_path = os.path.join(tmp, "calibration.yaml")
    shutil.copy(DEFAULT_PATH, cal_path)
    server_arm = Arm(SimBackend(), Calibration.load(cal_path))
    app = App(server_arm, NoCamera(), "sim")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, app, f"http://127.0.0.1:{srv.server_address[1]}", cal_path


SRV, APP, URL, CAL_PATH = start_server()


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(path):
    with urllib.request.urlopen(URL + path) as r:
        return r.read()


def test_remote_arm_moves_server_sim():
    pc_arm = Arm(RemoteBackend(URL), Calibration.load(CAL_PATH))
    pc_arm.move({"base": 30, "gripper": 1}, 500)
    sent = APP.arm.backend.history[-1][1]
    assert sent == {6: pc_arm.calib.to_pulse("base", 30), 1: pc_arm.calib.to_pulse("gripper", 1)}
    assert abs(APP.arm.angles["base"] - 30) < 0.1
    assert pc_arm.battery_mv() == 7400


def test_state_has_fk_once_arm_joints_known():
    APP.arm.rest()
    s = json.loads(get("/api/state"))
    assert s["backend"] == "sim" and s["fk"] is not None and len(s["fk"]["fingertip"]) == 3


def test_move_to_and_errors():
    code, r = post("/api/move_to", {"xyz": [100, 0, 30], "pitch": -90})
    assert code == 200 and abs(r["pitch"] + 90) < 1e-6
    code, r = post("/api/move_to", {"xyz": [900, 0, 0]})
    assert code == 400 and "unreachable" in r["error"]
    code, r = post("/api/move", {"angles": {"nope": 1}})
    assert code == 400
    code, r = post("/api/nothing", {})
    assert code == 404


def test_pulse_out_of_range_is_rejected_by_board_check():
    code, r = post("/api/pulse", {"joint": "base", "us": 2600})
    assert code == 400


def test_calibrate_marks_save_to_file():
    post("/api/pulse", {"joint": "shoulder", "us": 1470})
    code, j = post("/api/calibrate", {"joint": "shoulder", "mark": "zero", "angle": 90})
    assert code == 200 and j["zero_pulse"] == 1470 - j["trim"]
    post("/api/pulse", {"joint": "shoulder", "us": 1470 + 555})  # ~50 deg down from vertical
    post("/api/calibrate", {"joint": "shoulder", "mark": "min"})
    saved = Calibration.load(CAL_PATH)
    assert abs(saved.limits("shoulder")[0] - 40) < 0.2
    assert open(CAL_PATH).read().startswith("# ArmPi Mini servo calibration")  # header kept
    assert DEFAULT_PATH.read_text() != open(CAL_PATH).read()  # repo file untouched


def test_web_panel_served():
    assert b"<title>ArmPi Control</title>" in get("/")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
