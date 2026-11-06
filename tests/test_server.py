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
import fixed_calibration  # noqa: E402,F401  (tests use a frozen calibration)

from armpi import Arm, Calibration
from armpi.backends import SimBackend
from armpi.backends.remote import RemoteBackend
import armpi.calibration
from armpi.server import App, make_handler


class NoCamera:
    error = "no camera in tests"

    def frame(self, timeout=0, after=None):
        return None


def no_board(name):
    if name == "i2c":
        raise FileNotFoundError("/dev/i2c-1")
    return SimBackend()


def start_server(make_backend=no_board):
    tmp = tempfile.mkdtemp()
    cal_path = os.path.join(tmp, "calibration.yaml")
    shutil.copy(armpi.calibration.DEFAULT_PATH, cal_path)
    app = App(Calibration.load(cal_path), NoCamera(), "sim", make_backend)
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
    assert armpi.calibration.DEFAULT_PATH.read_text() != open(CAL_PATH).read()  # source untouched
    assert "\n    t:" not in open(CAL_PATH).read()  # response fields don't leak into the file


def test_live_angles_follow_sim_motion():
    post("/api/rest", {"duration_ms": 0})
    post("/api/move", {"angles": {"base": 40}, "duration_ms": 3000})
    s = json.loads(get("/api/state"))
    assert abs(s["angles"]["base"] - 40) < 0.1 and abs(s["live"]["base"] - 40) > 5  # still on its way
    assert s["target_fk"]["fingertip"] != s["fk"]["fingertip"]


def test_switch_backend_without_board_stays_on_sim():
    code, r = post("/api/backend", {"name": "i2c"})
    assert code == 502 and "/dev/i2c-1" in r["error"]
    code, r = post("/api/backend", {"name": "bogus"})
    assert code == 400
    s = json.loads(get("/api/state"))
    assert s["backend"] == "sim" and s["battery_source"] == "sim" and s["board_error"]


def test_switch_backend_keeps_each_arms_pose():
    board = SimBackend(battery_mv=6900)  # stands in for the I2C board
    srv, app, url, _ = start_server(lambda name: board if name == "i2c" else SimBackend())
    try:
        assert app.state()["battery_mv"] == 6900  # board battery shown even in sim mode
        app.post("/api/move", {"angles": {"base": 10}})
        app.post("/api/backend", {"name": "i2c"})
        assert app.state()["backend"] == "i2c" and app.arm.angles == {}  # real pose unknown
        app.post("/api/move", {"angles": {"base": -20}})
        assert board.history[-1][1] == {6: app.calib.to_pulse("base", -20)}
        app.post("/api/backend", {"name": "sim"})
        assert abs(app.arm.angles["base"] - 10) < 0.1
    finally:
        srv.shutdown()


def test_web_panel_served():
    assert b"<title>ArmPi Control</title>" in get("/")



def test_studio_served():
    assert b"<title>Arm Studio</title>" in get("/studio")


def test_history_and_log():
    APP.sample()  # the test server runs without the sampler thread
    post("/api/move", {"angles": {"base": 12}, "duration_ms": 300})
    APP.sample()
    h = json.loads(get("/api/history?since=-1"))
    assert h["items"] and abs(h["items"][-1]["cmd"]["base"] - 12) < 0.1
    newest = h["items"][-1]["t"]
    assert json.loads(get(f"/api/history?since={newest}"))["items"] == []
    log = json.loads(get("/api/log?since=-1"))["items"]
    assert any(e["kind"] == "move" and "base=12" in e["text"] for e in log)
    post("/api/move", {"angles": {"base": 500}})  # clamped -> logged as a warning
    assert json.loads(get("/api/log?since=-1"))["items"][-1]["warn"]
    post("/api/move_to", {"xyz": [900, 0, 0]})   # unreachable -> logged as an error
    assert json.loads(get("/api/log?since=-1"))["items"][-1]["kind"] == "error"



def test_real_url_drives_remote_servers_real_arm():
    # "Pi": a server whose real arm is a sim we can inspect
    pi_board = SimBackend()
    pi_srv, pi_app, pi_url, _ = start_server(lambda n: pi_board if n == "i2c" else SimBackend())
    pc = App(Calibration.load(CAL_PATH), NoCamera(), "sim", real_url=pi_url)
    assert pi_app.backend_name == "sim"          # nothing touched until selected
    pc.post("/api/backend", {"name": "i2c"})
    assert pi_app.backend_name == "i2c" and "via" in pc.label()
    pc.post("/api/move", {"angles": {"base": 15}, "duration_ms": 400})
    assert pi_board.history and pi_board.history[-1][1][6] == pc.calib.to_pulse("base", 15)
    pi_srv.shutdown()



def test_sim_backend_shares_calibration():
    """A sim that converts pulses itself (mujoco) must see new calibration marks live."""
    class CalibSim(SimBackend):
        calib = Calibration.load(CAL_PATH)  # its own stale copy, like MujocoBackend's

    made = []
    def factory(name):
        if name == "i2c":
            raise FileNotFoundError("/dev/i2c-1")
        made.append(CalibSim())
        return made[-1]
    app = App(Calibration.load(CAL_PATH), NoCamera(), "sim", factory)
    assert made and made[0].calib is app.calib


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
