"""System identification: recover known servo params from synthetic 'filmed' step moves."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("mujoco")

from armpi import sysid
from armpi.calibration import Calibration


def synthetic_log(joint, program=sysid.PROGRAM, fps=30.0):
    calib = Calibration.load()
    start = dict(sysid.START)
    t, moves, pulses = 1.0, [], {j: calib.to_pulse(j, a) for j, a in start.items()}
    for off, ms in program:
        tgt = start[joint] + off
        us = calib.to_pulse(joint, tgt)
        ms = int(max(ms, abs(us - pulses[joint]) * calib.min_ms_per_us))  # the speed guard
        pulses[joint] = us
        moves.append({"t": t, "angles": {joint: tgt}, "duration_ms": ms})
        t += ms / 1000 + 0.8
    frames = [{"t": float(x)} for x in np.arange(0.2, t, 1 / fps)]
    return {"joint": joint, "start": start, "moves": moves, "frames": frames}


def test_commanded_ramp():
    log = {"joint": "base", "start": {"base": 0.0},
           "moves": [{"t": 1.0, "angles": {"base": 10.0}, "duration_ms": 1000},
                     {"t": 3.0, "angles": {"base": 0.0}, "duration_ms": 0}]}
    got = sysid.commanded_at(log, [0.5, 1.5, 2.5, 3.0, 4.0])
    assert np.allclose(got, [0, 5, 10, 0, 0])


def test_rotation_angles_recovers_signed_angle():
    from scipy.spatial.transform import Rotation
    axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)
    R0 = Rotation.from_rotvec([0.4, 0.1, -0.2]).as_matrix()
    want = np.array([0, 2, 5, 9, 4, -3, -8, 0.0])
    Rs = [Rotation.from_rotvec(axis * math.radians(a)).as_matrix() @ R0 for a in want]
    got = sysid.rotation_angles(Rs, sign_ref=want)
    assert np.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("joint", ["base", "shoulder"])
def test_fit_recovers_params(joint):
    true = {joint: {"kp": 1.2, "damping": 0.06, "torque": 0.4}}
    delay = 0.06
    prog = sysid.PROGRAM[:4]
    log = synthetic_log(joint, prog)
    ts = np.array([f["t"] for f in log["frames"]])
    meas_true = sysid.simulate(log, ts, true, delay)
    rng = np.random.default_rng(0)
    meas = {"t": ts.tolist(), "angle": (meas_true + rng.normal(0, 0.15, len(ts))).tolist()}
    servo, d, rep = sysid.fit([(log, meas)], verbose=False)
    assert rep["rms_after_deg"] < 0.25 < rep["rms_before_deg"], rep
    assert abs(d - delay) < 0.015, d
    # params can trade off (kp vs torque under gravity); what must match is the behavior
    fitted = sysid.simulate(log, ts, servo, d)
    assert np.abs(fitted - meas_true).max() < 0.5


def test_save_params_feeds_the_model(tmp_path):
    path = tmp_path / "p.yaml"
    sysid.save_params({"base": {"kp": 7.5, "damping": 0.03, "torque": 0.3}}, 0.05,
                      {"rms_before_deg": 2.0, "rms_after_deg": 0.3}, path)
    from armpi.mujoco_model import load_params
    p = load_params(path)
    assert p["servo"]["base"]["kp"] == 7.5 and p["servo"]["shoulder"]["kp"] == 3.0


def test_mjpeg_recorder_reads_frame_times(tmp_path):
    import time
    sys.path.insert(0, os.path.dirname(__file__))
    from test_server import start_server

    class FakeCamera:
        error = None

        def __init__(self):
            self.n, self.stamp = 0, None

        def frame(self, timeout=0, after=None):
            time.sleep(0.02)
            self.n += 1
            self.stamp = time.monotonic()
            return b"\xff\xd8 fake jpeg %d \xff\xd9" % self.n

    _, app, url, _ = start_server()
    app.camera = FakeCamera()
    rec = sysid.MjpegRecorder(url, tmp_path)
    rec.start()
    time.sleep(0.4)
    rec.stop = True
    rec.join(2)
    assert rec.error is None and len(rec.frames) >= 5, rec.error
    ts = [f["t"] for f in rec.frames]
    assert ts == sorted(ts) and abs(ts[-1] - app.now()) < 1.0
    assert (tmp_path / rec.frames[0]["file"]).read_bytes().startswith(b"\xff\xd8")
