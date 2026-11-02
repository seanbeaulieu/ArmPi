"""Fit the MuJoCo servo model to the real arm, from step moves filmed by the wrist camera.

No position feedback, so the camera is the sensor: a ChArUco board lies on the table, and
each step moves ONE joint. Every joint downstream of it (camera included) turns about that
joint's axis by exactly the joint's motion, so the camera's rotation relative to its first
frame IS the joint's trajectory. That needs no hand-eye calibration and no joint
calibration, only the intrinsics (and is robust even to rough ones).

    # 1. on the Pi (same clock as the server, no wifi jitter), board on the table under the cam:
    python -m armpi.sysid capture --joint base --out runs/base
    # 2. on the PC:
    python -m armpi.sysid measure runs/base            # board pose per frame -> measured.json
    python -m armpi.sysid fit runs/base runs/shoulder runs/elbow runs/wrist   # -> sim_params.yaml
    python -m armpi.sysid plot runs/base               # measured vs sim before/after (png)

Fitted per joint: kp, damping, torque limit. Shared: camera delay (exposure + buffering).
The fit compares RELATIVE angles (each move vs. the settled start), so the unknown absolute
calibration and steady-state sag at the start pose drop out.
"""
import argparse
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from .calibration import JOINTS, Calibration

FIT_KEYS = ("kp", "damping", "torque")
BOUNDS = {"kp": (0.2, 50.0), "damping": (1e-4, 1.0), "torque": (0.02, 2.0)}
DELAY_BOUNDS = (0.0, 0.3)

# Step program per joint: (offset from start in deg, duration_ms). duration 0 = as fast as
# the speed guard allows (1 ms/us -> ~11 ms/deg). Small, so the board stays in view.
PROGRAM = [(10, 0), (0, 0), (-10, 0), (0, 0), (15, 600), (0, 600), (-15, 300), (0, 300)]
# Start poses: wrist cam looking down-forward at a board ~130mm in front of the base.
START = {"base": 0.0, "shoulder": 60.0, "elbow": -60.0, "wrist": -80.0, "gripper": 0.0}


# --- capture (stdlib only: runs on the Pi as-is) ---------------------------------------------
def _post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


class MjpegRecorder(threading.Thread):
    """Saves every frame of /stream.mjpg with the server's X-T grab time."""

    def __init__(self, url, out_dir):
        super().__init__(daemon=True)
        self.url, self.dir = url, Path(out_dir)
        self.frames, self.stop, self.error = [], False, None

    def run(self):
        try:
            r = urllib.request.urlopen(self.url + "/stream.mjpg", timeout=10)
            while not self.stop:
                hdr = {}
                line = r.readline()
                while line and not line.startswith(b"--frame"):
                    line = r.readline()
                while True:
                    line = r.readline().strip()
                    if not line:
                        break
                    k, _, v = line.decode().partition(":")
                    hdr[k.strip().lower()] = v.strip()
                jpg = r.read(int(hdr["content-length"]))
                if "x-t" not in hdr:
                    raise RuntimeError("server sends no X-T frame times: deploy the new code to the Pi")
                name = f"{len(self.frames):05d}.jpg"
                (self.dir / name).write_bytes(jpg)
                self.frames.append({"t": float(hdr["x-t"]), "file": name})
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"


def capture(url, joint, out_dir, start=None, program=PROGRAM, settle_s=1.2):
    url = url.rstrip("/")
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    start = dict(START, **(start or {}))
    _post(url, "/api/backend", {"name": "i2c"})
    r = _post(url, "/api/move", {"angles": start, "duration_ms": 2000})
    time.sleep(r["duration_ms"] / 1000 + 2.0)
    rec = MjpegRecorder(url, out / "frames")
    rec.start()
    time.sleep(1.0)
    moves = []
    for off, ms in program:
        r = _post(url, "/api/move", {"angles": {joint: start[joint] + off}, "duration_ms": ms})
        moves.append({"t": r["t"], "angles": r["angles"], "duration_ms": r["duration_ms"]})
        print(f"  {joint} {start[joint] + off:+.1f} in {r['duration_ms']}ms", flush=True)
        time.sleep(r["duration_ms"] / 1000 + settle_s)
    rec.stop = True
    rec.join(3)
    if rec.error and not rec.frames:
        raise SystemExit(f"camera stream failed: {rec.error}")
    log = {"joint": joint, "start": start, "moves": moves,
           "frames": [dict(f, file="frames/" + f["file"]) for f in rec.frames],
           "date": time.strftime("%Y-%m-%d %H:%M")}
    (out / "log.json").write_text(json.dumps(log, indent=1))
    fps = len(rec.frames) / max(1e-6, rec.frames[-1]["t"] - rec.frames[0]["t"]) if rec.frames else 0
    print(f"{len(rec.frames)} frames ({fps:.0f} fps), {len(moves)} moves -> {out}/log.json")
    return log


# --- measure: frames -> joint angle --------------------------------------------------------
def rotation_angles(Rs, sign_ref=None):
    """Rotations R_i (camera orientation, any fixed world frame) -> signed angle (deg) of
    each relative to R_0 about their common axis (found by SVD). sign_ref: commanded
    trajectory used to pick the sign (axis direction is otherwise ambiguous)."""
    from scipy.spatial.transform import Rotation
    R0 = Rs[0]
    rv = np.array([Rotation.from_matrix(R @ R0.T).as_rotvec() for R in Rs])
    _, _, vt = np.linalg.svd(rv, full_matrices=False)
    ang = np.degrees(rv @ vt[0])
    if sign_ref is not None and np.dot(ang - ang.mean(), np.asarray(sign_ref) - np.mean(sign_ref)) < 0:
        ang = -ang
    return ang


def commanded_at(log, ts):
    """The board's linear ramp of the tested joint at times ts (deg)."""
    j = log["joint"]
    segs, a = [], log["start"][j]  # (t0, duration_s, from, to); each ramp starts where the last one is at t0
    for m in sorted(log["moves"], key=lambda m: m["t"]):
        t0, d, to = m["t"], m["duration_ms"] / 1000, m["angles"].get(j)
        if to is None:
            continue
        frm = _ramp(segs[-1], t0) if segs else a
        segs.append((t0, d, frm, to))
    return np.array([_ramp(next((s for s in reversed(segs) if s[0] <= t), None), t)
                     if any(s[0] <= t for s in segs) else a for t in ts])


def _ramp(seg, t):
    t0, d, frm, to = seg
    return to if d <= 0 or t >= t0 + d else frm + (to - frm) * (t - t0) / d


def measure(run_dir, cam=None, square_mm=None):
    """Board pose per frame -> measured.json {t, angle (relative deg), commanded}."""
    import cv2
    from . import vision
    run = Path(run_dir)
    log = json.loads((run / "log.json").read_text())
    cam = cam or vision.Camera.load()
    board = vision.make_board(square_mm)
    ts, Rs = [], []
    for f in log["frames"]:
        T = vision.board_pose(cv2.imread(str(run / f["file"])), cam, board)
        if T is not None:
            ts.append(f["t"]); Rs.append(T[:3, :3].T)  # camera orientation in the board frame
    if len(ts) < 10:
        raise SystemExit(f"board found in {len(ts)}/{len(log['frames'])} frames; is it in view?")
    ts = np.array(ts)
    cmd = commanded_at(log, ts)
    ang = rotation_angles(Rs, cmd - cmd[0])
    ang -= ang[:5].mean()  # relative to the settled start
    out = {"t": ts.tolist(), "angle": ang.tolist(), "commanded": (cmd - log["start"][log["joint"]]).tolist(),
           "found": f"{len(ts)}/{len(log['frames'])}"}
    (run / "measured.json").write_text(json.dumps(out))
    print(f"{run}: board in {out['found']} frames, range {ang.min():.1f}..{ang.max():.1f} deg")
    return out


# --- simulate + fit ------------------------------------------------------------------------
def simulate(log, ts, servo=None, delay=0.0, calib=None, masses=None, dt=0.004):
    """Replay the logged moves in MuJoCo; tested joint's angle (deg, relative to the settled
    start) at times ts - delay. Records on a dt grid and interpolates, so delay is smooth."""
    from .backends.mujoco import MujocoBackend
    from .mujoco_model import load_model
    calib = calib or Calibration.load()
    start, j = log["start"], log["joint"]
    be = MujocoBackend(calib=calib, start_pose=start,
                       model=load_model(servo=servo or {}, masses=masses))
    be.set_pulses({calib.servo(jj): calib.to_pulse(jj, a) for jj, a in start.items()}, 0)
    be.advance(1.0)  # gravity sag at the start pose
    a0 = be.joint_angles()[j]
    ts = np.asarray(ts, float)
    moves = sorted(log["moves"], key=lambda m: m["t"])
    t_sim0 = min([ts[0] - delay] + [m["t"] for m in moves]) - 0.05  # log time at be.data.time==1.0
    t_end = ts[-1]
    grid_t, grid_a, k = [], [], 0
    while True:
        t_log = t_sim0 + be.data.time - 1.0
        while k < len(moves) and moves[k]["t"] <= t_log + 1e-9:
            m = moves[k]
            be.set_pulses({calib.servo(jj): calib.to_pulse(jj, a) for jj, a in m["angles"].items()},
                          m["duration_ms"])
            k += 1
        grid_t.append(t_log); grid_a.append(be.joint_angles()[j] - a0)
        if t_log > t_end:
            break
        be.advance(dt)
    return np.interp(ts - delay, grid_t, grid_a)


def _load_run(run_dir):
    run = Path(run_dir)
    return json.loads((run / "log.json").read_text()), json.loads((run / "measured.json").read_text())


def fit(runs, servo0=None, delay0=0.05, fit_delay=True, verbose=True):
    """runs: [(log, measured)]. Returns ({joint: {kp, damping, torque}}, delay, report)."""
    from scipy.optimize import least_squares
    from .mujoco_model import load_params
    base = load_params()["servo"]
    for jj, v in (servo0 or {}).items():
        base[jj].update(v)
    joints = sorted({log["joint"] for log, _ in runs})
    names = [(jj, k) for jj in joints for k in FIT_KEYS]
    x0 = [math.log(min(max(base[jj][k], BOUNDS[k][0] * 1.01), BOUNDS[k][1] * 0.99)) for jj, k in names]
    lo = [math.log(BOUNDS[k][0]) for _, k in names]
    hi = [math.log(BOUNDS[k][1]) for _, k in names]
    if fit_delay:
        x0.append(delay0); lo.append(DELAY_BOUNDS[0]); hi.append(DELAY_BOUNDS[1])

    def unpack(x):
        s = {jj: {} for jj in joints}
        for (jj, k), v in zip(names, x):
            s[jj][k] = math.exp(v)
        return s, (x[-1] if fit_delay else delay0)

    def resid(x):
        s, d = unpack(x)
        r = []
        for log, meas in runs:
            sim = simulate(log, meas["t"], s, d)
            r.append(sim - np.asarray(meas["angle"]))
        return np.concatenate(r)

    def rms(x):
        return float(np.sqrt(np.mean(resid(x) ** 2)))


    before = rms(np.array(x0))
    # multi-start: kp vs damping vs torque has local minima (a saturated servo looks slow)
    best = None
    for kp_scale in (1.0, 0.3, 3.0):
        x = np.array(x0, float)
        for i, (_, k) in enumerate(names):
            if k == "kp":
                x[i] = np.clip(x[i] + math.log(kp_scale), lo[i] + 1e-6, hi[i] - 1e-6)
        res = least_squares(resid, x, bounds=(lo, hi), diff_step=0.02, x_scale="jac",
                            max_nfev=150, verbose=2 if verbose else 0)
        if best is None or res.cost < best.cost:
            best = res
    res = best
    s, d = unpack(res.x)
    after = float(np.sqrt(np.mean(res.fun ** 2)))
    return s, d, {"rms_before_deg": before, "rms_after_deg": after, "nfev": res.nfev}


def save_params(servo, delay, report, path=None):
    from .mujoco_model import PARAMS_PATH, load_params
    path = Path(path or PARAMS_PATH)
    cur = load_params(path)
    for jj, v in servo.items():
        cur["servo"][jj].update({k: round(float(x), 5) for k, x in v.items()})
    data = {"servo": cur["servo"], "masses": cur["masses"],
            "camera_delay_s": round(float(delay), 4),
            "fit": {"date": time.strftime("%Y-%m-%d"), "joints": sorted(servo),
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in report.items()}}}
    with open(path, "w") as f:
        f.write("# MuJoCo servo params fitted to the real arm by `python -m armpi.sysid fit`.\n"
                "# Overrides the defaults in mujoco_model.py. Delete to go back to the guesses.\n")
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def plot(run_dir, out=None, params_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .mujoco_model import PARAMS_PATH, SERVO, load_params
    log, meas = _load_run(run_dir)
    t = np.asarray(meas["t"]); t0 = t[0]
    path = params_path or PARAMS_PATH
    p = load_params(path)
    try:
        with open(path) as f:
            delay = (yaml.safe_load(f) or {}).get("camera_delay_s", 0.0)
    except FileNotFoundError:
        delay = 0.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t - t0, meas["commanded"], "k--", lw=1, label="commanded (board ramp)")
    ax.plot(t - t0, meas["angle"], ".", ms=3, label="measured (camera)")
    ax.plot(t - t0, simulate(log, t, {j: dict(v) for j, v in SERVO.items()}, 0.0,
                             masses={}), lw=1, label="sim: default guesses")
    ax.plot(t - t0, simulate(log, t, p["servo"], delay, masses=p["masses"]), lw=1.5,
            label=f"sim: {Path(path).name} (delay {delay * 1000:.0f}ms)")
    ax.set(xlabel="s", ylabel=f"{log['joint']} (deg, relative)", title=str(run_dir))
    ax.legend(); fig.tight_layout()
    out = out or str(Path(run_dir) / "fit.png")
    fig.savefig(out, dpi=120)
    print(f"-> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="armpi.sysid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("capture", help="run the step program on the real arm and film it")
    s.add_argument("--url", default="http://localhost:8000")
    s.add_argument("--joint", required=True, choices=[j for j in JOINTS if j != "gripper"])
    s.add_argument("--out", required=True)
    s.add_argument("--start", help='json overrides for the start pose, e.g. \'{"shoulder": 80}\'')
    s = sub.add_parser("measure", help="frames -> measured joint angle")
    s.add_argument("runs", nargs="+")
    s.add_argument("--square-mm", type=float)
    s = sub.add_parser("fit", help="fit servo params to measured runs -> sim_params.yaml")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out", help="params file (default armpi/sim_params.yaml)")
    s.add_argument("--no-delay", action="store_true", help="keep the camera delay fixed at 0.05s")
    s = sub.add_parser("plot", help="measured vs sim png per run")
    s.add_argument("runs", nargs="+")
    s.add_argument("--params", help="params file (default armpi/sim_params.yaml)")
    a = p.parse_args(argv)
    if a.cmd == "capture":
        input(f"arm will step the {a.joint} by up to 15 deg. board under the camera? Enter to go ")
        capture(a.url, a.joint, a.out, json.loads(a.start) if a.start else None)
    elif a.cmd == "measure":
        for r in a.runs:
            measure(r, square_mm=a.square_mm)
    elif a.cmd == "fit":
        servo, delay, rep = fit([_load_run(r) for r in a.runs], fit_delay=not a.no_delay)
        for jj, v in servo.items():
            print(f"{jj:8s} " + "  ".join(f"{k} {x:.4g}" for k, x in v.items()))
        print(f"camera delay {delay * 1000:.0f}ms; rms {rep['rms_before_deg']:.2f} -> {rep['rms_after_deg']:.2f} deg")
        print(f"-> {save_params(servo, delay, rep, a.out)}")
    elif a.cmd == "plot":
        for r in a.runs:
            plot(r, params_path=a.params)


if __name__ == "__main__":
    main()
