"""HTTP server on the Pi: the Arm API as JSON, the camera as MJPEG, and a web control panel.

    python -m armpi.server            # simulator, http://armpi.local:8000
    python -m armpi.server --real     # start on the real arm

The panel switches between sim and real at runtime (POST /api/backend). Each keeps its own
Arm, so switching back finds the pose where it was left; both share one Calibration.
Stdlib only (plus opencv for the camera). No auth: anyone on the LAN can move the arm.

    GET  /                     web panel
    GET  /api/state            angles (commanded + live), pulses, fk, battery, calibration
    GET  /api/battery          from the real board whenever it answers, even in sim mode
    POST /api/backend          {"name": "sim" | "i2c"}
    POST /api/move             {"angles": {"base": 10, ...}, "duration_ms": 500}
    POST /api/move_to          {"xyz": [x, y, z], "pitch": -90, "duration_ms": 800}
    POST /api/pulse            {"joint": "base", "us": 1500, "duration_ms": 300}   raw, no angle limits
    POST /api/servo_pulses     {"pulses": {"6": 1500}, "duration_ms": 300}         used by RemoteBackend
    POST /api/rest             {"duration_ms": 1500}
    POST /api/calibrate        {"joint": "shoulder", "mark": "zero", "angle": 90}  see Calibrator
    GET  /stream.mjpg, /snapshot.jpg
"""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .arm import Arm
from .backends import make_backend
from .calibration import JOINTS, Calibration
from .kinematics import fk

WEB = Path(__file__).with_name("web")


class Camera:
    """Grabs frames in a background thread once someone asks for one; keeps the latest JPEG."""

    def __init__(self, device=0, width=640, height=480, quality=80):
        self.device, self.size, self.quality = device, (width, height), quality
        self.jpeg = None
        self.error = None
        self.cond = threading.Condition()
        self.thread = None

    def _run(self):
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if not cap.isOpened():
            self.error = f"cannot open /dev/video{self.device}"
            with self.cond:
                self.cond.notify_all()
            return
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def frame(self, timeout=5.0, after=None):
        """Latest JPEG (a newer one than `after` if given), or None if the camera failed."""
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        with self.cond:
            self.cond.wait_for(lambda: self.error or (self.jpeg is not None and self.jpeg is not after),
                               timeout)
            return self.jpeg


class Calibrator:
    """Records the joint's current (last commanded) pulse into calibration.yaml.

    marks: zero  -> this pulse is `angle` deg (default: the joint's zero_angle)
           min / max -> this pulse is the joint's lower / upper limit
           flip  -> reverse the joint's direction (sign)
           rest  -> this angle becomes the rest pose
           closed / open -> gripper end points
    """

    def __init__(self, arm):
        self.arm = arm

    def mark(self, joint, mark, angle=None):
        cal = self.arm.calib
        j = cal.joints[joint]
        if mark == "flip":
            j["sign"] = -j["sign"]
        else:
            if joint not in self.arm.pulses:
                raise ValueError(f"move {joint} first: its position is unknown")
            us = self.arm.pulses[joint]
            if mark == "zero" and joint != "gripper":
                j["zero_pulse"] = us - j["trim"]
                if angle is not None:
                    j["zero_angle"] = float(angle)
            elif mark in ("min", "max"):
                a = round(cal.to_angle(joint, us), 1)
                j["limits"][0 if mark == "min" else 1] = a
                j["limits"] = sorted(j["limits"])
            elif mark == "rest":
                j["rest"] = round(cal.to_angle(joint, us), 1)
            elif mark in ("closed", "open") and joint == "gripper":
                j[f"{mark}_pulse"] = us - j["trim"]
            else:
                raise ValueError(f"bad mark {mark!r} for {joint}")
        cal.save()
        return j


class App:
    BATTERY_EVERY_S = 3.0  # the I2C bus is slow and flaky; don't read it on every refresh

    def __init__(self, calib, camera, backend="sim", make_backend=make_backend):
        self.calib, self.camera, self.make_backend = calib, camera, make_backend
        self.lock = threading.Lock()
        self.arms = {}  # backend name -> Arm, created on first use
        self.board = None  # I2C backend, shared by the real Arm and the battery readout
        self.board_error = None
        self.battery_cache = (-1e9, None, None)  # (time, mv, source)
        self.calibrator = Calibrator(None)
        self.servo_to_joint = {calib.servo(j): j for j in JOINTS}
        self.use(backend)

    def _board(self):
        if self.board is None:
            try:
                self.board = self.make_backend("i2c")
                self.board_error = None
            except Exception as e:  # no smbus2, no /dev/i2c-1, ...
                self.board_error = f"{type(e).__name__}: {e}"
        return self.board

    def use(self, name):
        """Switch the backend the API drives. Raises OSError if the board can't be opened."""
        if name not in ("sim", "i2c"):
            raise ValueError(f"backend must be 'sim' or 'i2c', not {name!r}")
        if name not in self.arms:
            if name == "i2c":
                if self._board() is None:
                    raise OSError(f"cannot open the board: {self.board_error}")
                backend = self.board
            else:
                backend = self.make_backend("sim")
            self.arms[name] = Arm(backend, self.calib)
        self.backend_name, self.arm = name, self.arms[name]
        self.calibrator.arm = self.arm

    def state(self):
        with self.lock:
            commanded = self.arm.angles
            pulses = dict(self.arm.pulses)
            live = self._live_angles(commanded)
            battery, source = self._battery()
        return {"backend": self.backend_name, "angles": commanded, "live": live, "pulses": pulses,
                "fk": _fk(live), "target_fk": _fk(commanded),
                "battery_mv": battery, "battery_source": source, "board_error": self.board_error,
                "joints": JOINTS, "calibration": self.calib.joints}

    def _live_angles(self, commanded):
        """The sim knows where each servo is mid-move; the real arm can't tell us."""
        pulses_at = getattr(self.arm.backend, "pulses_at", None)
        if pulses_at is None:
            return commanded
        now = pulses_at()
        return {j: self.calib.to_angle(j, now[self.calib.servo(j)])
                for j in commanded if self.calib.servo(j) in now}

    def _battery(self):
        """Real board's voltage if it answers (the arm's battery matters even in sim mode),
        else the sim's. Cached for BATTERY_EVERY_S."""
        t, mv, source = self.battery_cache
        if time.monotonic() - t < self.BATTERY_EVERY_S:
            return mv, source
        mv, source = None, None
        try:
            if self._board() is not None:
                mv, source = self.board.battery_mv(), "board"
            else:
                mv, source = self.arm.battery_mv(), self.backend_name
        except OSError:
            pass
        self.battery_cache = (time.monotonic(), mv, source)
        return mv, source

    def post(self, path, body):
        d = body.get("duration_ms")
        with self.lock:
            if path == "/api/backend":
                self.use(body["name"])
                return {"backend": self.backend_name}
            if path == "/api/move":
                angles, ms = self.arm.move({k: float(v) for k, v in body["angles"].items()}, d)
                return {"angles": angles, "duration_ms": ms}
            if path == "/api/move_to":
                pitch = self.arm.move_to(tuple(body["xyz"]), body.get("pitch", -90), d)
                return {"pitch": pitch, "angles": self.arm.angles}
            if path == "/api/pulse":
                return {"duration_ms": self.arm.move_pulse(body["joint"], int(body["us"]), d)}
            if path == "/api/servo_pulses":
                pulses = {self.servo_to_joint[int(s)]: int(us) for s, us in body["pulses"].items()}
                return {"duration_ms": self.arm.send_pulses(pulses, d)}
            if path == "/api/rest":
                angles, ms = self.arm.rest(d)
                return {"angles": angles, "duration_ms": ms}
            if path == "/api/calibrate":
                return self.calibrator.mark(body["joint"], body["mark"], body.get("angle"))
        raise KeyError(path)


def _fk(angles):
    if all(j in angles for j in ("base", "shoulder", "elbow", "wrist")):
        return fk(angles)
    return None


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            if ctype == "application/json":
                body = json.dumps(body, default=_json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, app.state())
            if path == "/api/battery":
                with app.lock:
                    mv, source = app._battery()
                return self._send(200, {"battery_mv": mv, "source": source})
            if path == "/snapshot.jpg":
                jpg = app.camera.frame()
                return self._send(200, jpg, "image/jpeg") if jpg else self._send(503, {"error": app.camera.error})
            if path == "/stream.mjpg":
                return self._stream()
            self._send(404, {"error": "not found"})

        def _stream(self):
            if not app.camera.frame():
                return self._send(503, {"error": app.camera.error})
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    jpg = app.camera.frame(after=last)
                    if jpg is None:
                        return
                    if jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = jpg
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, app.post(self.path, body))
            except KeyError as e:
                self._send(404 if str(e).strip("'").startswith("/") else 400, {"error": f"missing/unknown {e}"})
            except (ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except OSError as e:
                self._send(502, {"error": f"board: {e}"})

    return Handler


def _json_default(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def main(argv=None):
    p = argparse.ArgumentParser(prog="armpi.server")
    p.add_argument("--real", action="store_true", help="start on the real arm (default: sim)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--camera", type=int, default=0, help="/dev/videoN")
    a = p.parse_args(argv)
    app = App(Calibration.load(), Camera(a.camera), "i2c" if a.real else "sim")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    srv.daemon_threads = True
    print(f"armpi server ({app.backend_name}) on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
