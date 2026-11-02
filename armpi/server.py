"""HTTP server on the Pi: the Arm API as JSON, the camera as MJPEG, and a web control panel.

    python -m armpi.server            # real arm, http://armpi.local:8000
    python -m armpi.server --sim      # simulator (no servo power needed)

Stdlib only (plus opencv for the camera). No auth: anyone on the LAN can move the arm.

    GET  /                     web panel
    GET  /api/state            angles, pulses, fk points, battery, calibration
    GET  /api/battery
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

from . import connect
from .calibration import JOINTS
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
    def __init__(self, arm, camera, backend_name):
        self.arm, self.camera, self.backend_name = arm, camera, backend_name
        self.lock = threading.Lock()
        self.calibrator = Calibrator(arm)
        self.servo_to_joint = {arm.calib.servo(j): j for j in JOINTS}

    def state(self):
        with self.lock:
            angles = self.arm.angles
            pulses = dict(self.arm.pulses)
            battery = self._battery()
        points = None
        if all(j in angles for j in ("base", "shoulder", "elbow", "wrist")):
            points = {k: v for k, v in fk(angles).items()}
        return {"backend": self.backend_name, "angles": angles, "pulses": pulses, "fk": points,
                "battery_mv": battery, "joints": JOINTS, "calibration": self.arm.calib.joints}

    def _battery(self):
        try:
            return self.arm.battery_mv()
        except OSError:
            return None

    def post(self, path, body):
        d = body.get("duration_ms")
        with self.lock:
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
                    return self._send(200, {"battery_mv": app._battery()})
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
    p.add_argument("--sim", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--camera", type=int, default=0, help="/dev/videoN")
    a = p.parse_args(argv)
    name = "sim" if a.sim else "i2c"
    app = App(connect(name), Camera(a.camera), name)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    srv.daemon_threads = True
    print(f"armpi server ({name}) on http://{a.host}:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
