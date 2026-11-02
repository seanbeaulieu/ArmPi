"""Talks to armpi.server over HTTP, so Arm code on the PC drives the arm on the Pi.

    arm = armpi.Arm(RemoteBackend("http://armpi.local:8000"))

The server's own Arm applies the speed guard again, so a too-fast move from here
is slowed down there too. Stdlib only.
"""
import json
import urllib.error
import urllib.request

from .base import Backend


class RemoteBackend(Backend):
    def __init__(self, url="http://armpi.local:8000", timeout=5.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = json.loads(e.read() or b"{}").get("error", e.reason)
            raise (ValueError if e.code == 400 else OSError)(f"{path}: {msg}") from None

    def set_pulses(self, pulses, duration_ms):
        self._call("/api/servo_pulses", {"pulses": {str(s): int(us) for s, us in pulses.items()},
                                         "duration_ms": int(duration_ms)})

    def battery_mv(self):
        return self._call("/api/battery")["battery_mv"]

    def state(self):
        return self._call("/api/state")
