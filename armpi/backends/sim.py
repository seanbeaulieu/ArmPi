"""Simulated board: same interface as I2CBackend, no hardware.

Mimics what the MCU does: each servo moves linearly from where it is to the new pulse over
the requested time. pulses_at(t) gives the interpolated state, so viewers and tests can
see motion, not just end points.
"""
import time

from .. import protocol
from .base import Backend


class SimBackend(Backend):
    def __init__(self, start_pulses=None, battery_mv=7400, clock=time.monotonic):
        self.clock = clock
        self.battery = battery_mv
        # servo id -> (from_pulse, to_pulse, t_start, duration_s)
        self.state = {sid: (us, us, 0.0, 0.0) for sid, us in (start_pulses or {}).items()}
        self.history = []  # (t, pulses, duration_ms), for tests and debugging

    def set_pulses(self, pulses, duration_ms):
        protocol.encode_servo_move(pulses, duration_ms)  # same validation as the real board
        now = self.clock()
        current = self.pulses_at(now)
        for sid, us in pulses.items():
            start = current.get(sid, us)  # unknown servo: jumps, like a real servo on power-up
            self.state[sid] = (start, int(round(us)), now, duration_ms / 1000)
        self.history.append((now, dict(pulses), duration_ms))

    def pulses_at(self, t=None):
        t = self.clock() if t is None else t
        out = {}
        for sid, (a, b, t0, dur) in self.state.items():
            f = 1.0 if dur <= 0 else min(max((t - t0) / dur, 0.0), 1.0)
            out[sid] = a + (b - a) * f
        return out

    def battery_mv(self):
        return self.battery
