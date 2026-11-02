"""Arm: named joints in degrees on top of any Backend.

The servos report nothing back, so everything here is "last commanded", not measured.
"""
import math
import time

from . import kinematics, protocol
from .calibration import JOINTS, Calibration


class Arm:
    def __init__(self, backend, calib=None):
        self.backend = backend
        self.calib = calib or Calibration.load()
        self.pulses = {}  # joint -> last commanded pulse (trim included)

    # --- joint space ---
    def move(self, angles, duration_ms=None, wait=False):
        """Move some or all joints. Angles outside the limits are clamped.

        duration_ms is raised if needed so no servo moves faster than calib.min_ms_per_us.
        Returns (clamped_angles, duration_ms actually used).
        """
        target = {}
        for joint, a in angles.items():
            if joint not in JOINTS:
                raise KeyError(f"unknown joint {joint!r}; expected one of {JOINTS}")
            if not math.isfinite(a):
                raise ValueError(f"{joint} angle is {a}")
            target[joint] = self.calib.clamp(joint, a)
        pulses = {j: self.calib.to_pulse(j, a) for j, a in target.items()}
        duration_ms = self.send_pulses(pulses, duration_ms)
        if wait:
            time.sleep(duration_ms / 1000)
        return target, duration_ms

    def move_pulse(self, joint, pulse, duration_ms=None, wait=False):
        """Raw pulse (trim included) for one joint, bypassing angle limits. For calibration."""
        if not protocol.PULSE_MIN <= pulse <= protocol.PULSE_MAX:
            raise ValueError(f"pulse {pulse} outside [{protocol.PULSE_MIN}, {protocol.PULSE_MAX}]")
        duration_ms = self.send_pulses({joint: int(pulse)}, duration_ms)
        if wait:
            time.sleep(duration_ms / 1000)
        return duration_ms

    def rest(self, duration_ms=None, wait=False):
        """The folded pose the arm boots into."""
        return self.move(self.calib.rest(), duration_ms, wait)

    def send_pulses(self, pulses, duration_ms=None):
        """{joint: pulse} (trim included), with the speed guard. Returns duration used."""
        known = [abs(us - self.pulses[j]) for j, us in pulses.items() if j in self.pulses]
        if len(known) < len(pulses):
            floor = self.calib.first_move_ms  # real position unknown: go slow
        else:
            floor = max(known) * self.calib.min_ms_per_us
        duration_ms = int(max(duration_ms or 0, floor))
        self.backend.set_pulses({self.calib.servo(j): us for j, us in pulses.items()}, duration_ms)
        self.pulses.update(pulses)
        return duration_ms

    @property
    def angles(self):
        """Last commanded angle of every joint that has been commanded."""
        return {j: self.calib.to_angle(j, us) for j, us in self.pulses.items()}

    # --- cartesian ---
    def move_to(self, xyz, pitch=-90, duration_ms=None, wait=False):
        """Put the fingertip at xyz (mm). Uses the pitch closest to the one asked for that
        stays inside the joint limits. Returns the pitch used; raises if unreachable."""
        limits = {j: self.calib.limits(j) for j in JOINTS if j != "gripper"}
        sol = kinematics.ik_search(xyz, pitch, limits)
        if sol is None:
            raise ValueError(f"{xyz} unreachable within joint limits")
        angles, used = sol
        self.move(angles, duration_ms, wait)
        return used

    def fk(self):
        return kinematics.fk(self.angles)

    # --- misc ---
    def battery_mv(self):
        return self.backend.battery_mv()

    def close(self):
        self.backend.close()
