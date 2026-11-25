"""Pick and place primitives on top of Arm: hover, descend, grip, lift.

Everything is open loop (the servos report nothing), so each primitive is a fixed sequence of
Cartesian waypoints with generous settle time. `wait` is how time passes: time.sleep on the
real arm, MujocoBackend.advance in a deterministic sim.

    from armpi.skills import Skills
    sk = Skills(arm)                  # real arm (or Skills.for_mujoco(backend))
    sk.pick(110, 60)                  # object centered at x=110, y=60 (mm, base frame)
    sk.place(110, -60)
    sk.home()

Heights are fingertip z above the table (mm). The fingers are 50 mm long, so for a cube of
height h the fingertip goes to about h - grip_depth (clamped >= min_z) to grab it mid-side.
"""
import time

from .calibration import JOINTS


class Skills:
    def __init__(self, arm, wait=time.sleep, hover_z=60.0, grip_depth=14.0, min_z=4.0,
                 pitch=-90.0, open_frac=1.0, move_ms=900, slow_ms=700, grip_ms=500, settle=0.25):
        self.arm = arm
        self.wait = wait
        self.hover_z = hover_z
        self.grip_depth = grip_depth
        self.min_z = min_z
        self.pitch = pitch
        self.open_frac = open_frac
        self.move_ms = move_ms
        self.slow_ms = slow_ms
        self.grip_ms = grip_ms
        self.settle = settle
        self.holding = False
        self.log = []  # (primitive, args) in order, for episode logs and debugging

    @classmethod
    def for_mujoco(cls, backend, arm=None, **kw):
        from .arm import Arm
        arm = arm or Arm(backend, backend.calib)
        return cls(arm, wait=backend.advance, **kw)

    # --- low level ---
    def _go(self, xyz, ms):
        """Fingertip to xyz; waits for the move (speed guard included) plus settle."""
        from .kinematics import ik_search
        limits = {j: self.arm.calib.limits(j) for j in JOINTS if j != "gripper"}
        sol = ik_search(xyz, self.pitch, limits)
        if sol is None:
            raise ValueError(f"{tuple(round(v) for v in xyz)} unreachable within joint limits")
        angles, pitch = sol
        self._move(angles, ms)
        return pitch

    def _move(self, angles, ms):
        _, ms = self.arm.move(angles, ms)
        self.wait(ms / 1000 + self.settle)

    def gripper(self, frac):
        self._move({"gripper": frac}, self.grip_ms)

    def reachable(self, x, y, z=None):
        """True if (x, y) can be reached at the hover height and the grip height."""
        from .kinematics import ik_search
        limits = {j: self.arm.calib.limits(j) for j in JOINTS if j != "gripper"}
        zs = [self.hover_z, self.min_z if z is None else z]
        return all(ik_search((x, y, zz), self.pitch, limits) for zz in zs)

    def grip_z(self, obj_h):
        return max(obj_h - self.grip_depth, self.min_z)

    # --- primitives ---
    def home(self):
        """Gripper up and out of the camera's way."""
        self.log.append(("home", ()))
        self._move({j: a for j, a in self.arm.calib.rest().items() if j != "gripper"}, 1200)

    def move_to(self, x, y, z):
        self.log.append(("move_to", (x, y, z)))
        return self._go((x, y, z), self.move_ms)

    def pick(self, x, y, obj_h=20.0):
        """Grab an object of height obj_h (mm) centered at (x, y)."""
        self.log.append(("pick", (x, y, obj_h)))
        self.gripper(self.open_frac)
        self._go((x, y, self.hover_z), self.move_ms)
        self._go((x, y, self.grip_z(obj_h)), self.slow_ms)
        self.gripper(0.0)
        self._go((x, y, self.hover_z), self.slow_ms)
        self.holding = True

    def place(self, x, y, obj_h=20.0, drop=2.0):
        """Set the held object down centered at (x, y), `drop` mm above the table."""
        self.log.append(("place", (x, y, obj_h)))
        self._go((x, y, self.hover_z), self.move_ms)
        self._go((x, y, self.grip_z(obj_h) + drop), self.slow_ms)
        self.gripper(self.open_frac)
        self._go((x, y, self.hover_z), self.slow_ms)
        self.holding = False
