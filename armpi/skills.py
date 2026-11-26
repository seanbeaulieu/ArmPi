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

import numpy as np

from .calibration import JOINTS

# Real arm, 2026-11-26: commanded fingertip z -> measured gap to the paper (mm), stepped down
# over (95, -65) with pitch -90. The kinematic z is ~35% too small (shoulder/elbow calibration
# scale, not a fixed offset); until the joints are refit, `true_z_to_cmd` inverts this table.
Z_MEASURED = [(-5.0, 4.0), (0.0, 9.5), (5.0, 13.5), (10.0, 20.0), (15.0, 29.0), (20.0, 38.0)]


def true_z_to_cmd(z, table=Z_MEASURED):
    """Real fingertip height (mm) -> the z to command. Interpolated inside the table; below it,
    the bottom segment's slope; above it, the top offset (heights above ~40 mm are unmeasured)."""
    cmd, real = np.array(table).T
    if z <= real[0]:
        k = (cmd[1] - cmd[0]) / (real[1] - real[0])
        return float(cmd[0] + k * (z - real[0]))
    if z >= real[-1]:
        return float(z - (real[-1] - cmd[-1]))
    return float(np.interp(z, real, cmd))


class Skills:
    def __init__(self, arm, wait=time.sleep, hover_z=60.0, grip_depth=14.0, min_z=4.0,
                 pitch=-90.0, open_frac=1.0, move_ms=900, slow_ms=700, grip_ms=500, settle=0.25,
                 z_map=None):
        self.arm = arm
        self.z_map = z_map  # real z -> commanded z (true_z_to_cmd on the real arm); None = exact
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
    def cmd_z(self, z):
        return self.z_map(z) if self.z_map else z

    def _go(self, xyz, ms, raw=False):
        """Fingertip to xyz (real height, unless raw: the kinematic z as is); waits for the move
        (speed guard included) plus settle."""
        from .kinematics import ik_search
        limits = {j: self.arm.calib.limits(j) for j in JOINTS if j != "gripper"}
        if not raw:
            xyz = (xyz[0], xyz[1], self.cmd_z(xyz[2]))
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
        return all(ik_search((x, y, self.cmd_z(zz)), self.pitch, limits) for zz in zs)

    def grip_z(self, obj_h):
        return max(obj_h - self.grip_depth, self.min_z)

    # --- primitives ---
    def home(self):
        """Gripper up and out of the camera's way."""
        self.log.append(("home", ()))
        self._move({j: a for j, a in self.arm.calib.rest().items() if j != "gripper"}, 1200)

    def move_to(self, x, y, z, raw=False):
        self.log.append(("move_to", (x, y, z)))
        return self._go((x, y, z), self.move_ms, raw)

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
