"""MuJoCo backend: the SimBackend's board model (linear pulse ramps, like the MCU) driving a
physics model of the arm through position-controlled servos.

The commanded ramp is what the real board does; the MuJoCo state is what the joints would
actually do under gravity and torque limits. The gap between the two is the sim2real story
for an arm with no position feedback.

    from armpi import Arm
    from armpi.backends.mujoco import MujocoBackend
    be = MujocoBackend()
    arm = Arm(be)
    arm.move({"base": 30}, 800)
    be.advance(1.0)             # step physics 1s (or be.sync() to catch up to wall clock)
    print(be.joint_angles())    # simulated, in degrees
"""
import math

from ..calibration import JOINTS, Calibration
from .sim import SimBackend


class MujocoBackend(SimBackend):
    def __init__(self, calib=None, start_pose=None, battery_mv=7400, realtime=False, model=None):
        """realtime=False: time only moves when you call advance() (deterministic, for
        scripts/tests/training). realtime=True: wall clock; call sync() to catch up."""
        import time
        import mujoco  # optional dependency: pip install mujoco
        self.mujoco = mujoco
        self.calib = calib or Calibration.load()
        clock = time.monotonic if realtime else (lambda: self._t0 + self.data.time)
        self._t0 = 0.0
        super().__init__(battery_mv=battery_mv, clock=clock)
        if model is None:
            from ..mujoco_model import load_model
            model = load_model()
        self.model = model
        self.data = mujoco.MjData(model)
        self._qadr = {j: model.joint(j).qposadr[0] for j in JOINTS if j != "gripper"}
        self._act = {j: model.actuator(j).id for j in JOINTS}
        self._grip_max = model.jnt_range[model.joint("finger_l").id][1]
        self.servo_to_joint = {self.calib.servo(j): j for j in JOINTS}
        pose = start_pose or self.calib.rest()
        for j, a in pose.items():
            self._set_ctrl(j, a)
            if j in self._qadr:
                self.data.qpos[self._qadr[j]] = math.radians(a)
        mujoco.mj_forward(model, self.data)
        if realtime:
            self._t0 = time.monotonic()  # wall time that matches data.time == 0

    # --- conversions ---
    def _set_ctrl(self, joint, angle):
        if joint == "gripper":
            self.data.ctrl[self._act[joint]] = min(max(angle, 0.0), 1.0) * self._grip_max
        else:
            self.data.ctrl[self._act[joint]] = math.radians(angle)

    def _apply_commanded(self, t):
        for sid, us in self.pulses_at(t).items():
            j = self.servo_to_joint.get(sid)
            if j:
                self._set_ctrl(j, self.calib.to_angle(j, us))

    # --- stepping ---
    def advance(self, seconds):
        """Step physics forward by `seconds` of sim time, following the commanded ramps."""
        end = self.data.time + seconds
        while self.data.time < end - 1e-12:
            self._apply_commanded(self._t0 + self.data.time)
            self.mujoco.mj_step(self.model, self.data)
        return self.data.time

    def sync(self):
        """Catch the physics up to the backend clock (wall time by default)."""
        return self.advance(max(0.0, self.clock() - self._t0 - self.data.time))

    # --- state ---
    def joint_angles(self):
        """Simulated joint angles (degrees; gripper 0..1), i.e. where the arm really is."""
        out = {j: math.degrees(self.data.qpos[a]) for j, a in self._qadr.items()}
        out["gripper"] = self.data.qpos[self.model.joint("finger_l").qposadr[0]] / self._grip_max
        return out

    def fingertip_mm(self):
        return tuple(1000.0 * x for x in self.data.site("fingertip").xpos)
