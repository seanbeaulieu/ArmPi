"""MuJoCo model + backend. Skipped when mujoco isn't installed (e.g. on the Pi)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

mujoco = pytest.importorskip("mujoco")

from armpi import Arm
from armpi.backends.mujoco import MujocoBackend
from armpi.kinematics import fk
from armpi.mujoco_model import load_model


def test_model_matches_fk():
    """MJCF geometry must agree with kinematics.fk (same measured links, same conventions)."""
    m = load_model()
    d = mujoco.MjData(m)
    for ang in [{"base": 30, "shoulder": 60, "elbow": -40, "wrist": -50},
                {"base": -75, "shoulder": 120, "elbow": -100, "wrist": 20},
                {"base": 0, "shoulder": 90, "elbow": 0, "wrist": 0}]:
        for j, a in ang.items():
            d.qpos[m.joint(j).qposadr[0]] = math.radians(a)
        mujoco.mj_forward(m, d)
        got = [x * 1000 for x in d.site("fingertip").xpos]
        want = fk(ang)["fingertip"]
        assert all(abs(g - w) < 1e-6 for g, w in zip(got, want)), (ang, got, want)


def test_backend_tracks_commands():
    be = MujocoBackend()
    arm = Arm(be)
    _, ms = arm.move({"base": 30, "shoulder": 90, "elbow": 0, "wrist": 0, "gripper": 1}, 1000)
    be.advance(ms / 1000 + 1.0)
    got = be.joint_angles()
    for j, want in {"base": 30, "shoulder": 90, "elbow": 0, "wrist": 0}.items():
        assert abs(got[j] - want) < 2.0, (j, got[j])
    assert got["gripper"] > 0.9
    tip, want = be.fingertip_mm(), fk(arm.angles)["fingertip"]
    assert all(abs(a - b) < 5 for a, b in zip(tip, want))


def test_ramp_is_gradual():
    """Mid-move the joint should be roughly mid-way, like the board's linear ramp."""
    be = MujocoBackend(start_pose={"base": 0, "shoulder": 90, "elbow": 0, "wrist": 0, "gripper": 0})
    arm = Arm(be)
    arm.move({"base": 0, "shoulder": 90, "elbow": 0, "wrist": 0, "gripper": 0}, 500)
    be.advance(0.6)
    _, ms = arm.move({"base": 60}, 2000)
    be.advance(ms / 2000)
    assert 15 < be.joint_angles()["base"] < 45
