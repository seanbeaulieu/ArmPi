"""armpi.grasp pure parts on synthetic images (no arm, no camera)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from armpi import grasp


def _die_img(u, v, side, angle, pips=True):
    img = np.full((480, 640, 3), (190, 170, 185), np.uint8)  # pinkish paper
    cv2.line(img, (50, 400), (600, 380), (40, 40, 40), 3)      # a pen line
    box = cv2.boxPoints(((u, v), (side, side), angle)).astype(np.int32)
    cv2.fillPoly(img, [box], (120, 30, 20))                     # dark blue die (BGR)
    if pips:
        for du in (-0.25, 0.25):
            for dv in (-0.25, 0, 0.25):
                cv2.circle(img, (int(u + du * side), int(v + dv * side)), max(2, side // 12),
                           (230, 230, 230), -1)
    return img


@pytest.mark.parametrize("angle", [0, 10, 30, 60])
def test_find_die_center_size_angle(angle):
    d = grasp.find_die(_die_img(300, 250, 90, angle))
    assert d is not None
    assert math.dist((d["u"], d["v"]), (300, 250)) < 4
    assert abs(d["size"] - 90) < 10
    assert abs(grasp.angle_off(d["angle"], angle)) < 4


def test_find_die_none_on_empty_paper():
    img = np.full((480, 640, 3), (190, 170, 185), np.uint8)
    cv2.line(img, (50, 400), (600, 380), (40, 40, 40), 3)
    assert grasp.find_die(img) is None


def test_find_die_ignores_frame_cut_blob():
    assert grasp.find_die(_die_img(20, 250, 90, 0)) is None


def test_angle_off_wraps_square_symmetry():
    assert grasp.angle_off(89, 1) == pytest.approx(-2)
    assert grasp.angle_off(1, 89) == pytest.approx(2)
    assert grasp.angle_off(40, 0) == pytest.approx(40)
    assert grasp.angle_off(50, 0) == pytest.approx(-40)


def test_correction_inverts_gain():
    # pixel shift per mm: radial move -> die goes up in the image, tangential -> right
    J = [[0.5, 4.0], [-4.0, 0.3]]
    x, y = 95.0, -65.0
    rad, tan = grasp.gripper_frame(x, y)
    true_rt = np.array([3.0, -2.0])   # the move that would line it up
    pix = -(np.asarray(J) @ true_rt)  # die offset that move cancels
    dx, dy = grasp.correction_mm(J, pix[0], pix[1], x, y)
    want = true_rt[0] * rad + true_rt[1] * tan
    assert (dx, dy) == pytest.approx(tuple(want), abs=1e-6)


def test_correction_step_is_clamped():
    dx, dy = grasp.correction_mm([[4, 0], [0, 4]], 400, 0, 100, 0, max_step=10)
    assert math.hypot(dx, dy) == pytest.approx(10)


def test_arm_waits_for_the_longer_remote_duration():
    from armpi.arm import Arm
    from armpi.calibration import Calibration

    class SlowServer:
        def set_pulses(self, pulses, duration_ms):
            return duration_ms + 2000  # the Pi's own guard slowed it down

        def battery_mv(self):
            return 7400

    arm = Arm(SlowServer(), Calibration.load())
    _, ms = arm.move({"base": 10.0}, 500)
    assert ms >= 2500


class _FakeSkills:
    open_frac, move_ms, slow_ms, grip_depth = 1.0, 900, 700, 14.0

    def __init__(self):
        self.calls = []

    def wait(self, s):
        pass

    def _go(self, xyz, ms):
        self.calls.append(("go", tuple(round(v, 1) for v in xyz)))

    def gripper(self, frac, ms=None):
        self.calls.append(("grip", frac, ms))


def _grasper(angles):
    """Grasper whose camera sees the die in the box at each angle in turn (then gone)."""
    sk = _FakeSkills()
    seq = iter(angles)
    g = grasp.Grasper(sk, snapshot=lambda: b"", log=lambda *a: None,
                      ref={"box": [320, 300], "angle": 0.0, "J": [[0, 4], [-4, 0]]})
    g.photo = lambda tag, settle=0.6: tag
    g.die_here = lambda tag, allow_cut=False: (lambda a: None if a is None else
                              {"u": 320, "v": 300, "angle": a, "size": 90})(next(seq))
    return g, sk


def test_pick_squeezes_a_turned_die_then_grabs(monkeypatch):
    monkeypatch.setattr(grasp, "MAX_SQUEEZES", 2)  # off by default on the real arm
    g, sk = _grasper([35, 3, None])  # 35 deg off -> squeeze -> square -> gone after grab
    ok, why, _ = g.pick(95, -65)
    assert ok, why
    slow = [c for c in sk.calls if c[0] == "grip" and c[2] == grasp.SQUEEZE_MS]
    assert len(slow) == 1


def test_pick_square_die_no_squeeze():
    g, sk = _grasper([4, None])
    ok, _, _ = g.pick(95, -65)
    assert ok and not [c for c in sk.calls if c[0] == "grip" and c[2] == grasp.SQUEEZE_MS]


def test_pick_gives_up_when_squeezes_stall(monkeypatch):
    monkeypatch.setattr(grasp, "MAX_SQUEEZES", 2)
    g, sk = _grasper([44, 44, 44])
    ok, why, _ = g.pick(95, -65)
    assert not ok and "rotated" in why


def test_move_picks_places_and_confirms(monkeypatch):
    g, sk = _grasper([4, None])  # pick: lined up + square, then gone from the source
    monkeypatch.setattr(grasp, "find_die", lambda img, **kw: {"u": 300, "v": 310, "angle": 5, "size": 90})
    monkeypatch.setattr(grasp, "_cv2", lambda: cv2)
    g.photo = lambda tag, settle=0.6: np.zeros((48, 64, 3), np.uint8)
    ok, log, jpg = g.move((95, -65), (95, 65))
    assert ok and jpg and "die seen at the destination" in log[-1]
    goes = [c[1] for c in sk.calls if c[0] == "go"]
    assert any(abs(x - 95) < 1e-6 and abs(y - 65) < 1e-6 for x, y, _ in goes)  # went to dst


def test_move_stops_when_the_pick_misses():
    g, sk = _grasper([4, 4, 4, 4])  # die still there after the grab, and after the retry
    ok, log, jpg = g.move((95, -65), (95, 65))
    assert not ok and jpg is None and "missed" in log[-1]


def test_knocked_die_counts_as_missed_and_is_retried(monkeypatch):
    g, sk = _grasper([4, 4, 4, None])  # first grab knocks it (still in view); retry gets it
    monkeypatch.setattr(grasp, "find_die", lambda img, **kw: {"u": 300, "v": 310, "angle": 5, "size": 90})
    monkeypatch.setattr(grasp, "_cv2", lambda: cv2)
    g.photo = lambda tag, settle=0.6: np.zeros((48, 64, 3), np.uint8)
    ok, log, _ = g.move((95, -65), (95, 65))
    assert ok and any("missed" in l for l in log) and any(l.startswith("retry") for l in log)


def test_pick_refuses_when_the_fingers_would_tilt(monkeypatch):
    g, sk = _grasper([4])
    monkeypatch.setattr(g, "grab_tilt", lambda x, y: grasp.MAX_TILT + 5)
    ok, why, _ = g.pick(105, -82)
    assert not ok and "too far" in why
    assert not [c for c in sk.calls if c[0] == "grip" and c[1] == 0.0]  # never closed


def test_grab_tilt_real_model():
    from armpi.arm import Arm
    from armpi.calibration import Calibration
    from armpi.skills import Skills, true_z_to_cmd

    class NoBoard:
        def set_pulses(self, p, ms):
            return None

    sk = Skills(Arm(NoBoard(), Calibration.load()), wait=lambda s: None, z_map=true_z_to_cmd)
    g = grasp.Grasper(sk, snapshot=None, ref={"box": [0, 0], "angle": 0, "J": [[1, 0], [0, 1]]},
                      log=lambda *a: None)
    assert g.grab_tilt(95, -65) < grasp.MAX_TILT       # square centre: near vertical
    assert g.grab_tilt(109, -88) > 10                  # the far corner: the wrist tilts


def test_ref_for_blends_near_and_far_by_distance():
    near = {"box": [320, 320], "angle": 0.0, "J": [[0, 10], [13, 0]], "grab_forward_mm": 1.5,
            "taught_at": [95.0, -65.0]}
    far = {"box": [330, 370], "angle": 84.0, "J": [[0, 8], [3, 0]], "grab_forward_mm": 0.5,
           "taught_at": [127.1, -86.9]}
    g = grasp.Grasper(None, None, ref=near, log=lambda *a: None)
    g.far = far
    assert g.ref_for(95, -65)["box"] == pytest.approx([320, 320], abs=0.5)
    assert g.ref_for(127.1, -86.9)["J"][1][0] == pytest.approx(3)
    mid = g.ref_for(111, -76)                           # about halfway out
    assert 320 < mid["box"][1] < 370 and 3 < mid["J"][1][0] < 13
    assert -6 <= mid["angle"] <= 0                      # 0 -> 84 deg is -6 deg the short way
    assert g.ref_for(200, -140)["box"] == pytest.approx([330, 370])  # clamped beyond far


def test_view_calibration_round_trip(monkeypatch):
    """Fake camera: the die's pixel is a perspective-ish function of where the die sits in
    the camera pose's own frame. calibrate_view must invert it, and locate must find a die
    anywhere in view from the fixed pose."""
    monkeypatch.setattr(grasp, "_cv2", lambda: cv2)
    sk = _FakeSkills()
    pose = {"xy": (0.0, 0.0)}
    sk._go = lambda xyz, ms: pose.update(xy=xyz[:2])
    die = {"xy": (95.0, -65.0)}

    def project(a, b):  # a: mm ahead of the camera pose, b: mm to its left
        w = 1.0 + 0.004 * (a - 25)
        return 320 - 9.8 * b / w, 323 - 13.0 * (a - 25) / w

    def die_here(tag, allow_cut=False):
        px, py = pose["xy"]
        r, t = grasp.gripper_frame(px, py)
        rel = np.array(die["xy"]) - np.array([px, py])
        u, v = project(float(rel @ r), float(rel @ t))
        return {"u": u, "v": v, "angle": 0.0, "size": 120}

    g = grasp.Grasper(sk, snapshot=None, ref={"box": [320, 323], "angle": 0.0,
                                              "J": [[0, 10], [13, 0]], "taught_at": [95, -65]},
                      log=lambda *a: None)
    g.die_here = die_here
    view = g.calibrate_view(95, -65)
    assert view["points"] == len(grasp.VIEW_GRID) and view["max_err_mm"] < 0.5
    for target in [(95, -65), (110, -78), (125, -88), (92, -55)]:
        die["xy"] = target
        xy, _ = g.locate(95, -65)
        assert math.dist(xy, target) < 1.0, (target, xy)
