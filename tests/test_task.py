"""Square -> circle: layout geometry, perception on synthetic images, and the full scripted
task in MuJoCo (skipped without mujoco / opencv)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from armpi.task import Layout, score, slots


def test_slots_inside_circle_and_apart():
    lay = Layout()
    s = slots(lay, 20)
    assert len(s) >= 5
    for p in s:
        assert lay.in_circle(*p, margin=20 * 0.7)
    for i, p in enumerate(s):
        for q in s[i + 1:]:
            assert math.dist(p, q) >= 20


def test_square_membership_rotated():
    lay = Layout(square_c=(100, 50), square_side=40, square_yaw=45)
    assert lay.in_square(100, 50)
    assert lay.in_square(100 + 27, 50)        # along a diagonal: corner is 28.3 out
    assert not lay.in_square(100 + 19, 50 + 19)  # along an edge normal: edge is 20 out
    assert score(lay, [{"xyz": lay.circle_c + (0,)}, {"xyz": (0, 0, 0)}])["in_circle"] == 1


def test_layout_reachable():
    """Every point of the default square and circle reachable at grip and hover height."""
    from armpi.calibration import Calibration
    from armpi.kinematics import ik_search
    cal = Calibration.load()
    lim = {j: cal.limits(j) for j in ("base", "shoulder", "elbow", "wrist")}
    lay = Layout()
    for c, r in ((lay.square_c, lay.square_side / 2), (lay.circle_c, lay.circle_r)):
        for dx in range(-int(r), int(r) + 1, 5):
            for dy in range(-int(r), int(r) + 1, 5):
                p = (c[0] + dx, c[1] + dy)
                if lay.in_square(*p) or lay.in_circle(*p):
                    for z in (6, 60):
                        assert ik_search((*p, z), -90, lim), (p, z)


def _synthetic(cv2, np):
    """Top-down photo: 1 px = 0.5 mm, table x up the image, y to the left. Returns
    (image, to_table, truth)."""
    img = np.full((480, 640, 3), (60, 70, 80), np.uint8)       # table
    cv2.rectangle(img, (40, 20), (600, 460), (235, 235, 230), -1)  # paper
    def px(x, y):  # table mm -> pixel
        return int(320 - 2 * y), int(470 - 2 * (x - 0))
    sq_c, sq_side, ci_c, ci_r = (150, 55), 60, (150, -60), 40
    half = sq_side
    cx, cy = px(*sq_c)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (20, 20, 20), 3)
    cv2.circle(img, px(*ci_c), 2 * ci_r, (20, 20, 20), 3)
    cubes = [(140, 45), (160, 68), (140, 70), (150, -60)]  # last one already in the circle
    colors = [(40, 40, 220), (40, 180, 40), (220, 80, 30), (0, 200, 230)]
    for (x, y), col in zip(cubes, colors):
        u, v = px(x, y)
        cv2.rectangle(img, (u - 18, v - 18), (u + 18, v + 18), col, -1)
    def to_table(uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.column_stack([(470 - uv[:, 1]) / 2, (320 - uv[:, 0]) / 2, np.zeros(len(uv))])
    return img, to_table, {"square": (sq_c, sq_side), "circle": (ci_c, ci_r), "cubes": cubes}


def test_perceive_synthetic():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from armpi import scene
    img, to_table, truth = _synthetic(cv2, np)
    s = scene.perceive(img, to_table)
    assert s["square"] and s["circle"]
    assert math.dist(s["square"]["center"], truth["square"][0]) < 3
    assert abs(s["square"]["side"] - truth["square"][1]) < 4
    assert math.dist(s["circle"]["center"], truth["circle"][0]) < 3
    assert abs(s["circle"]["radius"] - truth["circle"][1]) < 4
    got = sorted((round(o["xyz"][0]), round(o["xyz"][1])) for o in s["objects"])
    assert len(got) == len(truth["cubes"])
    for c in truth["cubes"]:
        assert min(math.dist(c, g) for g in got) < 3
    lay = scene.layout_from(s)
    inside = [o for o in s["objects"] if lay.in_square(*o["xyz"][:2])]
    assert len(inside) == 3


def test_homography_roundtrip():
    cv2 = pytest.importorskip("cv2")
    from armpi import scene
    to_table = scene.homography([(100, 100), (500, 100), (500, 400), (100, 400)],
                                [(200, 100), (200, -100), (50, -100), (50, 100)])
    p = to_table([(300, 250)])[0]
    assert abs(p[0] - 125) < 1e-6 and abs(p[1]) < 1e-6


def test_sim_task_moves_cubes_into_circle():
    pytest.importorskip("mujoco")
    from armpi.task import run, sim_scene
    lay = Layout()
    be, sk, perceive = sim_scene(lay, n=3, seed=1)
    assert all(lay.in_square(*o["xyz"][:2]) for o in perceive())
    res = run(sk, lay, perceive)
    assert res["in_circle"] == 3, perceive()
    assert [p for p, _ in sk.log].count("pick") == 3
