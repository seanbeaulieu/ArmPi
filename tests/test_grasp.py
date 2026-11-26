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
