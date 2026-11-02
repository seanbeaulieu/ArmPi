"""Wrist camera geometry + calibration, on synthetic images (no camera, no arm)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from armpi import vision
from armpi.vision import Camera, default_camera, hand_pose, pixel_to_xyz, project

LOOK = {"base": 10.0, "shoulder": 60.0, "elbow": -60.0, "wrist": -80.0}


def test_project_roundtrip():
    for dist in (np.zeros(5), np.array([-0.2, 0.05, 0, 0, 0])):
        c = default_camera()
        c.dist = dist
        pts = np.array([[130, 0, 0], [110, 25, 0], [150, -20, 0]], float)
        uv = project(pts, LOOK, c)
        assert not np.isnan(uv).any()
        back = pixel_to_xyz(uv, LOOK, c, plane_z=0.0)
        assert np.abs(back - pts).max() < 0.05, back


def test_ray_away_from_table_is_nan():
    out = pixel_to_xyz([(320, 50)], {"base": 0, "shoulder": 30, "elbow": 30, "wrist": 30},
                       default_camera())
    assert np.isnan(out).all()


def test_hand_pose_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    from armpi.mujoco_model import load_model
    m = load_model()
    d = mujoco.MjData(m)
    for ang in (LOOK, {"base": -40, "shoulder": 120, "elbow": -80, "wrist": 30}):
        for j, a in ang.items():
            d.qpos[m.joint(j).qposadr[0]] = math.radians(a)
        mujoco.mj_forward(m, d)
        T = hand_pose(ang)
        assert np.allclose(T[:3, :3], d.body("hand").xmat.reshape(3, 3), atol=1e-9)
        assert np.allclose(T[:3, 3], d.body("hand").xpos * 1000, atol=1e-6)


def _render_block(cam, angles, xyz, size_mm=25.0, color=(0, 0, 220)):
    """Top face of a block, drawn where the camera would see it."""
    img = np.full((cam.size[1], cam.size[0], 3), 200, np.uint8)
    x, y, _ = xyz
    h = size_mm / 2
    top = [(x - h, y - h, size_mm), (x + h, y - h, size_mm), (x + h, y + h, size_mm), (x - h, y + h, size_mm)]
    uv = project(top, angles, cam)
    cv2.fillPoly(img, [np.round(uv).astype(np.int32)], color)
    return img


def test_locate_block_end_to_end():
    cam = default_camera()
    img = _render_block(cam, LOOK, (135, 15, 0))
    blocks = vision.locate_blocks(img, LOOK, cam)
    assert blocks and blocks[0]["color"] == "red"
    x, y, z = blocks[0]["xyz"]
    assert abs(x - 135) < 1.5 and abs(y - 15) < 1.5 and z == 12.5


def _render_board(cam, T_cam_board, ppm=8.0, margin_mm=10):
    """Warp the printable board into the camera image for a known board pose."""
    b = vision.BOARD
    board = vision.make_board()
    w, h = (int(n * b["square_mm"] * ppm) for n in b["squares"])
    m = int(margin_mm * ppm)
    src = board.generateImage((w + 2 * m, h + 2 * m), marginSize=m)
    A = np.array([[ppm, 0, m], [0, ppm, m], [0, 0, 1.0]])  # board mm -> board image px
    H = cam.K @ np.column_stack([T_cam_board[:3, 0], T_cam_board[:3, 1], T_cam_board[:3, 3]])
    return cv2.warpPerspective(src, H @ np.linalg.inv(A), cam.size, borderValue=255)


def _pose(rvec, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(np.array(rvec, float))[0], t
    return T


def test_board_pose_from_image():
    cam = default_camera()
    true = _pose([0.25, -0.15, 0.05], [-100, -70, 300])
    img = _render_board(cam, true)
    got = vision.board_pose(img, cam, vision.make_board())
    assert got is not None
    assert np.abs(got[:3, 3] - true[:3, 3]).max() < 3.0
    dR = cv2.Rodrigues(got[:3, :3] @ true[:3, :3].T)[0]
    assert np.degrees(np.linalg.norm(dR)) < 0.5


def test_intrinsics_calibration_recovers_K():
    cam = default_camera()
    cam.K = np.array([[560.0, 0, 330], [0, 555.0, 235], [0, 0, 1]])
    rng = np.random.default_rng(1)
    imgs = []
    for _ in range(12):
        rv = rng.uniform(-0.45, 0.45, 3)
        imgs.append(_render_board(cam, _pose(rv, [-100 + rng.uniform(-20, 20), -70 + rng.uniform(-20, 20),
                                                   rng.uniform(280, 380)])))
    out, rms = vision.calibrate_intrinsics(imgs, vision.make_board(), default_camera())
    assert rms < 1.0
    assert abs(out.K[0, 0] - 560) < 15 and abs(out.K[1, 2] - 235) < 10
    assert out.guess["intrinsics"] is False


def test_handeye_recovers_mount():
    cam = default_camera()
    true = cam.T_hand_cam @ _pose([0.05, -0.08, 0.03], [3, -4, 6])
    poses = vision.handeye_poses((130, 0, 0), cam)
    assert len(poses) >= 6, poses
    T_base_board = _pose([math.pi, 0, 0.1], [100, -40, 0])  # flat, face up
    samples = [(a, np.linalg.inv(hand_pose(a) @ true) @ T_base_board) for a in poses]
    out, rep = vision.calibrate_handeye(samples, cam)
    assert np.abs(out.T_hand_cam[:3, 3] - true[:3, 3]).max() < 0.5
    assert np.allclose(out.T_hand_cam[:3, :3], true[:3, :3], atol=1e-3)
    assert rep["spread_mm"] < 0.5


def test_camera_yaml_roundtrip(tmp_path):
    c = default_camera()
    c.save(tmp_path / "cam.yaml")
    d = Camera.load(tmp_path / "cam.yaml")
    assert np.allclose(d.K, c.K) and np.allclose(d.T_hand_cam, c.T_hand_cam) and d.guess == c.guess


def test_handeye_with_noise_is_close():
    cam = default_camera()
    true = cam.T_hand_cam @ _pose([0.05, -0.08, 0.03], [3, -4, 6])
    rng = np.random.default_rng(0)
    T_base_board = _pose([math.pi, 0, 0.1], [100, -40, 0])
    samples = []
    for a in vision.handeye_poses((130, 0, 0), cam):
        noise = _pose(rng.normal(0, math.radians(0.15), 3), rng.normal(0, 0.5, 3))
        samples.append((a, noise @ np.linalg.inv(hand_pose(a) @ true) @ T_base_board))
    out, rep = vision.calibrate_handeye(samples, cam)
    # what matters: a table point seen by the camera lands in the right place
    p = np.array([[140, 10, 0]], float)
    look = vision.handeye_poses((130, 0, 0), cam)[0]
    t_true = Camera(cam.K, cam.dist, cam.size, true)
    uv = project(p, look, t_true)
    assert np.abs(pixel_to_xyz(uv, look, out) - p).max() < 3.0


def test_server_vision_endpoint():
    import json
    import urllib.request
    sys.path.insert(0, os.path.dirname(__file__))
    from test_server import start_server
    cam = default_camera()
    jpg = cv2.imencode(".jpg", _render_block(cam, LOOK, (135, 15, 0)))[1].tobytes()

    class FakeCamera:
        error, stamp = None, None

        def frame(self, timeout=0, after=None):
            return jpg

    _, app, url, _ = start_server()
    app.camera = FakeCamera()
    req = urllib.request.Request(url + "/api/move", json.dumps({"angles": dict(LOOK, gripper=0)}).encode(),
                                 {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req).read())
    assert "t" in out  # server-clock command time, for lining up camera frames
    got = json.loads(urllib.request.urlopen(url + "/api/vision/blocks").read())
    b = got["blocks"][0]
    assert b["color"] == "red" and abs(b["xyz"][0] - 135) < 3 and abs(b["xyz"][1] - 15) < 3
    assert "handeye" in got["camera_guess"]
