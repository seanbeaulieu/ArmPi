"""Wrist camera: calibration (intrinsics + hand-eye) and pixel -> table xyz.

The camera rides on the hand, so where it is depends on the joint angles. With the camera
calibrated, a block seen at pixel (u, v) is a ray from the camera; that ray hits the table
plane at one xyz in the arm's base frame (mm, same frame as kinematics.py), which is a reach
target for Arm.move_to. That replaces Hiwonder's color-sorting demos.

Frames (all 4x4, mm):
    base  : kinematics.py frame (origin = base center on the table, x forward, z up)
    hand  : origin at the wrist pivot, x along the gripper, y = base-rotated +y, z = x cross y
            (the same frame as the MuJoCo "hand" body)
    cam   : OpenCV camera frame (z = optical axis, x = image right, y = image down)
camera.yaml holds K, distortion and T_hand_cam. Until you run the calibrations it holds GUESSES.

The real arm reports no angles, so the camera pose uses the COMMANDED angles: any servo sag
shows up as a target error. Look at the board from the pose you'll pick from and check
`locate` against a ruler before trusting it.

    python -m armpi.vision board board.png                # print it, measure a square
    python -m armpi.vision intrinsics --url http://armpi.local:8000 --square-mm 30.0
    python -m armpi.vision handeye   --url http://armpi.local:8000 --square-mm 30.0
    python -m armpi.vision locate    --url http://armpi.local:8000
    python -m armpi.vision reach     --url http://armpi.local:8000 --color red
"""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from .kinematics import fk

CAMERA_PATH = Path(__file__).with_name("camera.yaml")

# ChArUco board: fits A4 / letter. Print at 100%, then MEASURE a square and pass --square-mm.
BOARD = {"squares": (7, 5), "square_mm": 30.0, "marker_mm": 22.0, "dict": "DICT_4X4_50"}

# HSV ranges (OpenCV: H 0..179). Red wraps, so it gets two ranges.
COLORS = {
    "red": [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green": [((40, 80, 50), (85, 255, 255))],
    "blue": [((95, 120, 50), (130, 255, 255))],
}
BLOCK_MM = 25.0  # Hiwonder's sorting cubes are ~25mm; override with --block-mm


# --- camera model ---------------------------------------------------------------------------
def default_camera():
    """Uncalibrated guess: 640x480, ~65 deg horizontal FOV, no distortion, camera on top of
    the hand looking along the gripper (upright when the gripper is horizontal)."""
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # columns: cam x, y, z in hand
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (20.0, 0.0, 25.0)
    return Camera(np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]), np.zeros(5),
                  (640, 480), T, guess={"intrinsics": True, "handeye": True})


class Camera:
    def __init__(self, K, dist, size, T_hand_cam, guess=None, info=None):
        self.K = np.asarray(K, float)
        self.dist = np.asarray(dist, float).ravel()
        self.size = tuple(int(x) for x in size)
        self.T_hand_cam = np.asarray(T_hand_cam, float)
        self.guess = dict(guess or {})
        self.info = dict(info or {})  # rms errors, dates, ...

    @classmethod
    def load(cls, path=CAMERA_PATH):
        try:
            with open(path) as f:
                d = yaml.safe_load(f)
        except FileNotFoundError:
            return default_camera()
        return cls(d["K"], d["dist"], d["size"], d["T_hand_cam"], d.get("guess"), d.get("info"))

    def save(self, path=CAMERA_PATH):
        d = {"size": list(self.size), "K": self.K.round(4).tolist(),
             "dist": self.dist.round(6).tolist(), "T_hand_cam": self.T_hand_cam.round(4).tolist(),
             "guess": self.guess, "info": self.info}
        with open(path, "w") as f:
            f.write("# Wrist camera calibration, written by python -m armpi.vision. Units: mm, px.\n"
                    "# T_hand_cam: camera pose in the hand frame (see armpi/vision.py).\n")
            yaml.safe_dump(d, f, sort_keys=False)

    def scaled(self, size):
        """Same camera at another resolution (K scales; distortion doesn't)."""
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return Camera(K, self.dist, size, self.T_hand_cam, self.guess, self.info)


# --- geometry -------------------------------------------------------------------------------
def hand_pose(angles):
    """4x4 base<-hand for these joint angles (deg), mm."""
    pts = fk(angles)
    yaw, pitch = math.radians(angles["base"]), math.radians(pts["pitch_deg"])
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])  # rotation about -y by pitch
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rz @ Ry, pts["wrist"]
    return T


def camera_pose(angles, cam):
    """4x4 base<-cam."""
    return hand_pose(angles) @ cam.T_hand_cam


def undistort(cam, uv):
    """Pixels (N,2) -> normalized image coords (N,2)."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    if not np.any(cam.dist):
        return (uv - cam.K[:2, 2]) / np.diag(cam.K)[:2]
    import cv2
    return cv2.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)


def pixel_to_xyz(uv, angles, cam, plane_z=0.0):
    """Pixel(s) -> point(s) on the horizontal plane z = plane_z (mm, base frame).
    Returns (N,3) array; rows are NaN where the ray points away from the plane."""
    n = undistort(cam, uv)
    T = camera_pose(angles, cam)
    rays = (T[:3, :3] @ np.column_stack([n, np.ones(len(n))]).T).T
    o = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (plane_z - o[2]) / rays[:, 2]
    s[~(s > 0)] = np.nan
    return o + s[:, None] * rays


def project(xyz, angles, cam):
    """Base-frame point(s) (N,3) -> pixels (N,2). NaN for points behind the camera."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    T = np.linalg.inv(camera_pose(angles, cam))
    pc = (T[:3, :3] @ xyz.T).T + T[:3, 3]
    if np.any(cam.dist):
        import cv2
        uv, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), cam.K, cam.dist)
        uv = uv.reshape(-1, 2)
    else:
        uv = (cam.K @ (pc / pc[:, 2:3]).T).T[:, :2]
    uv[pc[:, 2] <= 0] = np.nan
    return uv


# --- blocks ---------------------------------------------------------------------------------
def detect_blocks(bgr, colors=None, min_area=150):
    """Colored blobs -> [{"color", "u", "v", "area", "box"}], biggest first."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = []
    for name, ranges in (colors or COLORS).items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            mo = cv2.moments(c)
            out.append({"color": name, "u": mo["m10"] / mo["m00"], "v": mo["m01"] / mo["m00"],
                        "area": float(area), "box": [int(x) for x in cv2.boundingRect(c)]})
    return sorted(out, key=lambda b: -b["area"])


def locate_blocks(bgr, angles, cam, block_mm=BLOCK_MM, colors=None):
    """Blocks with base-frame xyz. The blob centroid is (mostly) the top face when looking
    down, so the ray is cut at z = block_mm; "xyz" is the block center (z = block_mm / 2)."""
    blocks = detect_blocks(bgr, colors)
    h, w = bgr.shape[:2]
    c = cam if (w, h) == cam.size else cam.scaled((w, h))
    for b in blocks:
        p = pixel_to_xyz([(b["u"], b["v"])], angles, c, plane_z=block_mm)[0]
        b["xyz"] = None if np.isnan(p).any() else [float(p[0]), float(p[1]), block_mm / 2]
    return blocks


# --- ChArUco board --------------------------------------------------------------------------
def make_board(square_mm=None, board=BOARD):
    import cv2
    sq = square_mm or board["square_mm"]
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board["dict"]))
    return cv2.aruco.CharucoBoard(board["squares"], sq, sq * board["marker_mm"] / board["square_mm"], d)


def board_image(px_per_mm=300 / 25.4, board=BOARD, margin_mm=10):
    """Printable board (300 dpi by default)."""
    b = make_board(board=board)
    w, h = (int(n * board["square_mm"] * px_per_mm) for n in board["squares"])
    m = int(margin_mm * px_per_mm)
    return b.generateImage((w + 2 * m, h + 2 * m), marginSize=m)


def detect_charuco(gray, board):
    """-> (corner_px (N,2), corner_ids (N,)) or (None, None)."""
    import cv2
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None, None
    return corners.reshape(-1, 2), ids.ravel()


def board_pose(bgr, cam, board):
    """Board pose in the camera frame: 4x4 cam<-board (mm), or None. Needs >= 6 corners."""
    import cv2
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    px, ids = detect_charuco(gray, board)
    if ids is None or len(ids) < 6:
        return None
    obj = board.getChessboardCorners()[ids]
    h, w = gray.shape[:2]
    c = cam if (w, h) == cam.size else cam.scaled((w, h))
    ok, rvec, tvec = cv2.solvePnP(obj, px, c.K, c.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return T


def calibrate_intrinsics(images, board, cam=None):
    """Images of the board from many angles -> Camera with new K/dist (keeps T_hand_cam).
    Returns (camera, rms_px)."""
    import cv2
    objs, imgs, size = [], [], None
    for im in images:
        gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        px, ids = detect_charuco(gray, board)
        if ids is not None and len(ids) >= 8:
            objs.append(board.getChessboardCorners()[ids].astype(np.float32))
            imgs.append(px.astype(np.float32))
    if len(objs) < 5:
        raise ValueError(f"board found in only {len(objs)} images; need >= 5 (aim for 15+)")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    base = cam or default_camera()
    out = Camera(K, dist, size, base.T_hand_cam, dict(base.guess, intrinsics=False),
                 dict(base.info, intrinsics_rms_px=round(float(rms), 3), intrinsics_views=len(objs),
                      intrinsics_date=time.strftime("%Y-%m-%d")))
    return out, float(rms)


def _T(x):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]
    return T


def _x(T):
    from scipy.spatial.transform import Rotation
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def calibrate_handeye(samples, cam, board_extent_mm=(210, 150)):
    """samples: [(angles, T_cam_board)] with the board fixed on the table.
    Finds T_hand_cam (and the board's pose in the base frame) so that every view puts the
    board in the same place: least squares on the board's corners, starting from cam's
    current T_hand_cam. Returns (camera, report); report["spread_mm"] is the leftover
    scatter of the board across views (big = joint calibration or intrinsics off)."""
    from scipy.optimize import least_squares
    if len(samples) < 4:
        raise ValueError(f"need >= 4 poses with the board visible, got {len(samples)}")
    w, h = board_extent_mm
    q = np.array([[0, 0, 0, 1], [w, 0, 0, 1], [0, h, 0, 1], [w, h, 0, 1], [w / 2, h / 2, 0, 1]], float).T
    H = [hand_pose(a) for a, _ in samples]
    C = [Tcb for _, Tcb in samples]
    X0 = cam.T_hand_cam
    B0 = H[0] @ X0 @ C[0]

    def resid(x):
        X, B = _T(x[:6]), _T(x[6:])
        Bq = B @ q
        return np.concatenate([((Hi @ X @ Ci @ q) - Bq)[:3].ravel() for Hi, Ci in zip(H, C)])

    res = least_squares(resid, np.concatenate([_x(X0), _x(B0)]), x_scale="jac")
    X, B = _T(res.x[:6]), _T(res.x[6:])
    err = res.fun.reshape(len(samples), 3, -1)
    spread = float(np.sqrt((err ** 2).sum(1).mean()))
    out = Camera(cam.K, cam.dist, cam.size, X, dict(cam.guess, handeye=False),
                 dict(cam.info, handeye_board_spread_mm=round(spread, 2), handeye_views=len(samples),
                      handeye_date=time.strftime("%Y-%m-%d")))
    return out, {"board_origin_mm": B[:3, 3].round(1).tolist(), "spread_mm": spread,
                 "board_z_axis": B[:3, 2].round(3).tolist()}


def handeye_poses(target=(130.0, 0.0, 0.0), cam=None, limits=None, n=12):
    """Joint poses that keep `target` (board center, base frame) near the image center,
    spread over base yaw and pitch so the hand-eye solve sees two rotation axes."""
    cam = cam or Camera.load()
    lim = limits or {"base": (-90, 90), "shoulder": (0, 155), "elbow": (-90, 90), "wrist": (-90, 90)}
    w, h = cam.size
    cands = []
    for b in (-20, -10, 0, 10, 20):
        b = b + math.degrees(math.atan2(target[1], target[0]))
        for sh in (40, 50, 60, 70, 80):
            for el in (-80, -65, -50, -35):
                for wr in (-90, -80, -70, -60):
                    a = {"base": b, "shoulder": sh, "elbow": el, "wrist": wr}
                    if not all(lim[j][0] <= a[j] <= lim[j][1] for j in a):
                        continue
                    uv = project([target], a, cam)[0]
                    if np.isnan(uv).any():
                        continue
                    off = math.hypot((uv[0] - w / 2) / w, (uv[1] - h / 2) / h)
                    if off < 0.3:
                        cands.append((off, a))
    # greedy spread: keep poses that differ from the chosen ones
    chosen = []
    for _, a in sorted(cands, key=lambda c: c[0]):
        if all(sum(abs(a[j] - c[j]) for j in a) > 25 for c in chosen):
            chosen.append(a)
        if len(chosen) >= n:
            break
    return chosen


# --- talking to the Pi ----------------------------------------------------------------------
def _get(url, path, timeout=10):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return r.read()


def _post(url, path, body, timeout=10):
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def snapshot(url):
    import cv2
    return cv2.imdecode(np.frombuffer(_get(url, "/snapshot.jpg"), np.uint8), cv2.IMREAD_COLOR)


def _angles(url):
    a = json.loads(_get(url, "/api/state"))["angles"]
    if not all(j in a for j in ("base", "shoulder", "elbow", "wrist")):
        raise SystemExit("arm pose unknown (no move since the server started): move it first")
    return a


def main(argv=None):
    p = argparse.ArgumentParser(prog="armpi.vision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("board", help="write a printable ChArUco board PNG (300 dpi)")
    s.add_argument("out", nargs="?", default="charuco_board.png")
    for name in ("intrinsics", "handeye", "locate", "reach"):
        s = sub.add_parser(name)
        s.add_argument("--url", default="http://armpi.local:8000")
        s.add_argument("--camera-file", default=str(CAMERA_PATH))
        s.add_argument("--square-mm", type=float, default=BOARD["square_mm"],
                       help="printed square size, measured with calipers")
        if name == "intrinsics":
            s.add_argument("--n", type=int, default=20, help="views to collect")
            s.add_argument("--dir", help="calibrate from saved .jpg/.png files instead")
        if name == "handeye":
            s.add_argument("--target", type=float, nargs=2, default=(130, 0),
                           help="board center x y (mm) as placed on the table")
            s.add_argument("--yes", action="store_true", help="don't ask before moving the arm")
        if name in ("locate", "reach"):
            s.add_argument("--block-mm", type=float, default=BLOCK_MM)
        if name == "reach":
            s.add_argument("--color", default=None)
            s.add_argument("--hover-mm", type=float, default=40, help="stop this far above it")
    a = p.parse_args(argv)
    import cv2

    if a.cmd == "board":
        cv2.imwrite(a.out, board_image())
        print(f"wrote {a.out}: print at 100% (no 'fit to page'), measure one square, "
              f"pass it as --square-mm (nominal {BOARD['square_mm']}mm)")
        return
    cam = Camera.load(a.camera_file)
    board = make_board(a.square_mm)

    if a.cmd == "intrinsics":
        if a.dir:
            files = sorted(f for f in Path(a.dir).iterdir() if f.suffix.lower() in (".jpg", ".png"))
            images = [cv2.imread(str(f)) for f in files]
        else:
            images = []
            print("hold the board in front of the camera at different angles/distances/corners;"
                  " press Enter for each view (q + Enter to finish early)")
            while len(images) < a.n:
                if input(f"view {len(images) + 1}/{a.n}: ").strip().lower() == "q":
                    break
                im = snapshot(a.url)
                px, ids = detect_charuco(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), board)
                print(f"  {0 if ids is None else len(ids)} corners", "" if ids is not None and len(ids) >= 8
                      else "(too few, not kept)")
                if ids is not None and len(ids) >= 8:
                    images.append(im)
        out, rms = calibrate_intrinsics(images, board, cam)
        out.save(a.camera_file)
        print(f"rms reprojection {rms:.3f}px ({'good' if rms < 0.6 else 'ok' if rms < 1.2 else 'poor: retake'})"
              f" -> {a.camera_file}")
        return

    if a.cmd == "handeye":
        if cam.guess.get("intrinsics"):
            print("warning: intrinsics are still the guess; run `intrinsics` first")
        poses = handeye_poses((a.target[0], a.target[1], 0.0), cam)
        print(f"{len(poses)} poses around the board at {a.target}. The arm WILL move.")
        if not a.yes and input("go? [y/N] ").strip().lower() != "y":
            return
        _post(a.url, "/api/backend", {"name": "i2c"})
        samples = []
        for i, ang in enumerate(poses):
            r = _post(a.url, "/api/move", {"angles": ang, "duration_ms": 1500})
            time.sleep(r["duration_ms"] / 1000 + 1.0)  # settle
            got = r["angles"]
            T = board_pose(snapshot(a.url), cam, board)
            print(f"  pose {i + 1}: {'board seen' if T is not None else 'NO board'}")
            if T is not None:
                samples.append((got, T))
        out, rep = calibrate_handeye(samples, cam)
        out.save(a.camera_file)
        print(f"board origin in base frame {rep['board_origin_mm']} mm, scatter "
              f"{rep['spread_mm']:.1f}mm across views (want < 5; big = joint calibration off), "
              f"board normal {rep['board_z_axis']} (want ~[0,0,+-1]) -> {a.camera_file}")
        return

    angles = _angles(a.url)
    blocks = locate_blocks(snapshot(a.url), angles, cam, a.block_mm)
    if cam.guess:
        g = [k for k, v in cam.guess.items() if v]
        if g:
            print(f"warning: camera {', '.join(g)} still guessed; xyz is rough")
    for b in blocks:
        xyz = "off-table" if b["xyz"] is None else "(%.0f, %.0f, %.0f) mm" % tuple(b["xyz"])
        print(f"{b['color']:6s} px ({b['u']:.0f}, {b['v']:.0f}) area {b['area']:.0f} -> {xyz}")
    if a.cmd == "reach":
        pick = [b for b in blocks if b["xyz"] and (a.color is None or b["color"] == a.color)]
        if not pick:
            raise SystemExit("no matching block seen")
        x, y, z = pick[0]["xyz"]
        target = [x, y, z + a.block_mm / 2 + a.hover_mm]
        r = _post(a.url, "/api/move_to", {"xyz": target, "pitch": -90, "duration_ms": 1500})
        print(f"hovering {a.hover_mm:.0f}mm over the {pick[0]['color']} block at {target}, pitch {r['pitch']:.0f}")


if __name__ == "__main__":
    main()
