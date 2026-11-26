"""Find the ball in a wrist-cam photo and turn the pixel into table mm, without a full camera
calibration.

Detector: `find_ball(bgr)` -> {"u", "v", "r", "score"} or None. Tuned on the real photos in
data/llm_runs (crumpled white paper ball on white paper, pen outlines).

Pixel -> mm: the wrist cam is not calibrated, but each `look` uses the SAME pose per region,
so a homography per region (pixel -> table x, y at ball-center height) is all we need. The arm
calibrates it itself with the ball as the target (`self_calibrate`): pick the ball at the
region center, place it at a few known offsets, look and detect each time, fit.

    python -m armpi.ball photo.jpg --out seen.jpg        # check the detector on a photo
"""
import argparse
import math
from pathlib import Path

import numpy as np
import yaml

LOOK_CALIB_PATH = Path(__file__).with_name("look_calib.yaml")
OFFSETS = [(0, 0), (12, 12), (12, -12), (-12, -12), (-12, 12), (0, 0)]  # mm around center


def _cv2():
    import cv2
    return cv2


# --- detector -------------------------------------------------------------------------------
def find_ball(bgr, roi=None, min_r=25, max_r=160, debug=False):
    """Best ball candidate in the image, or None.

    -> {"u", "v", "r", "score", "held", "kind"}. kind "color": a saturated object on the paper
    (e.g. the blue die), checked first; "white": the paper ball. held=True means the ball is in the gripper, right in
    front of the lens (big, out of focus, across the bottom of the frame): NOT on the paper.

    On the paper: a crumpled paper ball is bright like the paper but full of creases and
    shading, so it stands out as a compact blob of high local contrast (texture), while flat
    paper is smooth and pen lines are thin. Only the paper sheet is searched (the floor,
    keyboard and the paper's own edge are textured too); blobs cut off by the image edge are
    ignored. roi = (x0, y0, x1, y1) pixel box to search in (no held check then).
    """
    cv2 = _cv2()
    img = bgr if roi is None else bgr[roi[1]:roi[3], roi[0]:roi[2]]
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # paper = biggest bright, unsaturated region (the purple cast keeps S low), as its convex
    # hull (fills pen lines and the ball), shrunk so the sheet's own edge isn't searched
    light = ((hsv[..., 2] > 110) & (hsv[..., 1] < 110)).astype(np.uint8) * 255
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(light)
    if n < 2:
        return None
    paper = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
    cs, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper = cv2.drawContours(np.zeros_like(paper), [cv2.convexHull(max(cs, key=cv2.contourArea))],
                             -1, 255, -1)
    inner = cv2.erode(paper, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # a colored object (the blue die) on the paper wins outright: it is unambiguous, and its
    # sharp dark edges near the bottom can look like the out-of-focus "held" cue
    color = _color_blob(hsv, inner, W, H)
    if color:
        return _shift(color, roi)
    if roi is None:
        held = _held(gray)
        if held:
            return held
    # local contrast: |gray - blurred| averaged over a small window
    tex = cv2.blur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3)), (9, 9))
    ok = (hsv[..., 2] > 90) & (inner > 0)
    t = np.where(ok, tex, 0)
    thr = max(5.0, float(np.percentile(t[inner > 0], 85))) if inner.any() else 5.0
    mask = (t > thr).astype(np.uint8) * 255
    # pen lines are long + thin: close fills the ball, open with a disk removes the lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if debug:
        globals()["_last_dbg"] = np.maximum(mask // 2, inner // 4)
    # a ball touching a pen line merges with it into one non-compact blob: if nothing compact
    # turns up, open harder (cuts the ~15 px lines off, the ~150 px ball survives)
    for k in (13, 25):
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        best = _best_blob(opened, W, H, min_r, max_r)
        if best:
            break
    if best:
        best["kind"] = "white"
    return _shift(best, roi)


def _shift(b, roi):
    if b and roi is not None:
        b["u"] += roi[0]
        b["v"] += roi[1]
    return b


def _color_blob(hsv, inner, W, H, min_sat=100, min_val=35, dark_frac=0.5, min_r=25, max_r=90):
    """Most compact-and-big solid object inside the paper (the blue die, a colored block) or
    None: pixels that are saturated OR much darker than the paper. Dark blue reads anywhere
    from S ~90 / V ~55 to S ~200 / V ~160 depending on the light, so either cue counts. Pen
    lines are dark too but thin: an opening removes them before a closing fills the die's pips.
    The floor and keyboard lie outside the paper; min_r keeps the small dark creases of the
    paper ball out (a ~16 mm die is r ~40 px from the look pose)."""
    cv2 = _cv2()
    if not inner.any():
        return None
    v = hsv[..., 2]
    paper_v = float(np.median(v[inner > 0]))
    solid = ((hsv[..., 1] >= min_sat) & (v >= min_val)) | (v < dark_frac * paper_v)
    mask = (solid & (inner > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    b = _best_blob(mask, W, H, min_r, max_r, fill=0.4)  # a square fills 0.64 of its circle
    if b:
        b["kind"] = "color"
    return b


def _best_blob(mask, W, H, min_r, max_r, fill=0.45):
    cv2 = _cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        (u, v), r = cv2.minEnclosingCircle(c)
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue  # cut off by the frame
        if not min_r <= r <= max_r or area < fill * math.pi * r * r:
            continue  # compact and ball-sized only
        score = float(area * (0.5 + area / (math.pi * r * r)))
        if best is None or score > best["score"]:
            m = cv2.moments(c)
            best = {"u": m["m10"] / m["m00"], "v": m["m01"] / m["m00"], "r": float(r),
                    "score": score, "held": False}
    return best


def _held(gray, band=(0.79, 1.0), cols=(0.19, 0.81), min_coarse=3.5, min_ratio=1.5):
    """Ball in the gripper: it fills the bottom middle of the frame, so close it's out of
    focus. There, big soft shading (coarse contrast) dominates fine detail; on paper (flat,
    or with a sharp ball / pen lines on it) they're about equal. Real photos: held 1.8-2.7,
    not held 0.9-1.25."""
    cv2 = _cv2()
    H, W = gray.shape
    y0, x0, x1 = int(band[0] * H), int(cols[0] * W), int(cols[1] * W)
    g = gray[y0:, x0:x1]
    coarse = np.abs(cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 12))
    fine = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3))
    c, f = float(coarse.mean()), float(fine.mean())
    if c < min_coarse or c < min_ratio * f:
        return None
    return {"u": (x0 + x1) / 2, "v": float(H - 1), "r": (x1 - x0) / 2,
            "score": round(c / max(f, 1e-6), 2), "held": True}


def annotate(bgr, ball):
    cv2 = _cv2()
    img = bgr.copy()
    if ball:
        cv2.circle(img, (int(ball["u"]), int(ball["v"])), int(ball["r"]), (0, 0, 255), 2)
        cv2.circle(img, (int(ball["u"]), int(ball["v"])), 3, (0, 0, 255), -1)
    return img


# --- pixel -> table, per look pose -----------------------------------------------------------
class LookCalib:
    """{region: 3x3 homography pixel -> table (x, y) mm} for the fixed look pose of each region."""

    def __init__(self, H=None, info=None):
        self.H = {k: np.asarray(v, float) for k, v in (H or {}).items()}
        self.info = info or {}

    @classmethod
    def load(cls, path=LOOK_CALIB_PATH):
        try:
            with open(path) as f:
                d = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return cls()
        return cls(d.get("H"), d.get("info"))

    def save(self, path=LOOK_CALIB_PATH):
        with open(path, "w") as f:
            f.write("# pixel -> table mm homography per look region (python -m armpi.ball / "
                    "move_ball self-calibration)\n")
            yaml.safe_dump({"H": {k: v.tolist() for k, v in self.H.items()}, "info": self.info},
                           f, sort_keys=False)

    def has(self, region):
        return region in self.H

    def to_table(self, region, u, v):
        p = self.H[region] @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def fit(self, region, pixels, table_xy):
        """>= 4 pairs. Uses a least-squares homography; with exactly 4 it's exact."""
        cv2 = _cv2()
        H, _ = cv2.findHomography(np.asarray(pixels, float), np.asarray(table_xy, float), 0)
        if H is None:
            raise ValueError("homography fit failed (points collinear?)")
        self.H[region] = H
        err = [math.dist(self.to_table(region, *p), q) for p, q in zip(pixels, table_xy)]
        self.info[region] = {"points": len(pixels), "max_err_mm": round(max(err), 1)}
        return max(err)


def main(argv=None):
    cv2 = _cv2()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="directory for annotated copies")
    a = ap.parse_args(argv)
    for p in a.images:
        img = cv2.imread(p)
        b = find_ball(img)
        print(p, "->", b and {k: round(v) for k, v in b.items()})
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.out) / Path(p).name), annotate(img, b))


if __name__ == "__main__":
    main()
