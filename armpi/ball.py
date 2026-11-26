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
OFFSETS = [(0, 0), (15, 15), (15, -15), (-15, 15), (-15, -15), (0, 0)]  # mm around center


def _cv2():
    import cv2
    return cv2


# --- detector -------------------------------------------------------------------------------
def find_ball(bgr, roi=None, min_r=8, max_r=160):
    """Best ball candidate in the image, or None.

    The ball is a crumpled paper ball: bright like the paper, but full of creases and shading,
    so it stands out as a compact blob of high local contrast (texture), while flat paper is
    smooth and pen lines are thin. roi = (x0, y0, x1, y1) pixel box to search in.
    """
    cv2 = _cv2()
    img = bgr if roi is None else bgr[roi[1]:roi[3], roi[0]:roi[2]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # local contrast: |gray - blurred| averaged over a small window
    tex = cv2.blur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3)), (9, 9))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # drop dark pixels (pen, table, shadows) and saturated ones (colored stuff, floor)
    ok = (hsv[..., 2] > 90) & (hsv[..., 1] < 90)
    t = np.where(ok, tex, 0)
    thr = max(6.0, float(np.percentile(t, 90)))
    mask = (t > thr).astype(np.uint8) * 255
    # pen lines are long + thin: open with a disk removes them, close fills the ball
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        (u, v), r = cv2.minEnclosingCircle(c)
        if not min_r <= r <= max_r or area < 0.35 * math.pi * r * r:  # compact only
            continue
        score = float(area * (0.5 + area / (math.pi * r * r)))
        if best is None or score > best["score"]:
            best = {"u": float(u), "v": float(v), "r": float(r), "score": score}
    if best and roi is not None:
        best["u"] += roi[0]
        best["v"] += roi[1]
    return best


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
