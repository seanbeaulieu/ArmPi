"""Visual grasp for a small die with the wrist cam: line it up under the fingers, then grab,
then check it's really held.

The camera rides on the gripper, so from the pregrasp pose (open gripper, fingertip PREGRASP_Z
above the table, pointing down) a die that the fingers can grab cleanly always shows up in the
same pixel box, at the same size, square to the fingers. And a die that's really held sits in
one fixed spot in the photo after the lift. Both are recorded once (`teach`) into
grasp_ref.yaml:

    box     [u, v]  die center in the pregrasp photo when it's right under the fingers
    angle   deg     die edge angle in that photo when square to the fingers
    size    px      die side length there
    held    [u, v]  die center in the photo after closing + lifting to PREGRASP_Z
    held_size px
    J       2x2     pixel offset per mm of fingertip move in the GRIPPER frame
                    (radial = away from the base, tangential = +90 deg from radial). The image
                    turns with the base, so this is the same everywhere on the table.

No wrist roll: finger yaw = atan2(y, x) of the grasp point, so the die's rotation can't be
corrected, only checked (a parallel jaw closing on a die turned < ~25 deg usually twists it
square; more than that is reported as "rotated").
"""
import math
from pathlib import Path

import numpy as np
import yaml

REF_PATH = Path(__file__).with_name("grasp_ref.yaml")
PREGRASP_Z = 45.0
GRIP_MIN_Z = 2.0      # fingertip floor for the die (Skills default is 4: a bit high on the real arm)
BOX_TOL_PX = 12       # die center within this of `box` -> lined up
MAX_ANGLE = 25.0      # deg off square that we still try
MAX_NUDGES = 3
MAX_STEP_MM = 15.0
HELD_TOL_PX = 40


def _cv2():
    import cv2
    return cv2


# --- vision (pure) ------------------------------------------------------------------------
def find_die(bgr, min_side=20, dark_frac=0.5, min_sat=100):
    """Biggest solid (saturated or much darker than the paper), squarish blob not cut off by
    the frame -> {"u", "v", "size", "angle", "fill"} or None. angle in [0, 90): the die's edge
    direction in the image (minAreaRect)."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    v = hsv[..., 2]
    bright = v > np.percentile(v, 50)
    paper_v = float(np.median(v[bright])) if bright.any() else 255.0
    solid = ((hsv[..., 1] >= min_sat) & (v >= 35)) | (v < dark_frac * paper_v)
    mask = solid.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        (u, vv), (rw, rh), ang = cv2.minAreaRect(c)
        side = (rw + rh) / 2
        if side < min_side or min(rw, rh) < 0.6 * max(rw, rh):
            continue  # too small, or a long thing (pen line, finger edge)
        fill = cv2.contourArea(c) / max(rw * rh, 1.0)
        if fill < 0.6:
            continue
        if best is None or side > best["size"]:
            best = {"u": float(u), "v": float(vv), "size": float(side), "angle": float(ang % 90),
                    "fill": float(fill)}
    return best


def angle_off(angle, ref_angle):
    """Smallest rotation (deg, -45..45) between two square orientations."""
    d = (angle - ref_angle) % 90
    return d - 90 if d > 45 else d


def gripper_frame(x, y):
    """Unit radial and tangential vectors at table point (x, y)."""
    r = math.hypot(x, y) or 1.0
    rad = np.array([x / r, y / r])
    return rad, np.array([-rad[1], rad[0]])


def correction_mm(J, du, dv, x, y, max_step=MAX_STEP_MM):
    """Pixel error (die - box) -> table (dx, dy) to move the fingertip so the die lands in the
    box. J maps a (radial, tangential) fingertip move in mm to the die's pixel shift."""
    rt = np.linalg.solve(np.asarray(J, float), -np.array([du, dv], float))
    rad, tan = gripper_frame(x, y)
    d = rt[0] * rad + rt[1] * tan
    n = float(np.linalg.norm(d))
    if n > max_step:
        d *= max_step / n
    return float(d[0]), float(d[1])


# --- reference ------------------------------------------------------------------------------
def load_ref(path=REF_PATH):
    try:
        with open(path) as f:
            return yaml.safe_load(f) or None
    except FileNotFoundError:
        return None


def save_ref(ref, path=REF_PATH):
    with open(path, "w") as f:
        f.write("# visual grasp reference (python armpi/grasp.py teach); see armpi/grasp.py\n")
        yaml.safe_dump(ref, f, sort_keys=False)


# --- arm ------------------------------------------------------------------------------------
class Grasper:
    """skills: armpi.skills.Skills; snapshot: () -> jpeg bytes; save_dir: optional photo dir."""

    def __init__(self, skills, snapshot, ref=None, obj_h=12.0, save_dir=None, log=print):
        self.sk, self.snapshot, self.obj_h = skills, snapshot, obj_h
        self.ref = ref if ref is not None else load_ref()
        self.save = Path(save_dir) if save_dir else None
        if self.save:
            self.save.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.n = 0

    def grip_z(self):
        return max(self.obj_h - self.sk.grip_depth, GRIP_MIN_Z)

    def photo(self, tag, settle=0.6):
        cv2 = _cv2()
        self.sk.wait(settle)
        jpg = self.snapshot()
        if self.save:
            (self.save / f"{self.n:02d}_{tag}.jpg").write_bytes(jpg)
        self.n += 1
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    def pregrasp(self, x, y):
        self.sk.gripper(self.sk.open_frac)
        self.sk._go((x, y, PREGRASP_Z), self.sk.move_ms)

    def die_here(self, tag):
        d = find_die(self.photo(tag))
        self.log(f"{tag}: " + (f"die at pixel ({d['u']:.0f}, {d['v']:.0f}) size {d['size']:.0f} "
                               f"angle {d['angle']:.0f}" if d else "no die"))
        return d

    # one-time teaching -----------------------------------------------------------------------
    def teach_box(self, x, y):
        """Die placed by hand right under the open fingers, square to them; arm at pregrasp."""
        d = self.die_here("teach_box")
        if not d:
            raise RuntimeError("no die in the pregrasp photo")
        ref = {"box": [round(d["u"], 1), round(d["v"], 1)], "angle": round(d["angle"], 1),
               "size": round(d["size"], 1), "taught_at": [x, y]}
        # gain: move the fingertip 6 mm radially, then 6 mm tangentially; the die stays put
        rad, tan = gripper_frame(x, y)
        cols = []
        for name, vec in (("radial", rad), ("tangential", tan)):
            px, py = x + 6 * vec[0], y + 6 * vec[1]
            self.sk._go((px, py, PREGRASP_Z), self.sk.slow_ms)
            d2 = self.die_here(f"teach_{name}")
            if not d2:
                raise RuntimeError(f"die lost after a 6 mm {name} move")
            cols.append([(d2["u"] - d["u"]) / 6.0, (d2["v"] - d["v"]) / 6.0])
            self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)
        ref["J"] = [[round(cols[0][0], 2), round(cols[1][0], 2)],
                    [round(cols[0][1], 2), round(cols[1][1], 2)]]
        self.ref = ref
        return ref

    def teach_held(self, x, y):
        """After teach_box: grab at (x, y), lift to pregrasp height, record where the die sits."""
        self._grab(x, y)
        d = self.die_here("teach_held")
        if not d:
            raise RuntimeError("no die visible after the grab")
        self.ref["held"] = [round(d["u"], 1), round(d["v"], 1)]
        self.ref["held_size"] = round(d["size"], 1)
        return self.ref

    # the grasp -------------------------------------------------------------------------------
    def _grab(self, x, y):
        self.sk._go((x, y, self.grip_z()), self.sk.slow_ms)
        self.sk.gripper(0.0)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)

    def is_held(self, d):
        r = self.ref
        return bool(d and math.dist((d["u"], d["v"]), r["held"]) <= HELD_TOL_PX
                    and abs(d["size"] - r["held_size"]) <= 0.35 * r["held_size"])

    def pick(self, x, y):
        """Line up, grab, verify. -> (ok, reason, (x, y) actually grabbed at)."""
        r = self.ref
        if not r or "held" not in r:
            raise RuntimeError("no grasp reference: teach first")
        self.pregrasp(x, y)
        for i in range(MAX_NUDGES + 1):
            d = self.die_here(f"pregrasp_{i}")
            if not d:
                return False, "die not seen from the pregrasp pose", (x, y)
            du, dv = d["u"] - r["box"][0], d["v"] - r["box"][1]
            if math.hypot(du, dv) <= BOX_TOL_PX:
                break
            if i == MAX_NUDGES:
                return False, f"couldn't line up (still {math.hypot(du, dv):.0f} px off)", (x, y)
            dx, dy = correction_mm(r["J"], du, dv, x, y)
            x, y = x + dx, y + dy
            self.log(f"  nudge ({dx:+.1f}, {dy:+.1f}) mm -> ({x:.1f}, {y:.1f})")
            self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)
        off = angle_off(d["angle"], r["angle"])
        if abs(off) > MAX_ANGLE:
            return False, f"die is rotated {off:.0f} deg to the fingers; needs straightening", (x, y)
        self._grab(x, y)
        d = self.die_here("after_grab")
        if self.is_held(d):
            return True, f"held (angle was {off:+.0f} deg)", (x, y)
        self.sk.gripper(self.sk.open_frac)
        return False, "grab missed (die not between the fingers after the lift)", (x, y)

    def place(self, x, y, drop=2.0):
        self.sk._go((x, y, PREGRASP_Z), self.sk.move_ms)
        self.sk._go((x, y, self.grip_z() + drop), self.sk.slow_ms)
        self.sk.gripper(self.sk.open_frac)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)


def main(argv=None):
    """Step-by-step driver for the real arm (each step is its own process; the reference is
    kept in grasp_ref.yaml between steps):
        python -m armpi.grasp pregrasp X Y     open gripper over (X, Y) at PREGRASP_Z, photo
        python -m armpi.grasp teach-box X Y    (die lined up by hand) record box + gain
        python -m armpi.grasp teach-held X Y   grab, lift, record the held spot
        python -m armpi.grasp pick X Y         full visual pick (stays holding at PREGRASP_Z)
        python -m armpi.grasp place X Y
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument("step", choices=["pregrasp", "teach-box", "teach-held", "pick", "place"])
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("--url", default="http://10.0.0.133:8000")
    ap.add_argument("--obj-h", type=float, default=12.0)
    ap.add_argument("--save-dir")
    a = ap.parse_args(argv)
    from . import vision
    from .task import real_skills
    g = Grasper(real_skills(a.url), lambda: vision._get(a.url, "/snapshot.jpg"),
                obj_h=a.obj_h, save_dir=a.save_dir)
    if a.step == "pregrasp":
        g.pregrasp(a.x, a.y)
        g.die_here("pregrasp")
    elif a.step == "teach-box":
        g.pregrasp(a.x, a.y)
        g.ref = g.teach_box(a.x, a.y)
        save_ref(g.ref)
        print("saved", g.ref)
    elif a.step == "teach-held":
        g.pregrasp(a.x, a.y)
        g.teach_held(a.x, a.y)
        save_ref(g.ref)
        print("saved", g.ref)
    elif a.step == "pick":
        print(g.pick(a.x, a.y))
    else:
        g.place(a.x, a.y)


if __name__ == "__main__":
    main()
