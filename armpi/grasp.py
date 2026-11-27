"""Visual grasp for a small die with the wrist cam: line it up, grab blind, check it left.

The camera rides on the gripper but looks ahead of the fingers: with the fingertips right
above the die, the die is out of frame (cut off at the bottom edge at any height). So the
camera looks from a LOOK POSE, LOOK_BACK_MM back toward the base at PREGRASP_Z (real mm), where
a die that's right under the target point shows up in one pixel box. Taught once (`teach-box`,
die placed by hand under the open fingers) into grasp_ref.yaml:

    box     [u, v]  die center in the look photo when it's right under the grasp point
    angle   deg     die edge angle there when square to the fingers
    size    px      die side length there
    J       2x2     pixel shift per mm of fingertip move in the GRIPPER frame
                    (radial = away from the base, tangential = +90 deg from radial). The image
                    turns with the base, so this holds anywhere on the table.

pick: look, nudge until the die sits in the box, check its angle, then forward LOOK_BACK_MM,
straight down, close, up; back to the look pose: the die must be gone from the table.

No wrist roll: finger yaw = atan2(y, x) of the grasp point, so the hand can't turn to match the
die. Instead the die is turned: `square_up` drops the open fingers around it on the table,
closes them SLOWLY (parallel jaws squeezing a square twist it flat against them: sensorless
part orienting, Goldberg '93), opens, and looks again. Up to MAX_SQUEEZES; past MAX_ANGLE it's
reported as "rotated" (near 45 deg a squeeze can stall on the corners).
"""
import math
from pathlib import Path

import numpy as np
import yaml

REF_PATH = Path(__file__).with_name("grasp_ref.yaml")
FAR_REF_PATH = Path(__file__).with_name("grasp_ref_far.yaml")
PREGRASP_Z = 45.0     # real mm (real_skills maps real z -> commanded)
LOOK_BACK_MM = 25.0   # look from this far back toward the base: right above it the die is out of frame
GRIP_MIN_Z = 2.0      # real fingertip floor for the die (real_skills maps real z -> commanded)
BOX_TOL_PX = 15       # die center within this of `box` -> lined up
MAX_ANGLE = 25.0      # deg off square we still grab; 27-44 deg grabs hit corners and pushed the die
MAX_TILT = 40.0       # deg from vertical the fingers may be at the grab. Past ~115 mm from the base the
                      # wrist must tilt to reach the table (links: 55 + 60 mm); fingers_at shifts the
                      # tips out so the pads, not the tips, land on the die. Untaught beyond ~20 deg.
PAD_H = 0.0           # mm up the fingers where they meet the die. 6 (theory) overshot the far
                      # grab by 5-6 mm; the far reference's grab_forward_mm carries the real offset
MAX_NUDGES = 5        # off-centre the taught gain is a bit off: allow a few more steps
SQUARE_TOL = 10.0     # deg off square that's fine to grab as is
MAX_SQUEEZES = 0       # off: slow squeezes slid the die (and the paper) instead of turning it;
                      # a plain fast close + lift squared it enough (Sean, 2026-11-27)
SQUEEZE_MS = 1500     # slow close, so the die turns instead of popping out
MAX_STEP_MM = 15.0


def _cv2():
    import cv2
    return cv2


# --- vision (pure) ------------------------------------------------------------------------
def find_die(bgr, min_side=20, dark_frac=0.5, min_sat=150):
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
        # far reference (taught ~154 mm out, tilted wrist); only with the default near one
        self.far = load_ref(FAR_REF_PATH) if ref is None else None
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

    def look_pose(self, x, y):
        """Where the fingertip goes to SEE a die at (x, y): LOOK_BACK_MM back toward the base
        (the camera looks ahead of the fingers; right above the die it's out of frame)."""
        rad, _ = gripper_frame(x, y)
        return x - LOOK_BACK_MM * rad[0], y - LOOK_BACK_MM * rad[1]

    def pregrasp(self, x, y):
        self.sk.gripper(self.sk.open_frac)
        self.sk._go((*self.look_pose(x, y), PREGRASP_Z), self.sk.move_ms)

    def die_here(self, tag):
        d = find_die(self.photo(tag))
        self.log(f"{tag}: " + (f"die at pixel ({d['u']:.0f}, {d['v']:.0f}) size {d['size']:.0f} "
                               f"angle {d['angle']:.0f}" if d else "no die"))
        return d

    # one-time teaching -----------------------------------------------------------------------
    def teach_box(self, x, y):
        """Die placed by hand right under the open fingers at (x, y), square to them. Goes to
        the look pose and records where the die shows up + the pixel gain."""
        self.pregrasp(x, y)
        d = self.die_here("teach_box")
        if not d:
            raise RuntimeError("no die in the look photo")
        ref = {"box": [round(d["u"], 1), round(d["v"], 1)], "angle": round(d["angle"], 1),
               "size": round(d["size"], 1), "taught_at": [x, y], "look_back_mm": LOOK_BACK_MM,
               "pregrasp_z": PREGRASP_Z}
        # gain: move the look pose 6 mm radially, then 6 mm tangentially; the die stays put
        rad, tan = gripper_frame(x, y)
        lx, ly = self.look_pose(x, y)
        cols = []
        for name, vec in (("radial", rad), ("tangential", tan)):
            self.sk._go((lx + 6 * vec[0], ly + 6 * vec[1], PREGRASP_Z), self.sk.slow_ms)
            d2 = self.die_here(f"teach_{name}")
            if not d2:
                raise RuntimeError(f"die lost after a 6 mm {name} move")
            cols.append([(d2["u"] - d["u"]) / 6.0, (d2["v"] - d["v"]) / 6.0])
            self.sk._go((lx, ly, PREGRASP_Z), self.sk.slow_ms)
        ref["J"] = [[round(cols[0][0], 2), round(cols[1][0], 2)],
                    [round(cols[0][1], 2), round(cols[1][1], 2)]]
        self.ref = ref
        return ref

    # the grasp -------------------------------------------------------------------------------
    def ref_for(self, x, y):
        """The reference for a die lined up at (x, y): the near one, the far one, or a blend by
        distance from the base in between (the look view, pixel gain and finger offset all
        change as the wrist tilts further out: radial gain 13 px/mm near, ~3 px/mm far)."""
        near, far = self.ref, self.far
        if not far or not near:
            return near
        r0, r1 = math.hypot(*near["taught_at"]), math.hypot(*far["taught_at"])
        t = min(max((math.hypot(x, y) - r0) / (r1 - r0), 0.0), 1.0)
        mix = lambda a, b: a + t * (b - a)
        return {"box": [mix(a, b) for a, b in zip(near["box"], far["box"])],
                "J": [[mix(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(near["J"], far["J"])],
                "angle": near["angle"] + t * angle_off(far["angle"], near["angle"]),
                "grab_forward_mm": mix(near.get("grab_forward_mm", 0.0),
                                       far.get("grab_forward_mm", 0.0)),
                "taught_at": [x, y]}

    def fingers_at(self, x, y):
        """Where the fingertips go down for a die lined up at (x, y): ref grab_forward_mm
        further from the base (first real grab closed on the die's back half), plus, when the
        wrist has to tilt to reach (far out), the shift that puts the finger PADS (PAD_H mm up
        the fingers) over the die instead of the tips: the tilted fingers lean back toward the
        base by PAD_H * tan(tilt)."""
        f = (self.ref_for(x, y) or {}).get("grab_forward_mm", 0.0)
        rad, _ = gripper_frame(x, y)
        px, py = x + f * rad[0], y + f * rad[1]
        tilt = self._tilt_at(px, py) or 0.0
        k = PAD_H * math.tan(math.radians(tilt))
        return px + k * rad[0], py + k * rad[1]

    def _tilt_at(self, fx, fy):
        """Degrees from vertical the fingers are at with the tips at (fx, fy) at grab height;
        None if out of reach. (No arm model, e.g. a test double: 0.)"""
        arm = getattr(self.sk, "arm", None)
        if arm is None or not hasattr(self.sk, "cmd_z"):
            return 0.0
        from .calibration import JOINTS
        from .kinematics import ik_search
        limits = {j: arm.calib.limits(j) for j in JOINTS if j != "gripper"}
        sol = ik_search((fx, fy, self.sk.cmd_z(self.grip_z())), self.sk.pitch, limits)
        return None if sol is None else abs(sol[1] + 90.0)

    def grab_tilt(self, x, y):
        """Tilt from vertical (deg) of the grab for a die lined up at (x, y); None if out of reach."""
        return self._tilt_at(*self.fingers_at(x, y))

    def _grab(self, x, y):
        """From the look pose: forward to above (x, y), straight down, close, back up."""
        x, y = self.fingers_at(x, y)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)
        self.sk._go((x, y, self.grip_z()), self.sk.slow_ms)
        self.sk.gripper(0.0)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)

    def line_up(self, x, y, tag="look"):
        """From the look pose, nudge until the die sits in the box. -> (d, x, y, err) with
        err None when lined up, else the reason."""
        self.sk._go((*self.look_pose(x, y), PREGRASP_Z), self.sk.slow_ms)
        for i in range(MAX_NUDGES + 1):
            r = self.ref_for(x, y)
            d = self.die_here(f"{tag}_{i}")
            if not d:
                return None, x, y, "die not seen from the look pose"
            du, dv = d["u"] - r["box"][0], d["v"] - r["box"][1]
            if math.hypot(du, dv) <= BOX_TOL_PX:
                return d, x, y, None
            if i == MAX_NUDGES:
                return d, x, y, f"couldn't line up (still {math.hypot(du, dv):.0f} px off)"
            dx, dy = correction_mm(r["J"], du, dv, *self.look_pose(x, y))
            x, y = x + dx, y + dy
            self.log(f"  nudge ({dx:+.1f}, {dy:+.1f}) mm -> grasp at ({x:.1f}, {y:.1f})")
            self.sk._go((*self.look_pose(x, y), PREGRASP_Z), self.sk.slow_ms)

    def squeeze(self, x, y):
        """Open fingers down around the die, close slowly (turns it square to the jaws), open,
        back up. The die stays on the table."""
        x, y = self.fingers_at(x, y)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)
        self.sk._go((x, y, self.grip_z()), self.sk.slow_ms)
        self.sk.gripper(0.0, SQUEEZE_MS)
        self.sk.gripper(self.sk.open_frac)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)

    def square_up(self, x, y):
        """Line up; while the die is turned more than SQUARE_TOL, squeeze and look again.
        -> (d, x, y, off_deg, err)."""
        off = None
        for k in range(MAX_SQUEEZES + 1):
            d, x, y, err = self.line_up(x, y, f"look{k}")
            if err:
                return d, x, y, off, err
            off = angle_off(d["angle"], self.ref_for(x, y)["angle"])
            if abs(off) <= SQUARE_TOL or k == MAX_SQUEEZES:
                break
            self.log(f"  die {off:+.0f} deg off square: squeeze {k + 1}")
            self.squeeze(x, y)
        return d, x, y, off, None

    def pick(self, x, y):
        """Line up and square the die from the look pose, grab blind, verify it left the
        table. -> (ok, reason, (x, y) actually grabbed at)."""
        r = self.ref
        if not r:
            raise RuntimeError("no grasp reference: teach first")
        self.sk.gripper(self.sk.open_frac)
        d, x, y, off, err = self.square_up(x, y)
        if err:
            return False, err, (x, y)
        tilt = self.grab_tilt(x, y)
        if tilt is None or tilt > MAX_TILT:
            r_mm = math.hypot(*self.fingers_at(x, y))
            return False, ("too far to grab reliably: " + (f"the fingers would be {tilt:.0f} deg "
                           f"from vertical" if tilt is not None else "out of reach")
                           + f" {r_mm:.0f} mm from the base (max {MAX_TILT:.0f} deg)"), (x, y)
        if abs(off) > MAX_ANGLE:
            return False, f"die is rotated {off:.0f} deg to the fingers; needs straightening", (x, y)
        self._grab(x, y)
        # held check: the die must be gone from where it was (in the fingers, out of frame)
        self.sk._go((*self.look_pose(x, y), PREGRASP_Z), self.sk.slow_ms)
        d = self.die_here("after_grab")
        if d is None:  # any die still in view = on the table (a knocked die lands elsewhere)
            return True, f"die picked up (it was {off:+.0f} deg off square)", (x, y)
        self.sk.gripper(self.sk.open_frac)
        # where it is now, for a retry: pixel offset from the box -> mm, like a nudge
        dx, dy = correction_mm(r["J"], d["u"] - r["box"][0], d["v"] - r["box"][1],
                               *self.look_pose(x, y), max_step=40.0)
        return False, "grab missed (die still on the table)", (x + dx, y + dy)

    def place(self, x, y, drop=2.0):
        self.sk._go((x, y, PREGRASP_Z), self.sk.move_ms)
        self.sk._go((x, y, self.grip_z() + drop), self.sk.slow_ms)
        self.sk.gripper(self.sk.open_frac)
        self.sk._go((x, y, PREGRASP_Z), self.sk.slow_ms)

    def move(self, src, dst):
        """Pick the die near src (x, y), place it at dst, look to confirm.
        -> (ok, log lines, photo bytes of the final look or None)."""
        log, say = [], self.log
        self.log = note = lambda m: (log.append(m), say(m))
        try:
            ok, why, at = self.pick(*src)
            note(f"pick near ({src[0]:.0f}, {src[1]:.0f}): {why}")
            if not ok and "missed" in why:  # one retry from where the die is now
                ok, why, _ = self.pick(*at)
                note(f"retry near ({at[0]:.0f}, {at[1]:.0f}): {why}")
            if ok:
                self._finish_move(dst, note)
        finally:
            self.log = say
        if not ok:
            self.sk.gripper(self.sk.open_frac)
            return False, log, None
        return self._result, log, self._jpg

    def _finish_move(self, dst, note):
        self.place(*dst)
        note(f"placed at ({dst[0]:.0f}, {dst[1]:.0f})")
        self.pregrasp(*dst)
        img = self.photo("check")
        self._jpg = _cv2().imencode(".jpg", img)[1].tobytes()
        d = find_die(img)
        self._result = d is not None
        if d is None:
            note("check: die NOT seen at the destination")
            return
        off = angle_off(d["angle"], self.ref_for(*dst)["angle"])
        note(f"check: die seen at the destination, pixel ({d['u']:.0f}, {d['v']:.0f}), "
             f"{off:+.0f} deg off square")


def main(argv=None):
    """Step-by-step driver for the real arm (each step is its own process; the reference is
    kept in grasp_ref.yaml between steps):
        python -m armpi.grasp pregrasp X Y     open gripper at the look pose for (X, Y), photo
        python -m armpi.grasp teach-box X Y    (die lined up by hand) record box + gain
        python -m armpi.grasp square X Y       line up + squeeze the die square, no pick
        python -m armpi.grasp pick X Y         full visual pick (stays holding at PREGRASP_Z)
        python -m armpi.grasp place X Y
        python -m armpi.grasp move FROM TO     FROM/TO: a region (left, right) or X,Y
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument("step", choices=["pregrasp", "teach-box", "square", "pick", "place", "move"])
    ap.add_argument("x")
    ap.add_argument("y")
    ap.add_argument("--url", default="http://10.0.0.133:8000")
    ap.add_argument("--obj-h", type=float, default=12.0)
    ap.add_argument("--save-dir")
    a = ap.parse_args(argv)
    if a.step == "move":
        from .tools import REGIONS

        def where(s):
            return tuple(REGIONS[s]["center"]) if s in REGIONS else tuple(map(float, s.split(",")))
        src, dst = where(a.x), where(a.y)
    else:
        a.x, a.y = float(a.x), float(a.y)
    from . import vision
    from .task import real_skills
    g = Grasper(real_skills(a.url), lambda: vision._get(a.url, "/snapshot.jpg"),
                obj_h=a.obj_h, save_dir=a.save_dir)
    if a.step == "pregrasp":
        g.pregrasp(a.x, a.y)
        g.die_here("pregrasp")
    elif a.step == "teach-box":
        g.ref = g.teach_box(a.x, a.y)
        save_ref(g.ref)
        print("saved", g.ref)
    elif a.step == "square":
        print(g.square_up(a.x, a.y)[3:])
    elif a.step == "pick":
        print(g.pick(a.x, a.y))
    elif a.step == "move":
        ok, _, _ = g.move(src, dst)  # logs as it goes
        print("DONE" if ok else "FAILED")
    else:
        g.place(a.x, a.y)


if __name__ == "__main__":
    main()
