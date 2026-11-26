"""The arm as a small set of LLM tools: look (wrist cam image), pick, place, move, gripper, home.

One registry, two front ends:
    armpi.mcp_server   stdio MCP server, so Claude Code / Claude Desktop can drive the arm
    (later) an API loop that calls the same tools one action at a time

Each tool returns a list of MCP content blocks: {"type": "text", "text": ...} and, for
`look`, {"type": "image", "data": <base64 jpeg>, "mimeType": "image/jpeg"}.

Coordinates are the arm's base frame in mm: origin = center of the base on the table,
+x straight out in front of the arm, +y to the arm's LEFT, z up.
"""
import base64
import json
import math

import numpy as np

# Named spots on the paper (base frame mm). Defaults = Sean's two squares (60 mm, ~115 mm out).
REGIONS = {
    "left": {"center": (95.0, 65.0), "size": 60.0},
    "right": {"center": (95.0, -65.0), "size": 60.0},
}
# Look pose: the wrist cam points out along the gripper at a slant, so from right above a
# region it sees past it. Pull the fingertip back toward the base and aim at the region:
# on the real arm (66, +-45, 80) framed a whole 60 mm square at (95, +-65).
LOOK_Z = 80.0
LOOK_PULLBACK = 0.7   # fingertip at this fraction of the region center's (x, y)
LOOK_SETTLE_S = 0.6   # extra wait before the photo (the first photo after a move was blurred)


def _text(s):
    return {"type": "text", "text": s}


def _img(jpg):
    return {"type": "image", "data": base64.b64encode(jpg).decode(), "mimeType": "image/jpeg"}


def _near(a, b):
    """Two detections at (about) the same pixel?"""
    if not (a.get("pixel") and b.get("pixel")):
        return False
    return math.dist(a["pixel"], b["pixel"]) < max(25, a.get("r_px", 0))


def _xy(args):
    return float(args["x"]), float(args["y"])


class ArmTools:
    """backend: "sim" (MuJoCo, ground truth instead of a camera) or a Pi server URL."""

    def __init__(self, url=None, regions=None, obj_h=15.0, skills=None, snapshot=None,
                 truth=None):
        self.regions = regions or REGIONS
        self.obj_h = obj_h
        self.truth = truth  # sim only: () -> [{"name", "xyz"}]
        self.url = url
        if skills is not None:
            self.skills, self.snapshot = skills, snapshot
        elif url:
            from .task import real_skills
            from . import vision
            self.skills = real_skills(url)
            self.snapshot = lambda: vision._get(url, "/snapshot.jpg")
        else:
            raise ValueError("give url= or skills=")

    @classmethod
    def sim(cls, n=1, seed=0, **kw):
        """MuJoCo arm with n cubes in the left square. No camera images (the cloud and most
        WSL setups can't render); `look` returns the true object positions instead."""
        from .task import Layout, sim_scene
        c = REGIONS["left"]["center"]
        lay = Layout(square_c=c, square_side=REGIONS["left"]["size"],
                     circle_c=REGIONS["right"]["center"], circle_r=30)
        be, sk, perceive = sim_scene(lay, n, obj_mm=kw.get("obj_h", 15.0), seed=seed)
        t = cls(skills=sk, snapshot=None, truth=perceive, **kw)
        t.backend = be
        return t

    # --- registry ---
    def specs(self):
        regions = ", ".join(f"{k}: center ({v['center'][0]:g}, {v['center'][1]:g}) mm, "
                            f"{v['size']:g} mm square" for k, v in self.regions.items())
        xy = {"x": {"type": "number", "description": "mm forward of the base"},
              "y": {"type": "number", "description": "mm to the arm's left (negative = right)"}}
        return [
            {"name": "look", "description":
                "Move the wrist camera above a region and return a photo. The camera rides on "
                "the gripper, so the view is from the gripper looking down at a slant. "
                f"Regions: {regions}. 'here' = photo from the current pose without moving.",
             "inputSchema": {"type": "object", "properties": {
                 "where": {"type": "string", "enum": [*self.regions, "here"]}},
                 "required": ["where"]}},
            {"name": "pick", "description":
                "Open, hover at 60 mm, descend, close on the object centered at (x, y), lift. "
                "The arm is open loop: it can't feel whether it grabbed anything; look to check.",
             "inputSchema": {"type": "object", "properties": xy, "required": ["x", "y"]}},
            {"name": "place", "description": "Carry the held object to (x, y), lower, open, lift.",
             "inputSchema": {"type": "object", "properties": xy, "required": ["x", "y"]}},
            {"name": "move_to", "description":
                "Put the fingertip at (x, y, z) mm, gripper pointing down as far as reach allows. "
                "Reach at table height is ~85-170 mm from the base axis, yaw within +-90 deg.",
             "inputSchema": {"type": "object", "properties": {
                 **xy, "z": {"type": "number", "description": "mm above the table"}},
                 "required": ["x", "y", "z"]}},
            {"name": "gripper", "description": "Set the gripper: 0 = closed, 1 = fully open.",
             "inputSchema": {"type": "object", "properties": {
                 "open": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["open"]}},
            {"name": "home", "description": "Fold the arm to its rest pose (out of the camera's way).",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "move_ball", "description":
                "Do the whole move in one call: look at `from`, find the ball with the vision "
                "detector, pick it, look again to check it left the paper (retries once), place "
                "it at the `to` region's center (or at x, y), look to verify, go home. Returns "
                "a step log and the final photo. Use this first; fall back to look/pick/place "
                "when it reports a problem.",
             "inputSchema": {"type": "object", "properties": {
                 "from": {"type": "string", "enum": list(self.regions)},
                 "to": {"type": "string", "enum": list(self.regions)},
                 "x": {"type": "number", "description": "optional exact drop x, mm"},
                 "y": {"type": "number", "description": "optional exact drop y, mm"}},
                 "required": ["from", "to"]}},
            {"name": "calibrate_look", "description":
                "Self-calibrate pixel -> mm for one region's look pose. The ball must be at the "
                "region's center. The arm moves it to 5 known spots, photographs and detects it "
                "each time, fits a mapping, and leaves the ball at the center. ~1.5 min.",
             "inputSchema": {"type": "object", "properties": {
                 "region": {"type": "string", "enum": list(self.regions)}}, "required": ["region"]}},
            {"name": "status", "description":
                "Battery voltage, last commanded fingertip position, and the regions.",
             "inputSchema": {"type": "object", "properties": {}}},
        ]

    def call(self, name, args=None):
        args = args or {}
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise KeyError(f"unknown tool {name!r}")
        return fn(args)

    # --- tools ---
    def _fingertip(self):
        a = self.skills.arm.angles
        if not all(j in a for j in ("base", "shoulder", "elbow", "wrist")):
            return None
        return [round(v) for v in self.skills.arm.fk()["fingertip"]]

    def t_look(self, args):
        where = args.get("where", "here")
        if where != "here":
            cx, cy = self.regions[where]["center"]
            self.skills.move_to(cx * LOOK_PULLBACK, cy * LOOK_PULLBACK, LOOK_Z)
            self.skills.wait(LOOK_SETTLE_S)
        out = []
        if self.snapshot:
            jpg = self.snapshot()
            out.append({"type": "image", "data": base64.b64encode(jpg).decode(),
                        "mimeType": "image/jpeg"})
            out.append(_text(f"wrist cam photo, fingertip at {self._fingertip()} mm"
                             + self._detections(jpg)))
        else:
            objs = [{"name": o["name"], "xyz_mm": [round(v) for v in o["xyz"]]}
                    for o in self.truth()]
            out.append(_text("sim (no camera): objects at " + json.dumps(objs)
                             + f"; fingertip at {self._fingertip()} mm"))
        return out

    def _detections(self, jpg):
        """Colored objects in the photo with ROUGH table xyz from the (uncalibrated until
        camera.yaml is fitted) camera model. Best effort: never fails the look."""
        try:
            import cv2
            import numpy as np
            from . import scene, vision
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            objs = scene.find_objects(img)
            if not objs:
                return "; no colored objects detected (white/gray things aren't detected)"
            cam = vision.Camera.load()
            h, w = img.shape[:2]
            cam = cam if (w, h) == cam.size else cam.scaled((w, h))
            uv = [(o["u"], o["v"]) for o in objs]
            xyz = vision.pixel_to_xyz(uv, self.skills.arm.angles, cam, plane_z=self.obj_h)
            guess = " (camera NOT calibrated: xyz is a rough guess)" if cam.guess else ""
            items = [f"pixel ({o['u']:.0f}, {o['v']:.0f}) -> ~({p[0]:.0f}, {p[1]:.0f}) mm"
                     for o, p in zip(objs, xyz) if not np.isnan(p).any()]
            return "; colored objects" + guess + ": " + "; ".join(items)
        except Exception as e:  # missing opencv, unknown pose, ...
            return f"; (no detection: {e})"

    def t_pick(self, args):
        x, y = _xy(args)
        self.skills.pick(x, y, self.obj_h)
        return [_text(f"picked at ({x:.0f}, {y:.0f}); gripper closed, lifted to 60 mm")]

    def t_place(self, args):
        x, y = _xy(args)
        self.skills.place(x, y, self.obj_h)
        return [_text(f"placed at ({x:.0f}, {y:.0f}); gripper open, lifted to 60 mm")]

    def t_move_to(self, args):
        x, y = _xy(args)
        pitch = self.skills.move_to(x, y, float(args["z"]))
        return [_text(f"fingertip at ({x:.0f}, {y:.0f}, {float(args['z']):.0f}), "
                      f"gripper pitch {pitch:.0f} deg (-90 = straight down)")]

    def t_gripper(self, args):
        f = min(max(float(args["open"]), 0.0), 1.0)
        self.skills.gripper(f)
        return [_text(f"gripper {f:.2f} open")]

    def t_home(self, args):
        self.skills.home()
        return [_text("at rest pose")]

    # --- vision-driven moves ---
    def _look_pose(self, region):
        cx, cy = self.regions[region]["center"]
        self.skills.move_to(cx * LOOK_PULLBACK, cy * LOOK_PULLBACK, LOOK_Z)
        self.skills.wait(LOOK_SETTLE_S)

    def _calib(self):
        if not hasattr(self, "_lc"):
            from .ball import LookCalib
            self._lc = LookCalib.load()
        return self._lc

    def _locate(self, region):
        """Look at region -> (xy_mm or None, info dict, jpg or None)."""
        cx, cy = self.regions[region]["center"]
        half = self.regions[region]["size"] / 2 + 10
        self._look_pose(region)
        if self.snapshot is None:  # sim: ground truth
            for o in self.truth():
                x, y, z = o["xyz"]
                if abs(x - cx) <= half and abs(y - cy) <= half and z < self.obj_h:  # on the table
                    return (x, y), {"how": "sim truth"}, None
            return None, {"how": "sim truth", "seen": False}, None
        import cv2
        from .ball import find_ball
        jpg = self.snapshot()
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        b = find_ball(img)
        if b is None:
            return None, {"seen": False}, jpg
        info = {"seen": True, "pixel": (round(b["u"]), round(b["v"])), "r_px": round(b["r"])}
        if self._calib().has(region):
            x, y = self._calib().to_table(region, b["u"], b["v"])
            info["how"] = "calibrated"
            return (x, y), info, jpg
        info["how"] = "UNCALIBRATED: using region center"
        return (cx, cy), info, jpg

    def t_move_ball(self, args):
        src, dst = args["from"], args["to"]
        log = []
        xy, info, jpg = self._locate(src)
        log.append(f"look {src}: {info}")
        if xy is None:
            out = [_text("\n".join(log + ["no ball found; stopped (use look/pick/place)"]))]
            return out + ([_img(jpg)] if jpg else [])
        picked = False
        for attempt in (1, 2):
            self.skills.pick(*xy, self.obj_h)
            log.append(f"pick #{attempt} at ({xy[0]:.0f}, {xy[1]:.0f})")
            xy2, info2, jpg = self._locate(src)
            still = xy2 is not None and (self.snapshot is None or _near(info, info2))
            log.append(f"check {src}: {'ball still there ' + str(info2) if still else 'ball gone (in gripper)'}")
            if not still:
                picked = True
                break
            xy, info = xy2, info2
        if not picked:
            self.skills.gripper(self.skills.open_frac)
            self.skills.home()
            return [_text("\n".join(log + ["pick failed twice; stopped"]))] + ([_img(jpg)] if jpg else [])
        if "x" in args and "y" in args:
            tx, ty = float(args["x"]), float(args["y"])
        else:
            tx, ty = self.regions[dst]["center"]
        self.skills.place(tx, ty, self.obj_h)
        log.append(f"place at ({tx:.0f}, {ty:.0f})")
        xy3, info3, jpg = self._locate(dst)
        log.append(f"verify {dst}: {info3}" + (f" -> ~({xy3[0]:.0f}, {xy3[1]:.0f}) mm" if xy3 else ""))
        self.skills.home()
        ok = xy3 is not None
        log.append("DONE: ball is in " + dst if ok else "WARNING: ball not seen in " + dst)
        return [_text("\n".join(log))] + ([_img(jpg)] if jpg else [])

    def t_calibrate_look(self, args):
        region = args["region"]
        cx, cy = self.regions[region]["center"]
        if self.snapshot is None:
            return [_text("sim has no camera; nothing to calibrate")]
        import cv2
        from .ball import OFFSETS, find_ball
        pixels, table, log = [], [], []
        here = (cx, cy)
        for dx, dy in OFFSETS[1:]:  # ball starts at the center; end back at the center
            target = (cx + dx, cy + dy)
            self.skills.pick(*here, self.obj_h)
            self.skills.place(*target, self.obj_h)
            here = target
            self._look_pose(region)
            img = cv2.imdecode(np.frombuffer(self.snapshot(), np.uint8), cv2.IMREAD_COLOR)
            b = find_ball(img)
            log.append(f"ball at ({target[0]:.0f}, {target[1]:.0f}) -> "
                       + (f"pixel ({b['u']:.0f}, {b['v']:.0f})" if b else "NOT SEEN"))
            if b:
                pixels.append((b["u"], b["v"]))
                table.append(target)
        self.skills.home()
        pts = {}
        for p, t in zip(pixels, table):  # the center appears once in OFFSETS[1:] (last)
            pts[t] = p
        if len(pts) < 4:
            return [_text("\n".join(log + [f"only {len(pts)} spots seen; need 4. not saved"]))]
        err = self._calib().fit(region, list(pts.values()), list(pts.keys()))
        self._calib().save()
        return [_text("\n".join(log + [f"saved look calibration for {region}: "
                                        f"{len(pts)} points, max fit error {err:.1f} mm"]))]

    def t_status(self, args):
        mv = self.skills.arm.battery_mv()
        warn = " (LOW: servos weak, charge it)" if mv < 7000 else ""
        return [_text(json.dumps({"battery_v": round(mv / 1000, 2), "fingertip_mm": self._fingertip(),
                                  "regions": self.regions}) + warn)]

