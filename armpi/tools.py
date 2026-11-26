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

# Named spots on the paper (base frame mm). Defaults = Sean's two squares (60 mm, ~115 mm out).
REGIONS = {
    "left": {"center": (95.0, 65.0), "size": 60.0},
    "right": {"center": (95.0, -65.0), "size": 60.0},
}
LOOK_Z = 90.0  # fingertip height for a look; camera sees the region at a slant from here


def _text(s):
    return {"type": "text", "text": s}


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
            self.skills.move_to(cx, cy, LOOK_Z)
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

    def t_status(self, args):
        mv = self.skills.arm.battery_mv()
        warn = " (LOW: servos weak, charge it)" if mv < 7000 else ""
        return [_text(json.dumps({"battery_v": round(mv / 1000, 2), "fingertip_mm": self._fingertip(),
                                  "regions": self.regions}) + warn)]

