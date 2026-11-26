"""Square -> circle: move every object inside the square on the paper into the circle.

Pieces:
    Layout          where the square and circle are on the table (base frame, mm)
    slots()         free spots inside the circle to put objects (packed rings)
    score()         how many objects ended inside the circle
    run()           the scripted policy: perceive -> pick from square -> place in circle -> repeat
    sim_scene()     MuJoCo model of the arm + paper + shapes + cubes, for testing without the arm

`run` takes a `perceive()` callable returning [{"xyz": (x, y, z), "h": height_mm, ...}], so the
same policy runs on ground truth in sim, on the wrist cam (armpi.scene) or on an overhead cam.

    python -m armpi.task --sim               # run the whole task in MuJoCo, print the score
    python -m armpi.task --sim --cubes 4 --seed 3
    python -m armpi.task --url http://10.0.0.133:8000 demo    # real arm: one cube, hand-placed
                                                             # at the square center -> circle center
    python -m armpi.task --url http://10.0.0.133:8000 pick 95 65 place 95 -65 home
"""
import argparse
import math
import random
from dataclasses import dataclass


@dataclass
class Layout:
    """Defaults fit the reach of the ArmPi Mini (~85-170 mm out at table height, +-90 deg yaw):
    square on the left (+y), circle on the right (-y), both ~115 mm out."""
    square_c: tuple = (95.0, 65.0)   # center (x, y) mm
    square_side: float = 60.0
    square_yaw: float = 0.0          # deg; rotation of the square on the table
    circle_c: tuple = (95.0, -65.0)
    circle_r: float = 40.0

    def in_square(self, x, y, margin=0.0):
        dx, dy = x - self.square_c[0], y - self.square_c[1]
        a = math.radians(-self.square_yaw)
        u = dx * math.cos(a) - dy * math.sin(a)
        v = dx * math.sin(a) + dy * math.cos(a)
        h = self.square_side / 2 - margin
        return abs(u) <= h and abs(v) <= h

    def in_circle(self, x, y, margin=0.0):
        return math.dist((x, y), self.circle_c) <= self.circle_r - margin


def slots(layout, obj_mm=20.0, gap=4.0):
    """Spots inside the circle for objects of footprint obj_mm, center first, then rings.
    Every spot keeps the whole object inside the circle."""
    cx, cy = layout.circle_c
    pitch = obj_mm + gap
    out = [(cx, cy)]
    ring = 1
    while ring * pitch + obj_mm * 0.71 <= layout.circle_r:  # 0.71: cube corner, any yaw
        n = int(2 * math.pi * ring)  # spots on this ring at >= `pitch` spacing
        out += [(cx + ring * pitch * math.cos(2 * math.pi * k / n),
                 cy + ring * pitch * math.sin(2 * math.pi * k / n)) for k in range(n)]
        ring += 1
    return out


def score(layout, objects):
    """{"in_circle": n, "total": n, "frac": f} from objects' (x, y)."""
    n = sum(layout.in_circle(o["xyz"][0], o["xyz"][1]) for o in objects)
    return {"in_circle": n, "total": len(objects), "frac": n / len(objects) if objects else 1.0}


def run(skills, layout, perceive, obj_mm=20.0, max_picks=None, on_step=None):
    """The scripted policy. Picks objects still outside the circle (square ones first, nearest
    first) and places each on the free slot furthest from everything already there. Stops when
    nothing is left outside, or after max_picks (default 2x objects). Returns the final score."""
    objs = perceive()
    max_picks = max_picks if max_picks is not None else 2 * len(objs) + 1
    for step in range(max_picks):
        todo = [o for o in objs if not layout.in_circle(*o["xyz"][:2], margin=obj_mm / 2)]
        if not todo:
            break
        todo.sort(key=lambda o: (not layout.in_square(*o["xyz"][:2]), math.hypot(*o["xyz"][:2])))
        o = todo[0]
        taken = [p["xyz"][:2] for p in objs if layout.in_circle(*p["xyz"][:2])]
        free = [s for s in slots(layout, obj_mm)
                if all(math.dist(s, t) >= obj_mm for t in taken)]
        if not free:
            break
        x, y = o["xyz"][:2]
        skills.pick(x, y, o.get("h", obj_mm))
        sx, sy = free[0]
        skills.place(sx, sy, o.get("h", obj_mm))
        if on_step:
            on_step(step, o, (sx, sy))
        objs = perceive()
    skills.home()
    return score(layout, perceive())


# --- MuJoCo scene ---------------------------------------------------------------------------
CUBE_RGBA = ["0.85 0.15 0.15 1", "0.15 0.65 0.2 1", "0.15 0.35 0.85 1", "0.9 0.75 0.1 1",
             "0.6 0.2 0.7 1", "0.95 0.5 0.1 1"]


def scatter(layout, n, obj_mm=20.0, seed=0):
    """n non-overlapping cube centers inside the square."""
    rng = random.Random(seed)
    pts = []
    for _ in range(10000):
        if len(pts) == n:
            break
        h = layout.square_side / 2 - obj_mm * 0.6
        u, v = rng.uniform(-h, h), rng.uniform(-h, h)
        a = math.radians(layout.square_yaw)
        p = (layout.square_c[0] + u * math.cos(a) - v * math.sin(a),
             layout.square_c[1] + u * math.sin(a) + v * math.cos(a))
        if all(math.dist(p, q) >= obj_mm * 1.25 for q in pts):
            pts.append(p)
    if len(pts) < n:
        raise ValueError(f"can't fit {n} {obj_mm:g} mm cubes in the square")
    return pts


def scene_mjcf(layout, cubes, obj_mm=20.0, cube_g=5.0):
    """Arm MJCF + paper, drawn square + circle (visual only) and free cubes at `cubes` [(x, y)]."""
    from .mujoco_model import build_mjcf
    s = obj_mm / 2000.0
    sq, sc = layout.square_side / 1000.0, layout.square_c
    line = 0.0015  # 3 mm pen line
    sx, sy = sc[0] / 1000, sc[1] / 1000
    yaw = layout.square_yaw
    edges = "".join(
        f'<geom type="box" contype="0" conaffinity="0" rgba="0 0 0 1" size="{a} {b} 0.0003" '
        f'pos="{px} {py} 0.0006"/>'
        for a, b, px, py in [(sq / 2, line, 0, sq / 2), (sq / 2, line, 0, -sq / 2),
                             (line, sq / 2, sq / 2, 0), (line, sq / 2, -sq / 2, 0)])
    cx, cy, r = layout.circle_c[0] / 1000, layout.circle_c[1] / 1000, layout.circle_r / 1000
    ring = "".join(
        f'<geom type="box" contype="0" conaffinity="0" rgba="0 0 0 1" size="{line} {r * 0.11} 0.0003" '
        f'pos="{cx + r * math.cos(t)} {cy + r * math.sin(t)} 0.0006" euler="0 0 {math.degrees(t)}"/>'
        for t in [2 * math.pi * k / 60 for k in range(60)])
    bodies = "".join(
        f'<body name="cube{i}" pos="{x / 1000} {y / 1000} {s + 0.0005}">'
        f'<freejoint/><geom type="box" size="{s} {s} {s}" rgba="{CUBE_RGBA[i % len(CUBE_RGBA)]}" '
        f'mass="{cube_g / 1000}" friction="1.2 0.02 0.001"/></body>'
        for i, (x, y) in enumerate(cubes))
    extra = (f'<geom name="paper" type="box" contype="0" conaffinity="0" rgba="1 1 1 1" '
             f'size="0.1 0.1485 0.0002" pos="0.11 0 0.0002"/>'
             f'<body name="square" pos="{sx} {sy} 0" euler="0 0 {yaw}">{edges}</body>'
             f'{ring}{bodies}')
    return build_mjcf().replace("</worldbody>", extra + "</worldbody>")


def sim_scene(layout=None, n=3, obj_mm=20.0, seed=0):
    """(backend, skills, perceive) for a MuJoCo run. perceive() = ground-truth cube poses."""
    import mujoco
    from .backends.mujoco import MujocoBackend
    from .skills import Skills
    layout = layout or Layout()
    cubes = scatter(layout, n, obj_mm, seed)
    model = mujoco.MjModel.from_xml_string(scene_mjcf(layout, cubes, obj_mm))
    be = MujocoBackend(model=model)
    sk = Skills.for_mujoco(be)

    def perceive():
        return [{"name": f"cube{i}", "xyz": tuple(1000 * v for v in be.data.body(f"cube{i}").xpos),
                 "h": obj_mm} for i in range(len(cubes))]
    return be, sk, perceive


def real_skills(url, **kw):
    """Skills driving the real arm through the Pi's server (uses this checkout's calibration,
    so pull the Pi's calibration first: tools/pull-calibration.sh)."""
    from .arm import Arm
    from .backends.remote import RemoteBackend
    from .skills import Skills, true_z_to_cmd
    be = RemoteBackend(url)
    be.select("i2c")
    kw.setdefault("z_map", true_z_to_cmd)
    return Skills(Arm(be), **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="run in MuJoCo with ground-truth perception")
    ap.add_argument("--url", help="real arm via the Pi server, e.g. http://10.0.0.133:8000")
    ap.add_argument("--cubes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obj-h", type=float, default=20.0, help="object height, mm")
    ap.add_argument("cmds", nargs="*", help="with --url: demo | pick X Y | place X Y | home")
    a = ap.parse_args(argv)
    layout = Layout()
    if a.url:
        sk = real_skills(a.url)
        print(f"battery {sk.arm.battery_mv() / 1000:.2f} V")
        cmds = a.cmds or ["demo"]
        if cmds == ["demo"]:
            cmds = ["pick", *map(str, layout.square_c), "place", *map(str, layout.circle_c), "home"]
        i = 0
        while i < len(cmds):
            c = cmds[i]
            if c in ("pick", "place"):
                x, y = float(cmds[i + 1]), float(cmds[i + 2])
                print(f"{c} ({x:.0f}, {y:.0f})")
                getattr(sk, c)(x, y, a.obj_h)
                i += 3
            elif c == "home":
                sk.home()
                i += 1
            else:
                ap.error(f"unknown command {c!r}")
        return
    if not a.sim:
        ap.error("give --sim, or --url for the real arm")
    be, sk, perceive = sim_scene(layout, a.cubes, seed=a.seed)

    def step(i, o, slot):
        print(f"pick {o['name']} at ({o['xyz'][0]:.0f}, {o['xyz'][1]:.0f}) -> "
              f"place ({slot[0]:.0f}, {slot[1]:.0f})")
    res = run(sk, layout, perceive, on_step=step)
    for o in perceive():
        x, y, z = o["xyz"]
        print(f"  {o['name']}: ({x:.0f}, {y:.0f}, {z:.0f}) "
              f"{'IN circle' if layout.in_circle(x, y) else 'outside'}")
    print(f"score: {res['in_circle']}/{res['total']} in the circle, {be.data.time:.0f}s sim time")


if __name__ == "__main__":
    main()
