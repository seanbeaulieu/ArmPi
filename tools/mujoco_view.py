"""Watch the arm in MuJoCo running the same Arm code as the real thing.

    python tools/mujoco_view.py            # interactive viewer, loops a demo
    python tools/mujoco_view.py --save out.png   # headless snapshot of the demo end pose

Needs: pip install mujoco (works on CPU, any GPU or none).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armpi import Arm
from armpi.backends.mujoco import MujocoBackend

DEMO = [  # (angles or xyz-mm target, ms)
    ({"base": 0, "shoulder": 90, "elbow": 0, "wrist": 0, "gripper": 1}, 1500),
    ((120, 0, 40), 1200),
    ({"gripper": 0}, 500),
    ((90, 80, 90), 1200),
    ({"gripper": 1}, 500),
    (None, 1500),  # rest
]


def run_step(arm, step):
    target, ms = step
    if target is None:
        return arm.rest(ms)[1]
    if isinstance(target, tuple):
        arm.move_to(target, -90, ms)
        return ms
    return arm.move(target, ms)[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", help="render the demo end pose to this png and exit (headless)")
    args = p.parse_args()

    be = MujocoBackend()
    arm = Arm(be)
    import mujoco

    if args.save:
        for step in DEMO[:-1]:
            be.advance(run_step(arm, step) / 1000 + 0.3)
        r = mujoco.Renderer(be.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.05, 0, 0.1]
        cam.distance, cam.azimuth, cam.elevation = 0.6, 135, -25
        r.update_scene(be.data, cam)
        from PIL import Image
        Image.fromarray(r.render()).save(args.save)
        print("saved", args.save, "fingertip mm", [round(x) for x in be.fingertip_mm()])
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(be.model, be.data) as v:
        while v.is_running():
            for step in DEMO:
                ms = run_step(arm, step)
                end = time.monotonic() + ms / 1000 + 0.3
                while v.is_running() and time.monotonic() < end:
                    be.advance(1 / 60)
                    v.sync()
                    time.sleep(1 / 60)


if __name__ == "__main__":
    main()
