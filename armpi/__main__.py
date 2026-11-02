"""Command line: python -m armpi <command> [--sim]

    battery                      battery voltage
    move base=10 elbow=-30 ...   joint angles in degrees (gripper 0..1)
    to X Y Z [--pitch -90]       fingertip position in mm
    pulse JOINT US               raw pulse (trim included), ignores angle limits; for calibration
    rest                         the folded boot pose
    wiggle [JOINT ...]           small nudges around the rest pose, one joint at a time

Each run is a fresh process, so the arm's pose is unknown and the first move is slow
(calibration.yaml first_move_ms).
"""
import argparse
import time

from . import connect
from .calibration import JOINTS

# nudge size per joint: the gripper and the loaded shoulder need more before it's visible
WIGGLE = {"gripper": 0.3, "wrist": 5, "elbow": 5, "shoulder": 9, "base": 5}


def main(argv=None):
    p = argparse.ArgumentParser(prog="armpi")
    p.add_argument("--sim", action="store_true", help="use the simulator instead of the board")
    p.add_argument("--ms", type=int, default=None, help="move duration (raised if too fast)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("battery")
    m = sub.add_parser("move")
    m.add_argument("joints", nargs="+", metavar="JOINT=DEG")
    t = sub.add_parser("to")
    t.add_argument("xyz", nargs=3, type=float)
    t.add_argument("--pitch", type=float, default=-90)
    u = sub.add_parser("pulse")
    u.add_argument("joint", choices=JOINTS)
    u.add_argument("us", type=int)
    sub.add_parser("rest")
    w = sub.add_parser("wiggle")
    w.add_argument("joints", nargs="*", choices=JOINTS)
    a = p.parse_args(argv)

    arm = connect("sim" if a.sim else "i2c")
    if a.cmd == "battery":
        print(f"{arm.battery_mv()} mV")
    elif a.cmd == "move":
        target = {k: float(v) for k, v in (s.split("=", 1) for s in a.joints)}
        got, ms = arm.move(target, a.ms, wait=True)
        clamped = {j: round(got[j], 1) for j in got if abs(got[j] - target[j]) > 1e-9}
        print(f"moved in {ms} ms" + (f", clamped to limits: {clamped}" if clamped else ""))
    elif a.cmd == "to":
        pitch = arm.move_to(tuple(a.xyz), a.pitch, a.ms, wait=True)
        print(f"pitch {pitch:.0f}, joints " + ", ".join(f"{j}={v:.1f}" for j, v in arm.angles.items()))
    elif a.cmd == "pulse":
        ms = arm.move_pulse(a.joint, a.us, a.ms, wait=True)
        print(f"{a.joint} -> {a.us}us ({arm.calib.to_angle(a.joint, a.us):.1f}) in {ms} ms")
    elif a.cmd == "rest":
        arm.rest(a.ms, wait=True)
    elif a.cmd == "wiggle":
        rest = arm.calib.rest()
        arm.rest(wait=True)
        for j in a.joints or ["gripper", "wrist", "elbow", "shoulder", "base"]:
            print("wiggle", j)
            for d in (WIGGLE[j], -WIGGLE[j], 0):
                arm.move({j: rest[j] + d}, 300, wait=True)
                time.sleep(0.1)
    arm.close()


if __name__ == "__main__":
    main()
