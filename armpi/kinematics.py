"""Forward and inverse kinematics. Units: mm and degrees, conventions in calibration.yaml.

Frame: origin = base center on the table, x forward, y left, z up.
Link lengths measured on the real arm (carried over from armpi-sim).
"""
import math

LINKS = {
    "base_height": 68.0,           # table -> top of turntable
    "turntable_to_shoulder": 27.5,  # turntable top -> shoulder pivot
    "shoulder_offset": 0.0,         # horizontal shoulder shift from base axis (assumed 0)
    "upper_arm": 55.0,              # shoulder pivot -> elbow pivot
    "forearm": 60.0,                # elbow pivot -> wrist pivot
    "wrist_to_gripper": 36.5,       # wrist pivot -> where fingers start
    "fingers": 50.0,                # fingertip = 86.5 from wrist pivot
}
SHOULDER_Z = LINKS["base_height"] + LINKS["turntable_to_shoulder"]
TOOL = LINKS["wrist_to_gripper"] + LINKS["fingers"]


def fk(angles, links=LINKS):
    """Joint angles -> dict of named 3D points (mm) plus 'pitch_deg'.

    Works in the arm's vertical plane (r = reach, z = height), then rotates by base yaw.
    """
    yaw = math.radians(angles["base"])
    a1 = math.radians(angles["shoulder"])
    a2 = a1 + math.radians(angles["elbow"])
    a3 = a2 + math.radians(angles["wrist"])

    plane = {"table": (0.0, 0.0), "turntable": (0.0, links["base_height"])}
    r, z = links["shoulder_offset"], links["base_height"] + links["turntable_to_shoulder"]
    plane["shoulder"] = (r, z)
    r, z = r + links["upper_arm"] * math.cos(a1), z + links["upper_arm"] * math.sin(a1)
    plane["elbow"] = (r, z)
    r, z = r + links["forearm"] * math.cos(a2), z + links["forearm"] * math.sin(a2)
    plane["wrist"] = (r, z)
    plane["gripper_base"] = (r + links["wrist_to_gripper"] * math.cos(a3),
                             z + links["wrist_to_gripper"] * math.sin(a3))
    tool = links["wrist_to_gripper"] + links["fingers"]
    plane["fingertip"] = (r + tool * math.cos(a3), z + tool * math.sin(a3))

    pts = {k: (r * math.cos(yaw), r * math.sin(yaw), z) for k, (r, z) in plane.items()}
    pts["pitch_deg"] = math.degrees(a3)
    return pts


def ik(xyz, pitch, links=LINKS):
    """Fingertip position (mm) + gripper pitch (deg) -> joint angles, or None if unreachable.

    Elbow-up solution (elbow <= 0). Joint limits are not checked here; see ik_search.
    """
    x, y, z = xyz
    yaw = math.degrees(math.atan2(y, x)) if (x or y) else 0.0
    tool = links["wrist_to_gripper"] + links["fingers"]
    p = math.radians(pitch)
    # wrist pivot in the arm plane, relative to the shoulder
    rw = math.hypot(x, y) - links["shoulder_offset"] - tool * math.cos(p)
    zw = z - links["base_height"] - links["turntable_to_shoulder"] - tool * math.sin(p)
    l2, l3 = links["upper_arm"], links["forearm"]
    c = (rw * rw + zw * zw - l2 * l2 - l3 * l3) / (2 * l2 * l3)
    if abs(c) > 1:
        return None
    el = -math.acos(c)
    sh = math.atan2(zw, rw) - math.atan2(l3 * math.sin(el), l2 + l3 * math.cos(el))
    sh, el = math.degrees(sh), math.degrees(el)
    return {"base": yaw, "shoulder": sh, "elbow": el, "wrist": pitch - sh - el}


def ik_search(xyz, pitch, limits, span=90, step=1.0, links=LINKS):
    """Like ik(), but if the exact pitch is unreachable or breaks a joint limit, try pitches
    progressively further from it (up to +-span deg). limits: {joint: (lo, hi)}.
    Returns (angles, pitch_used) or None.
    """
    n = int(span / step)
    for i in range(2 * n + 1):
        d = (i + 1) // 2 * step * (1 if i % 2 else -1)
        a = ik(xyz, pitch + d, links)
        if a and all(limits[j][0] <= a[j] <= limits[j][1] for j in a):
            return a, pitch + d
    return None
