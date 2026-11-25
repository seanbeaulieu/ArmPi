"""See the task from a camera image: the drawn square, the drawn circle, and the objects.

Works in pixels, then maps to the table (base frame, mm) with any `to_table(uv) -> (N,3)`:
    wrist cam:    lambda uv: vision.pixel_to_xyz(uv, angles, cam)       (angles = survey pose)
    overhead cam: homography(...)                                       (4 touched points)

Assumes dark pen lines on light paper and colored (saturated) objects. Objects are found by
saturation, not by named color, so any colored thing counts; gray/black/white ones don't.

    python -m armpi.scene photo.jpg --out annotated.jpg        # pixels only, for tuning
    python -m armpi.scene --url http://10.0.0.133:8000          # wrist cam at its current pose
"""
import argparse
import math

import numpy as np


def _cv2():
    import cv2
    return cv2


def find_shapes(bgr, min_area_frac=0.003):
    """Drawn outlines -> {"square": {...} | None, "circle": {...} | None} in pixels.
    square: {"corners": 4x2, "center", "area"}; circle: {"center", "radius", "area"}.
    Picks the biggest closed square-ish and circle-ish outlines."""
    cv2 = _cv2()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # pen = dark AND unsaturated (so a dark-red object isn't a line)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    dark &= (hsv[..., 1] < 90).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # every closed contour (a drawn outline gives an outer and an inner one; the paper's own
    # edge on a darker table gives another): classify all, then pick
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    min_area = min_area_frac * gray.size
    quads, circles = [], []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area < min_area or x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue  # too small, or touches the image edge
        peri = cv2.arcLength(c, True)
        circularity = 4 * math.pi * area / (peri * peri)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if len(approx) == 4 and cv2.isContourConvex(approx) and circularity < 0.85:
            q = approx.reshape(4, 2).astype(float)
            sides = [np.linalg.norm(q[i] - q[i - 1]) for i in range(4)]
            if min(sides) / max(sides) > 0.75:  # square-ish (an A4 sheet is 0.71)
                quads.append({"corners": q.tolist(), "center": q.mean(0).tolist(),
                              "area": float(area), "contour": c})
        elif circularity > 0.8 and area > 0.7 * math.pi * r * r:
            circles.append({"center": [cx, cy], "radius": float(r), "area": float(area)})
    circle = max(circles, key=lambda d: d["area"], default=None)
    if circle:  # the drawn square never contains the circle (the paper outline would)
        quads = [q for q in quads if cv2.pointPolygonTest(q["contour"], tuple(circle["center"]), False) < 0]
    square = max(quads, key=lambda d: d["area"], default=None)
    if square:
        del square["contour"]
    return {"square": square, "circle": circle}


def find_objects(bgr, min_sat=110, min_val=60, min_area_frac=0.0006):
    """Colored blobs -> [{"u", "v", "area", "box", "hue"}], biggest first."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] >= min_sat) & (hsv[..., 2] >= min_val)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * mask.size:
            continue
        mo = cv2.moments(c)
        u, v = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        hue = float(np.median(hsv[..., 0][cv2.drawContours(np.zeros_like(mask), [c], -1, 1, -1) > 0]))
        out.append({"u": u, "v": v, "area": float(area),
                    "box": [int(x) for x in cv2.boundingRect(c)], "hue": hue})
    return sorted(out, key=lambda o: -o["area"])


def homography(pixels, table_xy):
    """Overhead camera: 4+ pixel points with known table (x, y) mm -> to_table(uv) for z = 0.
    Get the points by jogging the fingertip onto 4 marks on the paper and reading arm.fk()."""
    cv2 = _cv2()
    H, _ = cv2.findHomography(np.asarray(pixels, float), np.asarray(table_xy, float))

    def to_table(uv):
        p = cv2.perspectiveTransform(np.asarray(uv, float).reshape(-1, 1, 2), H).reshape(-1, 2)
        return np.column_stack([p, np.zeros(len(p))])
    return to_table


def perceive(bgr, to_table, obj_h=20.0):
    """Image -> {"square": {"center", "corners"}, "circle": {"center", "radius"},
    "objects": [{"xyz", "h", "uv", "hue"}]} in table mm (objects' xyz = center, z = h/2).

    `to_table` maps pixels onto z = 0. An object's blob is mostly its top face, so from an
    oblique view its xyz is off by ~h * tan(view angle); for a top-down view that's small."""
    shapes = find_shapes(bgr)
    out = {"square": None, "circle": None, "objects": []}
    if shapes["square"]:
        c = to_table(shapes["square"]["corners"])[:, :2]
        out["square"] = {"corners": c.tolist(), "center": c.mean(0).tolist(),
                         "side": float(np.mean([np.linalg.norm(c[i] - c[i - 1]) for i in range(4)]))}
    if shapes["circle"]:
        s = shapes["circle"]
        cu, cv_, r = s["center"][0], s["center"][1], s["radius"]
        ring = [(cu + r * math.cos(t), cv_ + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 16, False)]
        p = to_table(ring)[:, :2]
        ctr = p.mean(0)
        out["circle"] = {"center": ctr.tolist(), "radius": float(np.linalg.norm(p - ctr, axis=1).mean())}
    for o in find_objects(bgr):
        p = to_table([(o["u"], o["v"])])[0]
        if np.isnan(p).any():
            continue
        out["objects"].append({"xyz": (float(p[0]), float(p[1]), obj_h / 2), "h": obj_h,
                               "uv": (o["u"], o["v"]), "hue": o["hue"]})
    return out


def layout_from(scene, fallback=None):
    """Turn a perceived scene into a task.Layout (falls back to `fallback` for missing shapes)."""
    from .task import Layout
    lay = fallback or Layout()
    if scene.get("square"):
        c = np.asarray(scene["square"]["corners"])
        e = c[1] - c[0]
        lay.square_c = tuple(scene["square"]["center"])
        lay.square_side = scene["square"]["side"]
        lay.square_yaw = math.degrees(math.atan2(e[1], e[0])) % 90
    if scene.get("circle"):
        lay.circle_c = tuple(scene["circle"]["center"])
        lay.circle_r = scene["circle"]["radius"]
    return lay


def annotate(bgr, shapes, objects):
    cv2 = _cv2()
    img = bgr.copy()
    if shapes["square"]:
        cv2.polylines(img, [np.int32(shapes["square"]["corners"])], True, (0, 200, 255), 2)
    if shapes["circle"]:
        c = shapes["circle"]
        cv2.circle(img, (int(c["center"][0]), int(c["center"][1])), int(c["radius"]), (255, 120, 0), 2)
    for i, o in enumerate(objects):
        cv2.circle(img, (int(o["u"]), int(o["v"])), 5, (255, 255, 255), -1)
        cv2.putText(img, str(i), (int(o["u"]) + 6, int(o["v"]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return img


def main(argv=None):
    cv2 = _cv2()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image", nargs="?", help="image file (else --url snapshot)")
    ap.add_argument("--url", help="pi server, e.g. http://10.0.0.133:8000 (wrist cam + angles)")
    ap.add_argument("--out", help="write an annotated image here")
    ap.add_argument("--obj-h", type=float, default=20.0)
    a = ap.parse_args(argv)
    if a.image:
        bgr = cv2.imread(a.image)
        to_table = None
    elif a.url:
        from . import vision
        bgr = vision.snapshot(a.url)
        angles = vision._angles(a.url)
        cam = vision.Camera.load()
        h, w = bgr.shape[:2]
        cam = cam if (w, h) == cam.size else cam.scaled((w, h))
        to_table = lambda uv: vision.pixel_to_xyz(uv, angles, cam)  # noqa: E731
    else:
        ap.error("give an image or --url")
    shapes, objs = find_shapes(bgr), find_objects(bgr)
    print("square:", shapes["square"] and [round(v) for v in shapes["square"]["center"]], "px")
    print("circle:", shapes["circle"] and [round(v) for v in shapes["circle"]["center"]], "px")
    print(f"{len(objs)} objects:", [(round(o["u"]), round(o["v"])) for o in objs])
    if to_table:
        s = perceive(bgr, to_table, a.obj_h)
        print("table mm:", {k: s[k] for k in ("square", "circle")})
        for o in s["objects"]:
            print("  object at", [round(v) for v in o["xyz"]])
    if a.out:
        cv2.imwrite(a.out, annotate(bgr, shapes, objs))


if __name__ == "__main__":
    main()
