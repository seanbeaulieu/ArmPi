"""Die pick/place tasks run from the web panel (armpi.server), one at a time, in a thread.

    POST /api/task       {"name": "move", "from": "right", "to": "left"}   -> the run record
                         {"name": "locate", "region": "right"}             look only
                         {"name": "pick", "region": "right"}               pick and keep holding
                         {"name": "place", "region": "left"}               set the held die down
                         {"name": "home"}
    POST /api/task/stop  stop the running task after its current arm move
    GET  /api/tasks      {"running": id | null, "runs": [...newest last], "scene": {...}}
    GET  /api/task/photo.jpg   the last wrist-cam photo a task took

Each run: {"id", "name", "args", "status": "running" | "pass" | "fail" | "stopped",
"started", "ended", "steps": [{"t", "text"}], "result"}. Steps also go to the server's event
log (kind "task"), so the studio's log shows them in line with the arm commands.

The scene is what the tasks know about the die: {"die": {"xy": [x, y] | null, "status",
"t"}, "regions": {...}}; status "on the table", "held", "placed (unchecked)", "lost".
Tasks drive the server's real arm (its "i2c" Arm, so the 3D view follows them) with the
measured height correction, and take photos from the real arm's camera.
"""
import collections
import itertools
import threading


class Stopped(Exception):
    pass


class TaskRunner:
    NAMES = ("move", "locate", "pick", "place", "home")

    def __init__(self, app):
        self.app = app
        self.runs = collections.deque(maxlen=30)
        self.ids = itertools.count(1)
        self.running = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.die = {"xy": None, "status": "unknown", "t": None}
        self.photo = None
        self._skills = None

    # --- API ---
    def start(self, body):
        from .tools import REGIONS
        name = body.get("name")
        if name not in self.NAMES:
            raise ValueError(f"task must be one of {', '.join(self.NAMES)}, not {name!r}")
        for k in ("from", "to", "region"):
            if k in body and body[k] not in REGIONS:
                raise ValueError(f"{k} must be one of {', '.join(REGIONS)}")
        if name == "move" and not ("from" in body and "to" in body):
            raise ValueError("move needs from and to")
        if name in ("locate", "pick", "place") and "region" not in body:
            raise ValueError(f"{name} needs a region")
        with self.lock:
            if self.running is not None:
                raise ValueError(f"task {self.running['name']} #{self.running['id']} is still running")
            run = {"id": next(self.ids), "name": name,
                   "args": {k: v for k, v in body.items() if k != "name"},
                   "status": "running", "started": self.app.now(), "ended": None,
                   "steps": [], "result": None}
            self.runs.append(run)
            self.running = run
            self.stop_flag.clear()
        threading.Thread(target=self._run, args=(run,), daemon=True).start()
        return run

    def stop(self):
        run = self.running
        if run is None:
            return {"stopped": None}
        self.stop_flag.set()
        return {"stopping": run["id"]}

    def listing(self):
        return {"running": self.running and self.running["id"], "runs": list(self.runs),
                "scene": self.scene()}

    def scene(self):
        from .tools import REGIONS
        return {"die": dict(self.die), "regions": REGIONS,
                "task": self.running and f"{self.running['name']} #{self.running['id']}"}

    # --- running ---
    def _step(self, run, text):
        if self.stop_flag.is_set():
            raise Stopped()
        run["steps"].append({"t": self.app.now(), "text": text})
        self.app.event("task", f"#{run['id']} {text}")

    def _set_die(self, xy, status):
        self.die = {"xy": None if xy is None else [round(xy[0], 1), round(xy[1], 1)],
                    "status": status, "t": self.app.now()}

    def _grasper(self, run):
        from . import vision
        from .grasp import Grasper
        from .skills import Skills, true_z_to_cmd
        app = self.app
        if app.backend_name != "i2c":
            with app.lock:
                app.use("i2c")  # tasks need the real arm and its camera
        arm = app.arms["i2c"]
        runner = self

        class StoppableSkills(Skills):
            def _go(self, xyz, ms, raw=False):
                if runner.stop_flag.is_set():
                    raise Stopped()
                return super()._go(xyz, ms, raw)

        sk = StoppableSkills(arm, z_map=true_z_to_cmd)
        if app.real_url:
            def snapshot():
                jpg = vision._get(app.real_url, "/snapshot.jpg")
                self.photo = jpg
                return jpg
        else:
            def snapshot():
                jpg = app.camera.frame()
                if jpg is None:
                    raise OSError(app.camera.error or "no camera frame")
                self.photo = jpg
                return jpg
        g = Grasper(sk, snapshot, log=lambda m: self._step(run, m.strip()))
        g.on_die = self._set_die
        return g, sk

    def _run(self, run):
        from .tools import REGIONS
        a = run["args"]
        center = lambda r: tuple(REGIONS[r]["center"])  # noqa: E731
        try:
            g, sk = self._grasper(run)
            if run["name"] == "move":
                ok, _, _ = g.move(center(a["from"]), center(a["to"]))
                result = f"die moved {a['from']} -> {a['to']}" if ok else "move failed"
            elif run["name"] == "locate":
                if not g.view:
                    raise RuntimeError("no view map (grasp_view.yaml): teach-view first")
                xy, d = g.locate(*center(a["region"]))
                ok = xy is not None
                if not ok:
                    self._set_die(None, "not seen")
                result = (f"die at ({xy[0]:.0f}, {xy[1]:.0f}) mm" if ok else
                          "die only partly in view" if d else "no die in " + a["region"])
            elif run["name"] == "pick":
                pick = g.pick_direct if g.view else g.pick
                ok, why, _ = pick(*center(a["region"]))
                result = why
            elif run["name"] == "place":
                g.place(*center(a["region"]))
                self._set_die(center(a["region"]), "placed (unchecked)")
                ok, result = True, f"placed in {a['region']}"
            else:  # home
                sk.gripper(sk.open_frac)
                sk.home()
                ok, result = True, "at rest"
            self._step(run, ("PASS: " if ok else "FAIL: ") + result)
            run["status"], run["result"] = ("pass" if ok else "fail"), result
        except Stopped:
            run["status"], run["result"] = "stopped", "stopped from the panel"
            self.app.event("task", f"#{run['id']} stopped", warn=True)
        except Exception as e:  # network, camera, unreachable pose, ...
            run["status"], run["result"] = "fail", f"{type(e).__name__}: {e}"
            self.app.event("error", f"task #{run['id']}: {type(e).__name__}: {e}")
        finally:
            run["ended"] = self.app.now()
            with self.lock:
                self.running = None
