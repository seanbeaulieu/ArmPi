# ArmPi

Our own control stack for the Hiwonder ArmPi Mini, replacing HiwonderSDK. The Hiwonder
expansion board stays (it turns I2C commands into servo PWM); everything above it is ours.

## Hardware facts

- Raspberry Pi 4, 64-bit Pi OS (trixie). Host `armpi.local`, user `seanbeaulieu_armpi`.
- Servos are plain PWM hobby servos: **no position feedback**. Everything is "last commanded".
- They are driven by the expansion board MCU at I2C bus 1, address `0x7A` (bus at 40 kHz).
  `i2cdetect -y 1` stops at 0x77, so it won't show the board; use `i2cdetect -y -a 1`.
- Servo ids: gripper 1, wrist 3, elbow 4, shoulder 5, base 6.

## Layout

    armpi/protocol.py        0x7A packet encoding (pure, tested against Board.py bytes)
    armpi/backends/i2c.py    real board over smbus2
    armpi/backends/sim.py    simulated board (same interface, linear motion like the MCU)
    armpi/calibration.yaml   angle <-> pulse per joint, limits, speed guard
    armpi/calibration.py     the conversion
    armpi/kinematics.py      FK / IK (mm, degrees)
    armpi/arm.py             Arm: named joints, clamps to limits, speed guard
    armpi/mujoco_model.py    MuJoCo MJCF generated from the same link lengths (checked == fk)
    armpi/backends/mujoco.py physics sim: board-style pulse ramps -> position servos w/ torque limits
    tools/deploy.sh          sync to the Pi and install
    tools/mujoco_view.py     watch the arm run a demo in MuJoCo

Angle conventions are at the top of `calibration.yaml`. Gripper is 0 (closed) .. 1 (open).

## Use

```python
import armpi
arm = armpi.connect()            # or armpi.connect("sim")
arm.rest(wait=True)
arm.move({"base": 20, "gripper": 1}, 800, wait=True)
arm.move_to((100, 0, 30), pitch=-90, wait=True)   # fingertip, mm
print(arm.battery_mv())
```

From the shell on the Pi (`cd ~/ArmPi`; add `--sim` to try without the arm):

```bash
.venv/bin/python -m armpi battery
.venv/bin/python -m armpi wiggle
.venv/bin/python -m armpi move base=20 elbow=-40
.venv/bin/python -m armpi pulse shoulder 900    # raw, for calibration
```

Safety: angles are clamped to the limits in `calibration.yaml`, and moves are slowed so no
servo goes faster than `min_ms_per_us`. The first move of each process takes
`first_move_ms` because the starting pose is unknown.

## Develop

From Git Bash on the PC:

```bash
tools/deploy.sh --test    # copy the working tree to ~/ArmPi on the Pi, install, run tests
```

Tests need no hardware: `python tests/test_armpi.py` (or pytest).

## Quick start

Once, from the PC (WSL/Git Bash, in this repo):

    tools/deploy.sh --serve     # copy to the Pi, install, run its server at boot from now on

Every time:

    tools/studio.sh             # PC: http://localhost:8001/studio, Ctrl+C to stop

Pi off? Studio still runs with the MuJoCo sim only. After code changes: `git pull` and
`tools/deploy.sh --serve` again.

## Arm Studio (GUI)

`python -m armpi.server --sim mujoco` then open http://localhost:8000/studio (on the Pi:
http://armpi.local:8000/studio; `--sim mujoco` needs `pip install mujoco`, otherwise the plain
kinematic sim is used). One page shows:

- 3D view: actual pose (physics) solid, commanded target dashed, 5 s fingertip trail, reach
  envelope, joints turning orange near their calibration limits. Drag to orbit, wheel to zoom,
  shift-drag to pan. "click-to-reach" + click on the table sends a move_to at the given height.
- Joints: calibration limit bars with commanded and actual markers; drag a bar to move a joint.
- Joint tracking: last 10 s of commanded / board ramp / actual per joint, with max tracking error.
  This is the sim2real gap made visible: once gains/masses are tuned, it should match the arm.
- Event log: every command, clamps and pitch substitutions in orange, errors in bold.

Backed by `GET /api/history?since=t` (20 Hz samples) and `GET /api/log?since=t`.

The real arm is wired to the Pi, so "Real arm" on the PC needs the Pi's server running:
on the Pi `.venv/bin/python -m armpi.server`, on the PC
`python -m armpi.server --sim mujoco --real-url http://armpi.local:8000`. The PC studio then
drives the Pi's arm, while sim stays local.

## MuJoCo

`pip install mujoco` (CPU is fine). Same Arm code, physics underneath:

```python
from armpi import Arm
from armpi.backends.mujoco import MujocoBackend
be = MujocoBackend()            # deterministic: time advances only via be.advance()
arm = Arm(be)
arm.move_to((120, 0, 40), -90, 1200)
be.advance(1.5)
print(be.joint_angles(), be.fingertip_mm())   # where the joints actually ended up
```

`armpi.connect("mujoco")` gives a wall-clock version (call `sync()`). Servo gains, torque and
link masses in `mujoco_model.py` are estimates; tune them against the real arm (step response
filmed with the wrist cam) to close the sim2real gap. `python tools/mujoco_view.py` opens a viewer.

CI (`.github/workflows/test.yml`) runs all tests, MuJoCo included, on every push.
