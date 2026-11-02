# ArmPi

Our own control stack for the Hiwonder ArmPi Mini, replacing HiwonderSDK. The Hiwonder
expansion board stays (it turns I2C commands into servo PWM); everything above it is ours.

## Start here

Everything below is typed in the **Ubuntu (WSL) terminal** on the PC, unless it says PowerShell.

### First time only (once ever)

1. Fix WSL networking so it can see the Pi. In **PowerShell**:

       notepad $env:USERPROFILE\.wslconfig

   Paste these two lines, save, close Notepad:

       [wsl2]
       networkingMode=mirrored

   Then, still in PowerShell: `wsl --shutdown`, and reopen the Ubuntu terminal.

2. Install Python's venv tool (asks for your PC's Linux password):

       sudo apt install -y python3-venv

3. Go to the repo and grab the latest code:

       cd "/mnt/c/Projects/armpi mini/armpi/ArmPi"
       git pull

4. Put the code on the Pi and make its server start by itself on every boot
   (asks for the **Pi's** password a few times, that's normal):

       tools/deploy.sh --serve

   It should end with `active`.

### Every time

1. Turn the Pi on. Wait ~1 minute for wifi.
2. In the Ubuntu terminal:

       cd "/mnt/c/Projects/armpi mini/armpi/ArmPi"
       tools/studio.sh

   The first run takes a few minutes (installs MuJoCo). It prints either
   `pi server found` or `sim only`.
3. In your browser open **http://localhost:8001/studio**
4. **Sim** button = practice arm (MuJoCo). **Real arm** button = the actual arm.
5. Done? Click in the terminal and press **Ctrl+C**.

### After pulling new code

    git pull
    tools/deploy.sh --serve

### When something's wrong

| You see | Do |
|---|---|
| `sim only` but the Pi is on | Pi got a new IP. PowerShell: `ping armpi.local`, then `ARMPI_HOST=<that ip> tools/studio.sh` (and the same prefix for deploy.sh) |
| `No route to host` | Step 1 of first-time setup didn't take: redo `.wslconfig` + `wsl --shutdown` |
| `python: command not found` / `No module named mujoco` | Use `tools/studio.sh`, not `python -m ...` |
| Real arm doesn't move, or shoulder/claw stall | Arm battery is low (studio shows battery in the top bar; want 7V+). Charge it |
| Pi password prompts every time | Once: `ssh-copy-id seanbeaulieu_armpi@10.0.0.133` |
| Want the Pi's own page | http://armpi.local:8000/studio (simple sim, no MuJoCo) |

Joint limits in `armpi/calibration.yaml` are still guesses: keep real-arm moves small until
the arm is calibrated.

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
