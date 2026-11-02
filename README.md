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
    tools/deploy.sh          sync to the Pi and install

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
