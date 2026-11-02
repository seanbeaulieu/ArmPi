"""Byte-level protocol of the Hiwonder expansion board MCU (I2C bus 1, address 0x7A).

Pure functions only, no hardware imports, so they can be tested anywhere.
Reverse-read from HiwonderSDK/Board.py:

    servo move   write [40, n, t_lo, t_hi, id1, p1_lo, p1_hi, id2, ...]
                 n servos, t = move time in ms (0..30000), p = pulse in us (500..2500)
    battery      write [0], then read 2 bytes little-endian = millivolts
"""

I2C_BUS = 1
I2C_ADDR = 0x7A

REG_BATTERY = 0
REG_SERVO_ANGLE = 21   # per-servo angle register (unused: pulses are more precise)
REG_MOTOR = 31         # dc motor ports (unused on the ArmPi Mini)
REG_SERVO_CMD = 40

PULSE_MIN = 500
PULSE_MAX = 2500
TIME_MAX_MS = 30000
SERVO_IDS = range(1, 7)


def encode_servo_move(pulses, duration_ms):
    """pulses: {servo_id: pulse_us}. Returns the bytes to write to the board.

    Raises instead of clamping: callers (Arm) are responsible for limits, so an
    out-of-range value here is a bug, not something to silently fix.
    """
    if not pulses:
        raise ValueError("no servos given")
    duration_ms = int(duration_ms)
    if not 0 <= duration_ms <= TIME_MAX_MS:
        raise ValueError(f"duration {duration_ms}ms outside [0, {TIME_MAX_MS}]")
    buf = [REG_SERVO_CMD, len(pulses)] + list(duration_ms.to_bytes(2, "little"))
    for sid, us in pulses.items():
        if sid not in SERVO_IDS:
            raise ValueError(f"invalid servo id {sid}")
        us = int(round(us))
        if not PULSE_MIN <= us <= PULSE_MAX:
            raise ValueError(f"servo {sid} pulse {us}us outside [{PULSE_MIN}, {PULSE_MAX}]")
        buf += [sid] + list(us.to_bytes(2, "little"))
    return buf


def encode_battery_request():
    return [REG_BATTERY]


def decode_battery(data):
    """2 bytes little-endian -> millivolts."""
    return int.from_bytes(bytes(data[:2]), "little")
