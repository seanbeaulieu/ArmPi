"""Hardware-free tests. Run: python -m pytest tests   (or: python tests/test_armpi.py)"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fixed_calibration  # noqa: E402,F401  (tests use a frozen calibration)

from armpi import Arm, Calibration, protocol
from armpi.backends import SimBackend
from armpi.kinematics import fk, ik, ik_search

CAL = Calibration.load()


def close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# --- protocol: must produce the same bytes as HiwonderSDK/Board.py ---

def test_golden_gripper_packet():
    # Board.setPWMServoPulse(1, 1500, 300) with Deviation.yaml '1': -124 -> pulse 1376
    us = CAL.to_pulse("gripper", 0)
    assert us == 1376
    assert protocol.encode_servo_move({1: us}, 300) == [40, 1, 44, 1, 1, 96, 5]


def test_golden_multi_servo_packet():
    # Board.setPWMServosPulse([1000, 2, 3, 695, 6, 1500]) with trims 3:-39, 6:-73
    buf = protocol.encode_servo_move({3: 695 - 39, 6: 1500 - 73}, 1000)
    assert buf == [40, 2, 0xE8, 0x03, 3, 0x90, 0x02, 6, 0x93, 0x05]


def test_protocol_rejects_bad_values():
    for pulses, ms in [({1: 2600}, 100), ({7: 1500}, 100), ({1: 1500}, 40000), ({}, 100)]:
        try:
            protocol.encode_servo_move(pulses, ms)
        except ValueError:
            continue
        raise AssertionError(f"accepted {pulses} {ms}")


def test_battery_decode():
    assert protocol.decode_battery([0xE5, 0x11]) == 4581


# --- calibration: same pulses as Hiwonder's ArmIK.transformAngelAdaptArm ---

def hiwonder_pulses(t3, t4, t5, t6):
    k = 2000 / 180
    s6 = (90 - (90 - t6)) * k + 500 if t6 >= -90 else (90 + 90 + 180 + t6) * k
    return {"wrist": t3 * k + 1500, "elbow": t4 * k + 1500, "shoulder": 1500 + (90 - t5) * k,
            "base": s6}


def test_mapping_matches_hiwonder():
    for t3, t4, t5, t6 in [(0, 0, 90, 90), (-30, 60, 70, 45), (20, -10, 120, 150)]:
        ours = {"wrist": t3, "elbow": -t4, "shoulder": t5, "base": t6 - 90}
        theirs = hiwonder_pulses(t3, t4, t5, t6)
        for j, a in ours.items():
            got = CAL.to_pulse(j, a) - CAL.joints[j]["trim"]
            assert abs(got - theirs[j]) <= 1, (j, got, theirs[j])


def test_pulse_roundtrip():
    for j, a in [("base", 30), ("shoulder", 45), ("elbow", -60), ("wrist", 20), ("gripper", 0.5)]:
        assert abs(CAL.to_angle(j, CAL.to_pulse(j, a)) - a) < 0.1


def test_rest_pose_is_hiwonder_servo_test_center():
    # Servo_test.py centers before trim: 3->695, 4->2415, 5->780, 6->1500, 1->1500
    want = {"wrist": 695, "elbow": 2415, "shoulder": 780, "base": 1500, "gripper": 1500}
    for j, a in CAL.rest().items():
        assert abs(CAL.to_pulse(j, a) - CAL.joints[j]["trim"] - want[j]) <= 1, j


# --- kinematics ---

def test_fk_straight_up():
    p = fk({"base": 0, "shoulder": 90, "elbow": 0, "wrist": 0})
    assert close(p["fingertip"], (0, 0, 297.0))


def test_fk_straight_forward_and_left():
    assert close(fk({"base": 0, "shoulder": 0, "elbow": 0, "wrist": 0})["fingertip"], (201.5, 0, 95.5))
    assert close(fk({"base": 90, "shoulder": 0, "elbow": 0, "wrist": 0})["fingertip"], (0, 201.5, 95.5))


def test_ik_roundtrip():
    for xyz, pitch in [((100, 0, 30), -90), ((100, 80, 60), -45), ((120, -50, 150), 0)]:
        a = ik(xyz, pitch)
        assert a is not None, xyz
        p = fk(a)
        assert close(p["fingertip"], xyz, 1e-6) and math.isclose(p["pitch_deg"], pitch, abs_tol=1e-9)
        assert a["elbow"] <= 0


def test_ik_unreachable():
    assert ik((500, 0, 0), -90) is None


def test_ik_search_respects_limits():
    limits = {j: CAL.limits(j) for j in ["base", "shoulder", "elbow", "wrist"]}
    a, pitch = ik_search((150, 0, 30), -90, limits)
    assert all(limits[j][0] <= a[j] <= limits[j][1] for j in a)


# --- arm ---

def make_arm():
    clock = Clock()
    return Arm(SimBackend(clock=clock), CAL), clock


def test_first_move_is_slow_then_speed_guard():
    arm, clock = make_arm()
    _, ms = arm.move({"base": 0}, 100)
    assert ms == CAL.first_move_ms
    _, ms = arm.move({"base": 45}, 100)  # 500us change at 1 ms/us
    assert ms == 500


def test_move_clamps_to_limits():
    arm, _ = make_arm()
    got, _ = arm.move({"base": 500})
    assert got["base"] == CAL.limits("base")[1]


def test_move_sends_servo_ids_and_trim():
    arm, clock = make_arm()
    arm.move({"gripper": 0, "base": 0})
    _, pulses, _ = arm.backend.history[-1]
    assert pulses == {1: 1376, 6: 1427}


def test_sim_interpolates():
    arm, clock = make_arm()
    arm.move({"base": 0}, 0)
    _, ms = arm.move({"base": 45}, 1000)
    clock.t += ms / 2000
    mid = arm.backend.pulses_at()[6]
    assert abs(mid - (1427 + 250)) < 1


def test_move_to():
    arm, _ = make_arm()
    pitch = arm.move_to((150, 0, 30), -90)
    p = fk(arm.angles)
    assert close(p["fingertip"], (150, 0, 30), 1.0)  # pulses are whole us, so ~0.1deg error
    assert pitch is not None


def test_unknown_joint_raises():
    arm, _ = make_arm()
    try:
        arm.move({"elbow2": 0})
    except KeyError:
        return
    raise AssertionError


def test_i2c_backend_bytes():
    try:
        import smbus2  # noqa: F401
    except ImportError:
        return  # PC without smbus2: covered by the protocol tests
    from armpi.backends.i2c import I2CBackend

    class FakeBus:
        def __init__(self):
            self.msgs = []

        def i2c_rdwr(self, *msgs):
            self.msgs += [(m.addr, list(m)) for m in msgs]

    bus = FakeBus()
    I2CBackend(bus).set_pulses({1: 1376}, 300)
    assert bus.msgs == [(0x7A, [40, 1, 44, 1, 1, 96, 5])]



def test_clamp_respects_board_pulse_range():
    # wrist limit -90 with trim -39 would be 461us, below the board's 500us floor
    a = CAL.clamp("wrist", -90)
    assert protocol.PULSE_MIN <= CAL.to_pulse("wrist", a) <= protocol.PULSE_MAX
    arm = Arm(SimBackend(), CAL)
    arm.move({"wrist": -90}, 2000)  # must not raise

if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
