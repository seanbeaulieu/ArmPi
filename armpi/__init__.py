"""armpi: our own control stack for the Hiwonder ArmPi Mini (replaces HiwonderSDK)."""
from .arm import Arm
from .backends import make_backend
from .calibration import JOINTS, Calibration


def connect(backend="i2c", calib=None):
    """Arm on the real board ('i2c') or the simulator ('sim')."""
    return Arm(make_backend(backend), calib)


__all__ = ["Arm", "Calibration", "JOINTS", "connect", "make_backend"]
