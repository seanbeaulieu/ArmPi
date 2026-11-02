from .base import Backend
from .sim import SimBackend


def make_backend(name):
    """'i2c' for the real arm, 'sim' for the simulator."""
    if name == "i2c":
        from .i2c import I2CBackend
        return I2CBackend()
    if name == "sim":
        return SimBackend()
    raise ValueError(f"unknown backend {name!r}")
