from .base import Backend
from .sim import SimBackend


def make_backend(name):
    """'i2c' for the real arm, 'sim' for the simulator, 'mujoco' for physics sim, or an armpi.server URL."""
    if name.startswith("http://") or name.startswith("https://"):
        from .remote import RemoteBackend
        return RemoteBackend(name)
    if name == "i2c":
        from .i2c import I2CBackend
        return I2CBackend()
    if name == "sim":
        return SimBackend()
    if name == "mujoco":
        from .mujoco import MujocoBackend
        return MujocoBackend(realtime=True)
    raise ValueError(f"unknown backend {name!r}")
