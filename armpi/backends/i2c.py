"""Real hardware: the Hiwonder expansion board over I2C."""
from .. import protocol
from .base import Backend


class I2CBackend(Backend):
    def __init__(self, bus=None, retries=1):
        """bus: an object with i2c_rdwr() (smbus2.SMBus); opened on bus 1 if not given."""
        if bus is None:
            from smbus2 import SMBus
            bus = SMBus(protocol.I2C_BUS)
        self.bus = bus
        self.retries = retries  # Hiwonder retries every transfer once; the 40kHz bus is flaky

    def _rdwr(self, *msgs):
        for attempt in range(self.retries + 1):
            try:
                self.bus.i2c_rdwr(*msgs)
                return
            except OSError:
                if attempt == self.retries:
                    raise

    def set_pulses(self, pulses, duration_ms):
        from smbus2 import i2c_msg
        self._rdwr(i2c_msg.write(protocol.I2C_ADDR, protocol.encode_servo_move(pulses, duration_ms)))

    def battery_mv(self):
        from smbus2 import i2c_msg
        self._rdwr(i2c_msg.write(protocol.I2C_ADDR, protocol.encode_battery_request()))
        read = i2c_msg.read(protocol.I2C_ADDR, 2)
        self._rdwr(read)
        return protocol.decode_battery(list(read))

    def close(self):
        self.bus.close()
