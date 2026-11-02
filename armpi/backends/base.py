"""The interface every backend implements. Backends speak servo ids and raw pulses only;
angles, limits and calibration live in Arm."""


class Backend:
    def set_pulses(self, pulses, duration_ms):
        """pulses: {servo_id: pulse_us}. Start moving all of them together over duration_ms."""
        raise NotImplementedError

    def battery_mv(self):
        raise NotImplementedError

    def close(self):
        pass
