"""Joint angle <-> servo pulse conversion, driven by calibration.yaml."""
from pathlib import Path

import yaml

JOINTS = ["base", "shoulder", "elbow", "wrist", "gripper"]
DEFAULT_PATH = Path(__file__).with_name("calibration.yaml")


class Calibration:
    def __init__(self, data, path=None):
        self.data = data
        self.path = path  # where save() writes; None = the packaged calibration.yaml
        self.us_per_deg = data["us_per_deg"]
        self.min_ms_per_us = data["min_ms_per_us"]
        self.first_move_ms = data["first_move_ms"]
        self.joints = data["joints"]

    @classmethod
    def load(cls, path=None):
        with open(path or DEFAULT_PATH) as f:
            return cls(yaml.safe_load(f), path)

    def save(self, path=None):
        """Write the yaml back, keeping the file's leading comment block (the conventions)."""
        path = path or self.path or DEFAULT_PATH
        header = ""
        try:
            with open(path) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        break
                    header += line
        except FileNotFoundError:
            pass
        with open(path, "w") as f:
            f.write(header)
            yaml.safe_dump(self.data, f, sort_keys=False)

    def servo(self, joint):
        return self.joints[joint]["servo"]

    def limits(self, joint):
        return tuple(self.joints[joint]["limits"])

    def rest(self):
        return {j: self.joints[j]["rest"] for j in JOINTS}

    def clamp(self, joint, angle):
        """Clamp to the joint limits AND to what the board accepts (500..2500us after trim)."""
        lo, hi = self.limits(joint)
        from .protocol import PULSE_MAX, PULSE_MIN
        a, b = sorted((self.to_angle(joint, PULSE_MIN), self.to_angle(joint, PULSE_MAX)))
        return min(max(angle, lo, a), hi, b)

    def to_pulse(self, joint, angle):
        """Angle (deg, or 0..1 for the gripper) -> pulse in us, trim included."""
        j = self.joints[joint]
        if joint == "gripper":
            us = j["closed_pulse"] + angle * (j["open_pulse"] - j["closed_pulse"])
        else:
            us = j["zero_pulse"] + j["sign"] * (angle - j["zero_angle"]) * self.us_per_deg
        return int(round(us + j["trim"]))

    def to_angle(self, joint, pulse):
        j = self.joints[joint]
        us = pulse - j["trim"]
        if joint == "gripper":
            return (us - j["closed_pulse"]) / (j["open_pulse"] - j["closed_pulse"])
        return j["zero_angle"] + (us - j["zero_pulse"]) / (j["sign"] * self.us_per_deg)
