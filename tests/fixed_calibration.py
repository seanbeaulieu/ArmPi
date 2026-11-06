"""Point armpi at a frozen copy of the seed calibration, so the tests don't depend on the
live armpi/calibration.yaml (which changes every time the real arm is calibrated)."""
from pathlib import Path

import armpi.calibration

FIXTURE = Path(__file__).with_name("calibration.yaml")
armpi.calibration.DEFAULT_PATH = FIXTURE
