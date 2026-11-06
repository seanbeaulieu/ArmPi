import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fixed_calibration  # noqa: E402,F401  (before any test module loads a calibration)
