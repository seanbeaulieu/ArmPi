#!/bin/bash
# Copy the Pi's calibration.yaml (edited by the web panel's Calibrate buttons) into this repo.
set -e
PI=${PI:-seanbeaulieu_armpi@armpi.local}
cd "$(dirname "$0")/.."
scp "$PI:ArmPi/armpi/calibration.yaml" armpi/calibration.yaml
git --no-pager diff --stat armpi/calibration.yaml
