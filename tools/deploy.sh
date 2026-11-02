#!/bin/bash
# Sync this repo to the Pi and (re)install it there. Run from Git Bash on the PC:
#   tools/deploy.sh                 # sync + install
#   tools/deploy.sh --test          # sync + install + run the tests on the Pi
#   tools/deploy.sh --test --serve  # ... then (re)start the web panel service on :8000
# Override the target with PI=user@host. Copies the files git tracks (plus uncommitted
# edits to them), so the Pi gets exactly your working tree, no caches or venvs.
# The Pi's armpi/calibration.yaml is kept: calibration happens on the Pi, so bring it
# back with tools/pull-calibration.sh and commit it.
set -e
PI=${PI:-seanbeaulieu_armpi@armpi.local}
cd "$(dirname "$0")/.."
git ls-files -co --exclude-standard | tar -cf - -T - | ssh "$PI" '
  set -e
  mkdir -p ~/ArmPi && cd ~/ArmPi
  CAL=armpi/calibration.yaml
  [ -f $CAL ] && cp $CAL /tmp/armpi-calibration.yaml
  tar -xf -
  [ -f /tmp/armpi-calibration.yaml ] && mv /tmp/armpi-calibration.yaml $CAL
  # venv sees the apt packages (smbus2, yaml, numpy, opencv) so nothing is downloaded
  [ -d .venv ] || python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q --no-build-isolation --no-deps -e .
  echo "deployed to $(hostname):~/ArmPi"'
for arg in "$@"; do
  case $arg in
  --test)
    ssh "$PI" 'cd ~/ArmPi && .venv/bin/python tests/test_armpi.py && .venv/bin/python tests/test_server.py' ;;
  --serve)  # (re)start the web panel as a systemd user service, http://armpi.local:8000
    ssh "$PI" 'mkdir -p ~/.config/systemd/user
      cp ~/ArmPi/tools/armpi-panel.service ~/.config/systemd/user/
      systemctl --user daemon-reload && systemctl --user enable -q armpi-panel
      systemctl --user restart armpi-panel && sleep 2 && systemctl --user is-active armpi-panel' ;;
  esac
done
