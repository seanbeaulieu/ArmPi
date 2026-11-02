#!/bin/bash
# Sync this repo to the Pi and (re)install it there. Run from Git Bash on the PC:
#   tools/deploy.sh                 # sync + install
#   tools/deploy.sh --test          # sync + install + run the tests on the Pi
# Override the target with PI=user@host. Copies the files git tracks (plus uncommitted
# edits to them), so the Pi gets exactly your working tree, no caches or venvs.
set -e
PI=${PI:-seanbeaulieu_armpi@armpi.local}
cd "$(dirname "$0")/.."
git ls-files -co --exclude-standard | tar -cf - -T - | ssh "$PI" '
  set -e
  mkdir -p ~/ArmPi && tar -xf - -C ~/ArmPi && cd ~/ArmPi
  # venv sees the apt packages (smbus2, yaml, numpy, opencv) so nothing is downloaded
  [ -d .venv ] || python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q --no-build-isolation --no-deps -e .
  echo "deployed to $(hostname):~/ArmPi"'
if [ "$1" = --test ]; then
  ssh "$PI" 'cd ~/ArmPi && .venv/bin/python tests/test_armpi.py'
fi
