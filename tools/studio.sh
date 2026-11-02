#!/bin/bash
# One command for the PC (WSL or Git Bash): Arm Studio with MuJoCo sim + the real arm.
#   tools/studio.sh            -> http://localhost:8001/studio   (Ctrl+C stops it)
# First run creates .venv and installs armpi + mujoco. The real arm is reached through the
# Pi's always-on server (set up once with: tools/deploy.sh --serve). Override the Pi with
# ARMPI_HOST=ip-or-name (WSL can't resolve armpi.local, so it falls back to the last known IP).
set -e
cd "$(dirname "$0")/.."
PORT=${PORT:-8001}
HOST=${ARMPI_HOST:-armpi.local}
if ! getent hosts "$HOST" >/dev/null 2>&1 && [ -z "$ARMPI_HOST" ]; then HOST=10.0.0.133; fi

if [ ! -d .venv ]; then
  echo "first run: creating .venv (needs python3-venv: sudo apt install -y python3-venv)"
  python3 -m venv .venv
fi
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe   # Git Bash on Windows
"$PY" -c "import armpi, mujoco" 2>/dev/null || "$PY" -m pip install -q -e . mujoco

REAL=""
if curl -s -m 2 "http://$HOST:8000/api/battery" >/dev/null 2>&1; then
  REAL="--real-url http://$HOST:8000"
  echo "pi server found at $HOST: 'Real arm' drives it"
else
  echo "pi server not reachable at $HOST:8000, sim only (start it: tools/deploy.sh --serve)"
fi
echo "open http://localhost:$PORT/studio   (Ctrl+C to stop)"
exec "$PY" -m armpi.server --sim mujoco --port "$PORT" $REAL
