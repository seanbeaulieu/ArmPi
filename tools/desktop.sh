#!/bin/bash
# Run the arm server on the PC, driving the real arm through the Pi's server, with the die
# tasks (Move / Locate / Pick / Place, pass/fail log, die in the 3D view) at /studio:
#   tools/desktop.sh                      # http://localhost:8010/studio
#   PI=http://10.0.0.133:8000 PORT=8010 tools/desktop.sh
set -e
cd "$(dirname "$0")/.."
exec .venv/bin/python -m armpi.server --real-url "${PI:-http://10.0.0.133:8000}" --port "${PORT:-8010}"
