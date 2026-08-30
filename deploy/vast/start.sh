#!/usr/bin/env bash
# Start the MT3 model server, then the PyWorker in front of it.
#
# The PyWorker decides readiness from the model server's log and /health, so
# the server must log to MT3_LOG_FILE and must be started first.
set -euo pipefail

mkdir -p "$(dirname "${MT3_LOG_FILE:-/var/log/mt3/server.log}")"

python /opt/musicnote/mt3_worker.py >>"${MT3_LOG_FILE}" 2>&1 &
MODEL_PID=$!

# If the model server dies the worker is useless; do not leave a healthy-looking
# PyWorker in front of a dead backend.
trap 'kill -TERM "$MODEL_PID" 2>/dev/null || true' EXIT
(
  while kill -0 "$MODEL_PID" 2>/dev/null; do sleep 5; done
  echo "mt3_worker exited; stopping PyWorker" >&2
  kill -TERM $$ 2>/dev/null || true
) &

exec python /opt/musicnote/worker.py
