#!/usr/bin/env bash
# Release the rented GPU once its deadline passes. Driven by cron, because the
# deadline has to survive a reboot and this host has no atd and no user linger.
#
# deploy/vast/stop-at holds a unix timestamp. `gpu.sh down` removes it, so a
# manual release cancels the timer too.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEADLINE="$ROOT/deploy/vast/stop-at"
[ -f "$DEADLINE" ] || exit 0
now=$(date +%s)
until=$(cat "$DEADLINE" 2>/dev/null || echo 0)
[ "$now" -lt "$until" ] && exit 0
echo "$(date -Is) deadline passed, releasing the GPU" >> "$ROOT/logs/gpu-autostop.log"
bash "$ROOT/deploy/vast/gpu.sh" down >> "$ROOT/logs/gpu-autostop.log" 2>&1 || true
rm -f "$DEADLINE"
