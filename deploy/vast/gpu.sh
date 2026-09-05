#!/usr/bin/env bash
# Rent, point at, and release a GPU MT3 worker.
#
#   deploy/vast/gpu.sh up       rent a box, wait for /health, switch the app to it
#   deploy/vast/gpu.sh down     destroy the box, switch the app back to local CPU
#   deploy/vast/gpu.sh status   what is running, and what the app is pointed at
#   gpu.sh stop-after 3h        release it automatically after that long
#                               (MT3_STOP_AFTER=3h gpu.sh up does it at launch)
#
# The app reads deploy/vast/current-url per request (mt3_bridge.remote_url), so
# switching costs a file write and no restart. That matters: JOBS lives only in
# memory, and restarting the app to pick up an environment variable threw away
# every result the user had open. No URL file means the local CPU worker, so a
# failed `up` leaves a working system.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VAST="${VASTAI:-$ROOT/.venv/bin/vastai}"
URL_FILE="$ROOT/deploy/vast/current-url"
DEADLINE="$ROOT/deploy/vast/stop-at"
ID_FILE="$ROOT/deploy/vast/current-instance"
IMAGE="${MT3_IMAGE:-ddyoru1015/musicnote-mt3:3}"
DISK="${MT3_DISK:-20}"

# vast forces image_runtype=ssh unless --args is used, and its ssh injection
# fails on this minimal image (repeated "remote port forwarding failed", and the
# image CMD never runs). --onstart-cmd bash + --args runs the container as-is.
#
# The pip line repairs what the image is missing. It belongs in the image, and
# the Dockerfile now has it, but rebuilding costs a ten-minute qemu cross-build
# and a 3.6 GB push; doing it here costs about 20 s of cold start. Drop it once
# the image is rebuilt on an x86_64 host.
# mt3_pytorch fetches its checkpoint with git-lfs, which the minimal image does
# not carry — it answers 500 with "Git or Git LFS not found". The Dockerfile
# avoids apt because gpgv fails under qemu when cross-building from arm64, but
# that is a BUILD-host problem: apt works fine at runtime on the x86 worker.
# Off by default because it costs about 30 s of cold start and only the
# cross-model experiment needs it.
APT=''
[ -n "${MT3_WITH_GIT:-}" ] && APT='apt-get update -qq && apt-get install -y -qq git git-lfs >/dev/null 2>&1 && git lfs install >/dev/null 2>&1;'

FIXUP='pip install --force-reinstall --no-deps --no-cache-dir \
  nvidia-cuda-cupti-cu12==12.4.127 transformers==4.44.2 tokenizers==0.19.1 \
  huggingface_hub==0.36.2 safetensors==0.8.0'

# The worker is also baked into the image, but fetching it here is what keeps a
# worker change from needing a rebuild: a qemu cross-build of this image takes
# ten minutes and a 3.6 GB push, for one file. Committing and running `up` again
# is the whole update path instead.
#
# So the GPU runs the COMMITTED worker, not the working tree. Commit first.
# On any failure the baked copy stays in place, which is why this is `|| true`
# and writes through a temp file — a half-downloaded worker is worse than an old
# one. Point MT3_WORKER_URL at another ref to test a branch.
WORKER_URL="${MT3_WORKER_URL:-https://raw.githubusercontent.com/jaewooli/MusicNote/main/backend/mt3_worker.py}"
FETCH="curl -fsSL --retry 2 -o /tmp/mt3_worker.py '$WORKER_URL' \
  && python -c 'import ast,sys; ast.parse(open(\"/tmp/mt3_worker.py\").read())' \
  && mv /tmp/mt3_worker.py /opt/musicnote/mt3_worker.py \
  && echo 'worker: using the committed copy' \
  || echo 'worker: fetch failed, using the copy baked into the image'"

# Storage is billed while an instance merely exists, and at 20 GB that outruns
# the GPU rent within days. Prefer the cheapest storage among cheap GPUs.
QUERY='num_gpus=1 compute_cap>=750 compute_cap<=900 gpu_ram>=12 inet_down>=400
       disk_space>=40 reliability>0.97 rentable=true'

pick_offer() {
  "$VAST" search offers "$QUERY" -o 'dph+' --raw 2>/dev/null \
    | "$ROOT/.venv/bin/python" "$ROOT/deploy/vast/pick_offer.py"
}

instance_url() {
  "$VAST" show instance "$1" --raw 2>/dev/null \
    | "$ROOT/.venv/bin/python" "$ROOT/deploy/vast/instance_url.py"
}

# Storage is billed for as long as the instance exists, so a box left running
# overnight costs more than the day's work did. autostop.sh reads this.
# Accepts 90m / 3h / 2d, or anything GNU date already understands. `date -d
# "+3h"` is NOT one of those — it wants a unit word — which is worth spelling
# out, because the silent failure mode is a timer that was never armed.
arm_deadline() {
  local spec="$1" until
  case "$spec" in
    *[0-9]s) spec="${spec%s} seconds" ;;
    *[0-9]m) spec="${spec%m} minutes" ;;
    *[0-9]h) spec="${spec%h} hours" ;;
    *[0-9]d) spec="${spec%d} days" ;;
  esac
  until=$(date -d "now + $spec" +%s 2>/dev/null) || {
    echo "bad duration: $1 (try 90m, 3h, 2d)" >&2; return 1; }
  [ "$until" -gt "$(date +%s)" ] || { echo "duration must be in the future" >&2; return 1; }
  echo "$until" > "$DEADLINE"
  echo "will release the GPU at $(date -d "@$until" '+%Y-%m-%d %H:%M %Z')"
}

case "${1:-status}" in
stop-after)
  [ -f "$ID_FILE" ] || { echo "nothing is running"; exit 1; }
  arm_deadline "${2:?usage: gpu.sh stop-after 3h}"
  ;;
up)
  [ -f "$ID_FILE" ] && { echo "already up: instance $(cat "$ID_FILE"). down first."; exit 1; }
  offer="$(pick_offer)"
  id="$("$VAST" create instance "$offer" --image "$IMAGE" --disk "$DISK" \
        --env '-p 8732:8732' --onstart-cmd 'bash' --raw \
        --args -c "$APT $FIXUP; $FETCH; exec python /opt/musicnote/mt3_worker.py" \
        | "$ROOT/.venv/bin/python" -c 'import sys,json; print(json.load(sys.stdin)["new_contract"])')"
  echo "$id" > "$ID_FILE"
  echo "instance $id — waiting for the worker (image pull + pip, about 2-4 min)"
  for _ in $(seq 1 30); do
    url="$(instance_url "$id")"
    if [ -n "$url" ] && curl -sf -m 8 "$url/health" >/dev/null 2>&1; then
      echo "$url" > "$URL_FILE"
      echo "up: $url"
      curl -s -m 8 "$url/health"; echo
      echo "app switched to the GPU worker (no restart needed)"
      [ -n "${MT3_STOP_AFTER:-}" ] && arm_deadline "$MT3_STOP_AFTER"
      exit 0
    fi
    sleep 20
  done
  echo "worker never answered. logs:" >&2
  "$VAST" logs "$id" 2>&1 | tail -15 >&2
  echo "instance $id left running for inspection; 'gpu.sh down' to release it" >&2
  exit 1
  ;;
down)
  [ -f "$ID_FILE" ] || { echo "nothing to release"; exit 0; }
  id="$(cat "$ID_FILE")"
  yes | "$VAST" destroy instance "$id" >/dev/null 2>&1 || true
  rm -f "$ID_FILE" "$URL_FILE" "$DEADLINE"
  echo "app switched back to the local CPU worker (no restart needed)"
  echo "destroyed $id"
  ;;
status)
  if [ -f "$DEADLINE" ]; then
    echo "auto-release at: $(date -d "@$(cat "$DEADLINE")" '+%Y-%m-%d %H:%M %Z')"
  fi
  if [ -f "$URL_FILE" ]; then
    url="$(cat "$URL_FILE")"
    echo "app points at: $url"
    echo -n "health: "; curl -s -m 8 "$url/health" || echo "(no answer)"
    echo
  else
    echo "app points at: local CPU worker"
  fi
  "$VAST" show instances 2>&1 | tail -5
  ;;
*)
  sed -n '2,9p' "$0"; exit 1;;
esac
