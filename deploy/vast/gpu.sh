#!/usr/bin/env bash
# Rent, point at, and release a GPU MT3 worker.
#
#   deploy/vast/gpu.sh up       rent a box, wait for /health, switch the app to it
#   deploy/vast/gpu.sh down     destroy the box, switch the app back to local CPU
#   deploy/vast/gpu.sh status   what is running, and what the app is pointed at
#
# The app reads deploy/vast/current-url through ecosystem.config.js. No URL file
# means the local CPU worker, so a failed `up` leaves a working system.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VAST="${VASTAI:-$ROOT/.venv/bin/vastai}"
URL_FILE="$ROOT/deploy/vast/current-url"
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

# ecosystem.config.js reads the URL file at require() time, and neither
# `pm2 restart` nor `--update-env` re-evaluates it — both replay the environment
# pm2 captured when the process was first started. Only a delete/start does.
reload_app() {
  pm2 delete musicnote >/dev/null 2>&1 || true
  pm2 start "$ROOT/ecosystem.config.js" --only musicnote >/dev/null 2>&1
}

pick_offer() {
  "$VAST" search offers "$QUERY" -o 'dph+' --raw 2>/dev/null | "$ROOT/.venv/bin/python" -c '
import sys, json
offers = [o for o in json.load(sys.stdin) if o.get("storage_cost", 9) < 0.25]
if not offers:
    sys.exit("no offer with cheap storage")
o = offers[0]
print(f"{o[\"gpu_name\"]} ${o[\"dph_total\"]:.4f}/hr storage ${o[\"storage_cost\"]:.3f}/GB/mo", file=sys.stderr)
print(o["id"])'
}

instance_url() {
  "$VAST" show instance "$1" --raw 2>/dev/null | "$ROOT/.venv/bin/python" -c '
import sys, json
d = json.load(sys.stdin)
port = ((d.get("ports") or {}).get("8732/tcp") or [{}])[0].get("HostPort")
print(f"http://{d[\"public_ipaddr\"]}:{port}" if port and d.get("actual_status") == "running" else "")'
}

case "${1:-status}" in
up)
  [ -f "$ID_FILE" ] && { echo "already up: instance $(cat "$ID_FILE"). down first."; exit 1; }
  offer="$(pick_offer)"
  id="$("$VAST" create instance "$offer" --image "$IMAGE" --disk "$DISK" \
        --env '-p 8732:8732' --onstart-cmd 'bash' --raw \
        --args -c "$FIXUP; $FETCH; exec python /opt/musicnote/mt3_worker.py" \
        | "$ROOT/.venv/bin/python" -c 'import sys,json; print(json.load(sys.stdin)["new_contract"])')"
  echo "$id" > "$ID_FILE"
  echo "instance $id — waiting for the worker (image pull + pip, about 2-4 min)"
  for _ in $(seq 1 30); do
    url="$(instance_url "$id")"
    if [ -n "$url" ] && curl -sf -m 8 "$url/health" >/dev/null 2>&1; then
      echo "$url" > "$URL_FILE"
      echo "up: $url"
      curl -s -m 8 "$url/health"; echo
      reload_app && echo "app switched to the GPU worker"
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
  rm -f "$ID_FILE" "$URL_FILE"
  reload_app && echo "app switched back to the local CPU worker"
  echo "destroyed $id"
  ;;
status)
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
