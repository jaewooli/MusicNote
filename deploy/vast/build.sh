#!/usr/bin/env bash
# Build the MusicNote MT3 worker image.
#
#   deploy/vast/build.sh docker.io/<user>/musicnote-mt3:2
#   deploy/vast/build.sh docker.io/<user>/musicnote-mt3:2 --push
#
# Env:
#   MT3_TARGET          runtime (default) | serverless   see the Dockerfile
#   MT3_BAKE_MODELS     checkpoint subdirs to bake, space separated
#                       (default "yourmt3"; mr_mt3 adds 175 MB and the GPU
#                       image pins MT3_MODEL=yourmt3, so it is left out)
#   MT3_CHECKPOINT_DIR  source checkpoints (default ~/mt3-ckpts)
#
# Checkpoints are staged from the local MT3 worker's own directory so the image
# carries the exact weights the CPU baseline was measured with.
set -euo pipefail

TAG="${1:?usage: build.sh <registry>/<name>:<tag> [--push]}"
PUSH="${2:-}"
TARGET="${MT3_TARGET:-runtime}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_SRC="${MT3_CHECKPOINT_DIR:-$HOME/mt3-ckpts}"
CKPT_DST="$ROOT/deploy/vast/checkpoints"
MODELS="${MT3_BAKE_MODELS:-yourmt3}"

if [ ! -d "$CKPT_SRC" ]; then
  echo "no checkpoints at $CKPT_SRC — set MT3_CHECKPOINT_DIR" >&2
  exit 1
fi

rm -rf "$CKPT_DST"
mkdir -p "$CKPT_DST"
trap 'rm -rf "$CKPT_DST"' EXIT
for m in $MODELS; do
  if [ ! -d "$CKPT_SRC/$m" ]; then
    echo "no checkpoint dir $CKPT_SRC/$m" >&2
    exit 1
  fi
  cp -r "$CKPT_SRC/$m" "$CKPT_DST/$m"
done
echo "staged checkpoints: $MODELS ($(du -sh "$CKPT_DST" | cut -f1))"

echo "building $TAG (linux/amd64, target=$TARGET)"
# Explicit platform: vast workers are x86_64. On an arm64 build host this needs
# qemu-user-static + binfmt-support and takes considerably longer.
docker build --platform linux/amd64 --target "$TARGET" \
  -f "$ROOT/deploy/vast/Dockerfile" -t "$TAG" "$ROOT"

if [ "$PUSH" = "--push" ]; then
  echo "pushing $TAG"
  docker push "$TAG"
fi

echo "done: $TAG"
docker image inspect "$TAG" --format '  unpacked size: {{.Size}} bytes'
