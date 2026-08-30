#!/usr/bin/env bash
# Build the MusicNote MT3 serverless image.
#
#   deploy/vast/build.sh docker.io/<user>/musicnote-mt3:1
#   deploy/vast/build.sh ghcr.io/<user>/musicnote-mt3:1 --push
#
# Checkpoints are staged from the local MT3 worker's own directory so the image
# carries the exact weights the CPU baseline was measured with.
set -euo pipefail

TAG="${1:?usage: build.sh <registry>/<name>:<tag> [--push]}"
PUSH="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_SRC="${MT3_CHECKPOINT_DIR:-$HOME/mt3-ckpts}"
CKPT_DST="$ROOT/deploy/vast/checkpoints"

if [ ! -d "$CKPT_SRC" ]; then
  echo "no checkpoints at $CKPT_SRC — set MT3_CHECKPOINT_DIR" >&2
  exit 1
fi

echo "staging checkpoints from $CKPT_SRC ($(du -sh "$CKPT_SRC" | cut -f1))"
rm -rf "$CKPT_DST"
mkdir -p "$CKPT_DST"
cp -r "$CKPT_SRC"/. "$CKPT_DST"/
trap 'rm -rf "$CKPT_DST"' EXIT

echo "building $TAG (linux/amd64)"
# Explicit platform: vast workers are x86_64. On an arm64 build host this needs
# qemu-user-static + binfmt-support and takes considerably longer.
docker build --platform linux/amd64 \
  -f "$ROOT/deploy/vast/Dockerfile" -t "$TAG" "$ROOT"

if [ "$PUSH" = "--push" ]; then
  echo "pushing $TAG"
  docker push "$TAG"
fi

echo "done: $TAG"
docker image inspect "$TAG" --format '  image size: {{.Size}} bytes'
