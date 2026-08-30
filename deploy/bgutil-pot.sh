#!/usr/bin/env bash
# Launches the bgutil PO-token provider HTTP server (companion to yt-dlp for
# YouTube URL input). Needs Node >= 22, which is NOT the system node here, so we
# use a private copy under ~/.local. Built once via:
#   cd ~/bgutil-ytdlp-pot-provider/server && npm ci && npx tsc
set -euo pipefail

NODE22="$HOME/.local/node-v22.14.0-linux-arm64/bin/node"
SERVER="$HOME/bgutil-ytdlp-pot-provider/server/build/main.js"
PORT="${BGUTIL_POT_PORT:-4416}"

exec "$NODE22" "$SERVER" --port "$PORT"
