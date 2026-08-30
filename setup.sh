#!/usr/bin/env bash
# One-shot environment setup for MusicNote.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> system packages (ffmpeg, libsndfile)"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq ffmpeg libsndfile1
fi

echo "==> python virtualenv"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -q -r backend/requirements.txt
.venv/bin/pip install -q -U yt-dlp yt-dlp-ejs bgutil-ytdlp-pot-provider  # keep current; YouTube breaks old versions fast

# YouTube URL input also needs (not installed by this script):
#   - Deno on PATH           (JS "n" challenge runtime; ecosystem.config.js adds ~/.local/deno/bin)
#   - bgutil-pot HTTP server  (pm2 app 'bgutil-pot', Node >= 22, port 4416)
#   - backend/cookies.txt     (throwaway Google account; see README + deploy/yt_login.py)

echo "==> optional: basic-pitch (polyphonic transcription, onnxruntime backend)"
# basic-pitch pins an ancient numpy; install without its deps and add the
# runtime pieces it actually needs. Failure here is non-fatal – the app falls
# back to a CQT-based engine.
if .venv/bin/pip install -q --no-deps basic-pitch mir_eval pretty_midi \
   && .venv/bin/pip install -q onnxruntime "resampy>=0.4"; then
  if .venv/bin/python -c "from basic_pitch.inference import predict" 2>/dev/null; then
    echo "    basic-pitch OK"
  else
    echo "    basic-pitch import failed – will use CQT fallback"
  fi
else
  echo "    basic-pitch install failed – will use CQT fallback"
fi

echo "==> optional: demucs (stems mode = multi-instrument separation, ~1.5 GB)"
if [ "${MUSICNOTE_INSTALL_STEMS:-0}" = "1" ]; then
  .venv/bin/pip install -q "torch==2.4.1" "torchaudio==2.4.1" \
    --index-url https://download.pytorch.org/whl/cpu \
  && .venv/bin/pip install -q "demucs==4.0.1" \
  && .venv/bin/python -c "import demucs.pretrained" 2>/dev/null \
  && echo "    demucs OK" || echo "    demucs install failed – stems mode stays hidden"
else
  echo "    skipped (set MUSICNOTE_INSTALL_STEMS=1 to install; demucs>=4.1 needs Rust — pin 4.0.1)"
fi

echo "==> optional: MT3 multi-instrument transcription worker (isolated venv)"
echo "    build with:  deploy/mt3-setup.sh yourmt3    (then pm2 start ... --only mt3-worker)"
echo "    heavy & slow on CPU — see README '#mt3 모드'."

mkdir -p logs uploads
echo "==> done. Start with:  pm2 start ecosystem.config.js"
