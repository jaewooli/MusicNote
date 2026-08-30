#!/usr/bin/env bash
# Build the isolated MT3 worker environment (~/mt3-venv) and download a model.
# PyTorch-only (mt3-infer) — kept out of the main MusicNote venv on purpose.
#
#   deploy/mt3-setup.sh                # MR-MT3 (light, ~0.7 GB RAM)
#   deploy/mt3-setup.sh yourmt3        # YourMT3 (best, ~7.5 GB RAM, needs git-lfs)
#
# Then:  pm2 start ecosystem.config.js --only mt3-worker
set -euo pipefail

MODEL="${1:-mr_mt3}"
VENV="${MT3_VENV:-$HOME/mt3-venv}"
CKPTS="${MT3_CHECKPOINT_DIR:-$HOME/mt3-ckpts}"

echo ">> creating venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo ">> installing CPU torch + mt3-infer"
"$VENV/bin/pip" install -q torch==2.4.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q mt3-infer
# transformers version is delicate: 5.x needs torch>=2.5 and has an import bug;
# >=4.45 breaks YourMT3's decoder ('NoneType' not subscriptable). 4.44.2 is the
# sweet spot where YourMT3 works. (MR-MT3 also needs the T5 kwarg patch below.)
"$VENV/bin/pip" install -q "transformers==4.44.2"

echo ">> patching vendored MR-MT3 T5 (past_key_values -> past_key_value kwarg)"
MRT5="$("$VENV/bin/python" -c 'import mt3_infer,os;print(os.path.join(os.path.dirname(mt3_infer.__file__),"models","mr_mt3","t5.py"))')"
sed -i 's/^\( *\)past_key_values=past_key_value,$/\1past_key_value=past_key_value,/' "$MRT5"

if [ "$MODEL" = "yourmt3" ] || [ "$MODEL" = "all" ]; then
  command -v git-lfs >/dev/null || { echo "git-lfs required for $MODEL — run: sudo apt install git-lfs"; exit 1; }
fi

echo ">> downloading checkpoint(s): $MODEL"
mkdir -p "$CKPTS"
MT3_CHECKPOINT_DIR="$CKPTS" "$VENV/bin/mt3-infer" download "$MODEL"

echo ">> done. start the worker with:"
echo "   pm2 start ecosystem.config.js --only mt3-worker && pm2 save"
