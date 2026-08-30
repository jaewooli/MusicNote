# MusicNote MT3 worker for vast.ai serverless.
#
# vast's stock serverless templates cover vLLM and ComfyUI; MT3 is neither, so
# this image carries the model server and its checkpoint, and the PyWorker runs
# in front of it. Checkpoints are baked in on purpose: a scale-to-zero worker
# would otherwise re-download them on every cold start.
# --platform is pinned because vast.ai GPU workers are x86_64 while a build host
# may not be (this repo's server is arm64 Ampere). Without it, docker silently
# builds for the host arch and the image fails to start on a worker.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MT3_PORT=8732 \
    MT3_DEVICE=cuda \
    MT3_THREADS=8 \
    MT3_LOG_FILE=/var/log/mt3/server.log \
    MT3_CHECKPOINT_DIR=/opt/mt3-checkpoints \
    MT3_MODEL=yourmt3 \
    MT3_IDLE_UNLOAD=0

WORKDIR /opt/musicnote

# No apt layer on purpose. Two reasons:
#   * It is not needed. soundfile's wheel bundles libsndfile, and the worker
#     only ever reads WAV/FLAC and resamples, so ffmpeg never gets used.
#   * apt's gpgv fails under qemu when this image is cross-built from arm64
#     ("keyring ... has an unsupported filetype"), which broke the build here.
# Verified in the target image: soundfile FLAC round-trip and librosa.resample
# both work with pip packages alone.
RUN pip install --no-cache-dir --root-user-action=ignore \
        mt3-infer pretty_midi soundfile librosa numpy

# PyWorker dependencies.
COPY deploy/vast/requirements.txt /opt/musicnote/pyworker-requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore \
        -r /opt/musicnote/pyworker-requirements.txt

# Bake the checkpoints in. A scale-to-zero worker is a fresh container on every
# cold start, so an un-baked checkpoint is re-downloaded every single time.
#
# These are copied from the machine that produced the CPU baseline rather than
# re-downloaded, so the GPU worker runs byte-identical weights and the note F1
# comparison in deploy/VAST_SERVERLESS.md actually means something.
# deploy/vast/build.sh stages this directory; it is not in git.
COPY deploy/vast/checkpoints/ /opt/mt3-checkpoints/

COPY backend/mt3_worker.py /opt/musicnote/mt3_worker.py
COPY deploy/vast/worker.py /opt/musicnote/worker.py
COPY deploy/vast/start.sh /opt/musicnote/start.sh
RUN chmod +x /opt/musicnote/start.sh && mkdir -p /var/log/mt3

CMD ["/opt/musicnote/start.sh"]
