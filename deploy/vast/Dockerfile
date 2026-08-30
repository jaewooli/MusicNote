# MusicNote MT3 worker for vast.ai serverless.
#
# vast's stock serverless templates cover vLLM and ComfyUI; MT3 is neither, so
# this image carries the model server and its checkpoint, and the PyWorker runs
# in front of it. Checkpoints are baked in on purpose: a scale-to-zero worker
# would otherwise re-download them on every cold start.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/musicnote

# Model server dependencies.
RUN pip install --no-cache-dir \
        mt3-infer pretty_midi soundfile librosa numpy

# PyWorker dependencies.
COPY deploy/vast/requirements.txt /opt/musicnote/pyworker-requirements.txt
RUN pip install --no-cache-dir -r /opt/musicnote/pyworker-requirements.txt

# Bake the checkpoint so a cold start does not pay for a download. If this
# fails at build time the entrypoint still works: mt3_infer downloads on first
# use, it is just slower.
RUN mkdir -p "$MT3_CHECKPOINT_DIR" \
    && python -c "from mt3_infer import load_model; load_model('${MT3_MODEL}', device='cpu')" \
    || echo "checkpoint prefetch skipped; will download on first request"

COPY backend/mt3_worker.py /opt/musicnote/mt3_worker.py
COPY deploy/vast/worker.py /opt/musicnote/worker.py
COPY deploy/vast/start.sh /opt/musicnote/start.sh
RUN chmod +x /opt/musicnote/start.sh && mkdir -p /var/log/mt3

CMD ["/opt/musicnote/start.sh"]
