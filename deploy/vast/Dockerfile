# MusicNote MT3 worker image for vast.ai.
#
# Two targets:
#   --target runtime     model server only. For a rented on-demand box started
#                        and stopped by hand; backend/mt3_bridge.py reaches it
#                        with MUSICNOTE_MT3_BACKEND=remote.
#   --target serverless  adds the PyWorker, which reports load to vast's
#                        autoscaler. Only a scale-to-zero endpoint needs it.
#
# The build is staged so that deletions actually shrink the image: a file
# removed in a later layer still occupies space in the layer that created it,
# so the trim runs in its own stage and only the surviving tree is copied into
# the final one.
#
# --platform is pinned because vast.ai GPU workers are x86_64 while a build host
# may not be (this repo's server is arm64 Ampere). Without it, docker silently
# builds for the host arch and the image fails to start on a worker.

# --- install ----------------------------------------------------------------
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS deps

ENV PIP_NO_CACHE_DIR=1

# No apt layer on purpose. Two reasons:
#   * It is not needed. soundfile's wheel bundles libsndfile, and the worker
#     only ever reads WAV/FLAC and resamples, so ffmpeg never gets used.
#   * apt's gpgv fails under qemu when this image is cross-built from arm64
#     ("keyring ... has an unsupported filetype"), which broke the build here.
# Verified in the target image: soundfile FLAC round-trip and librosa.resample
# both work with pip packages alone.
# Versions are pinned to the ones the CPU baseline in ACCURACY.md was measured
# with, taken from `~/mt3-venv/bin/pip freeze` on the app server. Regenerate
# them from there, not from whatever is current on PyPI.
#
# Unpinned, a build picks up head. The 2026-08-30 build did, and hit two
# failures in a row on the GPU box: a transformers new enough to have dropped
# `transformers.utils.model_parallel_utils` (which YourMT3 imports), and then a
# huggingface-hub 1.x that transformers 4.44 refuses to run against. Neither is
# reproducible on the build host, which has no GPU to import torch against.
#
# Pinning is also what makes the GPU/CPU note-F1 comparison in VAST_GPU.md mean
# anything: unpinned, the two would be running different code.
RUN pip install --no-cache-dir --root-user-action=ignore \
        mt3-infer==0.2.0 transformers==4.44.2 tokenizers==0.19.1 \
        huggingface_hub==0.36.2 safetensors==0.8.0 einops==0.8.2 \
        pretty_midi==0.2.11.post0 soundfile==0.14.0 librosa==1.0.0

# --- trim -------------------------------------------------------------------
# Measured against the first build: 7.82 GB unpacked, 4.26 GB pulled. Each line
# notes what it reclaims. Nothing here is loaded by eager GPU inference; the
# CUDA libraries that libtorch_cuda.so links against (cudnn, cublas, cufft,
# cusparse, cusolver, curand, nccl) are all left alone.
#
# CUPTI is NOT among the things that can go, even though it is only the
# profiler's dependency. `import torch` resolves it eagerly and dies with
#   ImportError: libcupti.so.12: cannot open shared object file
# on a machine that has a GPU. This cannot be caught on an arm64 build host —
# the image is amd64 and there is no CUDA there to load — so it only surfaced on
# the rented box. Anything removed here has to be proven safe by importing torch
# on an actual GPU worker, not by reading which library links against what.
FROM deps AS trim
ARG SP=/opt/conda/lib/python3.11/site-packages
RUN set -eux; \
    # 556 MB. torch.compile / inductor only, imported lazily behind try/except.
    rm -rf "$SP/triton"; \
    # 149 MB. conda's package cache, already unpacked into lib/.
    rm -rf /opt/conda/pkgs; \
    #  70 MB. Headers are build-time only.
    rm -rf "$SP/torch/include" /opt/conda/include; \
    find "$SP" \( -name '*.h' -o -name '*.hpp' \) -delete; \
    # 346 MB. Regenerated on first import, which costs a few seconds once.
    find /opt/conda -name '__pycache__' -type d -prune -exec rm -rf {} +

# --- runtime ----------------------------------------------------------------
FROM --platform=linux/amd64 ubuntu:22.04 AS runtime

ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    MT3_PORT=8732 \
    MT3_DEVICE=cuda \
    MT3_THREADS=8 \
    MT3_LOG_FILE=/var/log/mt3/server.log \
    MT3_CHECKPOINT_DIR=/opt/mt3-checkpoints \
    MT3_MODEL=yourmt3 \
    MT3_IDLE_UNLOAD=0

# The pytorch base's apt layer added ca-certificates plus the libjpeg/libpng
# *dev* packages. Only the certificates matter at runtime: torchvision's image
# codecs are dlopen'd lazily and MT3 never decodes an image. Copied rather than
# apt-installed for the qemu reason above.
COPY --from=deps /etc/ssl/certs /etc/ssl/certs
COPY --from=deps /usr/share/ca-certificates /usr/share/ca-certificates

COPY --from=trim /opt/conda /opt/conda

# Bake the checkpoint in. On a scale-to-zero worker every cold start is a fresh
# container, so an un-baked checkpoint is re-downloaded every single time; on an
# on-demand box it just saves the first-run download.
#
# deploy/vast/build.sh stages this directory (yourmt3 only by default, which
# leaves mr_mt3's 175 MB out) from the machine that produced the CPU baseline,
# so the GPU worker runs byte-identical weights and the note F1 comparison in
# deploy/VAST_GPU.md actually means something. It is not in git.
COPY deploy/vast/checkpoints/ /opt/mt3-checkpoints/

WORKDIR /opt/musicnote
COPY backend/mt3_worker.py /opt/musicnote/mt3_worker.py
RUN mkdir -p /var/log/mt3

# The worker binds to loopback by default, because the pm2 one shares a host
# with the app. Here the app is on a different machine, so it has to listen on
# the container's external interface.
#
# Set AFTER the big COPY layers on purpose. An ENV earlier in this stage
# invalidates the 3.3 GB /opt/conda layer, and re-copying that under qemu on an
# arm64 build host takes over ten minutes and re-pushes the whole image — for a
# one-line change. Measured the hard way.
#
# The worker has no authentication. A box that listens on 0.0.0.0 must have its
# 8732 reachable only from the app server.
ENV MT3_HOST=0.0.0.0

EXPOSE 8732
CMD ["python", "/opt/musicnote/mt3_worker.py"]

# --- serverless -------------------------------------------------------------
# vastai pulls borb and fontTools (a PDF stack, ~70 MB) that the model server
# has no use for, so this lands only in the serverless target.
FROM runtime AS serverless

COPY deploy/vast/requirements.txt /opt/musicnote/pyworker-requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore \
        -r /opt/musicnote/pyworker-requirements.txt

COPY deploy/vast/worker.py /opt/musicnote/worker.py
COPY deploy/vast/start.sh /opt/musicnote/start.sh
RUN chmod +x /opt/musicnote/start.sh

EXPOSE 3000 3001
CMD ["/opt/musicnote/start.sh"]
