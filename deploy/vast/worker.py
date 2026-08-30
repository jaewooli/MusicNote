"""PyWorker for the MusicNote MT3 endpoint on vast.ai serverless.

vast.ai's serverless does not run your model server directly. A PyWorker sits
in front of it: it watches the model server's log and health endpoint to decide
when the worker is ready, forwards requests, and reports load to the autoscaler
so it can scale to zero and back.

So `backend/mt3_worker.py` is the model server here, unchanged apart from
reading MT3_DEVICE=cuda and accepting base64 audio (a remote worker cannot see
the caller's filesystem). The same file still serves the local pm2 worker.
"""
from __future__ import annotations

import base64
import io
import os

from vastai import (
    BenchmarkConfig,
    HandlerConfig,
    LogActionConfig,
    Worker,
    WorkerConfig,
)

MODEL_PORT = int(os.environ.get("MT3_PORT", "8732"))
MODEL_LOG = os.environ.get("MT3_LOG_FILE", "/var/log/mt3/server.log")
BENCH_SECONDS = float(os.environ.get("MT3_BENCH_SECONDS", "6"))


def _audio_seconds(payload: dict) -> float:
    """Work in this request, in seconds of audio.

    MT3 runtime is close to linear in duration, so audio seconds is the honest
    unit for the autoscaler. The client sends it; the encoded byte count would
    be a poor proxy because FLAC is lossless and compresses by content.
    """
    return max(1.0, float(payload.get("audio_seconds") or 30.0))


def _benchmark_payload() -> dict:
    """A short real clip so the benchmark measures inference, not decoding."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, BENCH_SECONDS, int(sr * BENCH_SECONDS), endpoint=False)
    # A few sustained pitches: silence would let the model emit nothing and
    # make the benchmark far cheaper than a real request.
    y = sum(0.2 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    buf = io.BytesIO()
    sf.write(buf, y.astype("float32"), sr, format="FLAC")
    return {
        "audio_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "audio_ext": ".flac",
        "audio_seconds": BENCH_SECONDS,
    }


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=MODEL_PORT,
    model_log_file=MODEL_LOG,
    model_healthcheck_url="/health",
    # One inference at a time. YourMT3 peaks multiple GB and a second
    # concurrent request multiplies that; the model server enforces this too.
    max_sessions=1,
    handlers=[
        HandlerConfig(
            route="/transcribe",
            allow_parallel_requests=False,
            workload_calculator=_audio_seconds,
            benchmark_config=BenchmarkConfig(
                generator=_benchmark_payload,
                runs=3,
                concurrency=1,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["mt3_worker: listening"],
        on_error=["Traceback", "CUDA out of memory", "OSError"],
        on_info=["Downloading", "load_model"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
