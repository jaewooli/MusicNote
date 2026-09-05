#!/usr/bin/env python3
"""
MT3 transcription worker — runs in its OWN venv (~/mt3-venv, PyTorch-only) as a
pm2 app, so the mt3-infer dependency tree never touches the main MusicNote venv.

  GET  /health           -> {ok, model, loaded, models}
  POST /transcribe       body {"model": ..., "shift": 0.0} plus ONE audio source:
                           "wav_path"  local file (same-host worker), or
                           "audio_b64" base64 of an audio file (remote worker)
                         -> {ok, notes:[{start,end,pitch,velocity,program,is_drum,track}],
                             tracks:[{track,program,is_drum,count}], model, seconds}

The model is lazy-loaded on the first request and (optionally) released after an
idle period, because YourMT3 is multi-GB on CPU.

Env:
  MT3_MODEL           default model name                 (default "mr_mt3")
  MT3_DEVICE         torch device: cpu|cuda|auto        (default "cpu")
  MT3_PORT            listen port                        (default 8732)
  MT3_HOST            listen address                     (default 127.0.0.1)
  MT3_THREADS        torch intra-op threads             (default 2)
  MT3_IDLE_UNLOAD    seconds of idle before unload; 0=never (default 600)
  MT3_BATCH_SEGMENTS segments per inference batch; 0=the backend's own
                     (default 2 on cpu, 0 elsewhere) — this is what bounds peak
                     memory, see `_cap_batch`
  MT3_CHUNK_SEGMENTS feed the file in blocks of this many segments; 0=whole file
                     (default: only as a fallback when the batch cap did not
                     apply, and only on cpu)
  MT3_CHECKPOINT_DIR passed through to mt3-infer
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
import torch

MODEL_NAME = os.environ.get("MT3_MODEL", "mr_mt3")
PORT = int(os.environ.get("MT3_PORT", "8732"))
# Loopback by default: the pm2 worker shares a host with the app and must not be
# reachable from outside it. A rented GPU box is the opposite case — the app is
# on another machine — so the container image sets MT3_HOST=0.0.0.0. There is no
# authentication here, so anything but loopback needs the port firewalled to the
# app server.
HOST = os.environ.get("MT3_HOST", "127.0.0.1")
THREADS = int(os.environ.get("MT3_THREADS", "2"))
# "auto" lets mt3_infer pick CUDA when present. Kept at cpu by default so the
# local pm2 worker never silently changes behaviour.
DEVICE = os.environ.get("MT3_DEVICE", "cpu")
IDLE_UNLOAD = float(os.environ.get("MT3_IDLE_UNLOAD", "600"))
TARGET_SR = 16000
END_PADDING_SECONDS = float(os.environ.get("MT3_END_PADDING_SECONDS", "0.75"))
# What bounds peak memory here is the INFERENCE BATCH, not the length of the
# clip. YourMT3's adapter calls `inference_file(bsz=8, ...)` with the batch
# written into the function body, so a 27 s clip and a 3 min clip peak at nearly
# the same place — and on this 11 GB box that place was 6.3 GB, which the kernel
# eventually picked the worker for. `_cap_batch` shrinks that batch.
#
# Feeding the file in blocks of whole segments (CHUNK_SEGMENTS) bounds the peak
# the same way and keeps every segment boundary where it was, but notes are tied
# back together only within one decode call, so a note held across a block
# boundary comes out as two. It is therefore the fallback, used when the batch
# cap could not be applied.
#
# Both are off on a GPU: it has the memory for the full batch, and that batch is
# what its speed comes from. MT3_CHUNK_SEGMENTS / MT3_BATCH_SEGMENTS override.
SEGMENT_SECONDS = 2.048
_chunk_env = os.environ.get("MT3_CHUNK_SEGMENTS")
CHUNK_SEGMENTS = int(_chunk_env) if _chunk_env not in (None, "") else None
MT3_BATCH = int(os.environ.get("MT3_BATCH_SEGMENTS", "2" if DEVICE == "cpu" else "0"))
# Upload ceiling for a remote worker. 16 kHz mono is ~32 kB/s, so this is about
# 25 minutes of audio — well past the app's own duration cap.
MAX_AUDIO_BYTES = int(os.environ.get("MT3_MAX_AUDIO_BYTES", str(48 * 1024 * 1024)))

torch.set_num_threads(max(1, THREADS))

_lock = threading.Lock()
# The HTTP server is threaded, but a single MT3 model instance is not a
# concurrent inference service. Parallel requests multiply its multi-GB peak
# memory and let pm2 kill the worker mid-response. Keep one real inference at
# a time even if the main web process is restarted.
_inference_lock = threading.Lock()
_model = None
_model_name = None
_last_use = 0.0
# Whether the batch cap actually took. If it did not — a backend that batches
# somewhere else, or an upstream rename — chunking is still there to bound the
# peak, so the worker degrades to the older, slightly lossier protection instead
# of running with none.
_batch_capped = False


def _shim_t5_kwargs() -> bool:
    """Let the mt3_pytorch / mr_mt3 backends run on transformers 4.44.

    Those backends vendor a T5 stack written against transformers >= 4.45: it
    calls each block with `past_key_values=` and `cache_position=`, and unpacks
    the result as `(hidden_states, self_attn_position_bias, ...)`. On 4.44 —
    which yourmt3 pins, so it is what is installed — the block takes
    `past_key_value` (singular), has no `cache_position`, and returns
    `(hidden_states, present_key_value_state, position_bias, ...)` when caching.

    Only the first mismatch raises ("unexpected keyword argument
    'cache_position'"). Translating the keywords alone would fix the crash and
    leave the SECOND mismatch silent: index 1 would be the cache, and the
    vendored code would feed it onward as the position bias. So the shim drops
    the cache entry from the tuple as well, which is what the caller already
    assumes — it sets `present_key_value_state = None` and passes its own list.

    Returns True if the shim was needed and applied.
    """
    try:
        import inspect
        from transformers.models.t5 import modeling_t5
        block = modeling_t5.T5Block
        params = inspect.signature(block.forward).parameters
        if "cache_position" in params or getattr(block.forward, "_shimmed", False):
            return False                     # new enough, or already done
        inner = block.forward

        def forward(self, *a, **kw):
            kw.pop("cache_position", None)
            if "past_key_values" in kw:
                kw["past_key_value"] = kw.pop("past_key_values")
            out = inner(self, *a, **kw)
            if kw.get("use_cache") and len(out) > 1:
                out = (out[0],) + tuple(out[2:])     # drop the cache entry
            return out

        forward._shimmed = True
        block.forward = forward
        return True
    except Exception as e:  # noqa: BLE001
        print(f"t5 compatibility shim not applied: {e}", flush=True)
        return False


def _cap_batch(model) -> None:
    """Make the backend infer in smaller batches.

    YourMT3's adapter calls `inference_file(bsz=8, ...)` with the batch size
    written into the function body, and that batch — not the length of the clip
    — is what sets peak memory. Measured here, one 27 s clip peaked at 6.3 GB
    even with the file already being fed in 6-segment blocks, because 6 segments
    is barely under the 8 the adapter was going to use anyway.

    Wrapping the inner call is deliberate: shrinking the batch keeps every
    segment boundary AND keeps the whole file in one decode, so notes tied
    across segments still merge. Cutting the file into blocks bounds memory the
    same way but adds a boundary the tie merge cannot cross.
    """
    global _batch_capped
    _batch_capped = False
    if MT3_BATCH <= 0:
        return
    inner = getattr(getattr(model, "model", None), "inference_file", None)
    if inner is None:
        return          # a backend that batches differently: leave it alone
    if getattr(inner, "_capped", False):
        _batch_capped = True
        return

    def capped(*a, **kw):
        kw["bsz"] = MT3_BATCH
        return inner(*a, **kw)

    capped._capped = True
    try:
        model.model.inference_file = capped
    except Exception:
        return
    _batch_capped = True


def _get_model(name: str):
    global _model, _model_name, _last_use
    with _lock:
        if _model is not None and _model_name != name:
            _model = None
            _model_name = None
        if _model is None:
            from mt3_infer import load_model
            if name != "yourmt3" and _shim_t5_kwargs():
                print(f"t5 kwargs shimmed for {name} on transformers 4.44",
                      flush=True)
            _model = load_model(name, device=DEVICE)
            _cap_batch(_model)
            _model_name = name
        _last_use = time.time()
        return _model


def _trim_heap() -> None:
    """Return freed arenas to the kernel.

    `gc.collect()` drops the Python objects, but glibc keeps the arenas they
    lived in, so RSS never falls back: measured here, an idle worker went
    1298 MB -> 1981 MB after a single 27 s clip and stayed there. Several
    requests later the ratcheted baseline plus one ~5 GB inference peak is past
    what this box has, and the kernel picks the worker. malloc_trim gives the
    pages back. It is glibc-only; anywhere else this is a no-op.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _maybe_unload():
    global _model, _model_name
    if IDLE_UNLOAD <= 0:
        return
    while True:
        time.sleep(30)
        with _lock:
            if _model is not None and time.time() - _last_use > IDLE_UNLOAD:
                _model = None
                _model_name = None
                import gc
                gc.collect()
                _trim_heap()


def _to_pretty_midi(result):
    """mt3-infer backends return either a pretty_midi.PrettyMIDI or a
    mido.MidiFile (YourMT3). Normalise to pretty_midi."""
    import pretty_midi
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, pretty_midi.PrettyMIDI):
        return result
    # mido.MidiFile -> write to a temp file, reload with pretty_midi. The file
    # has to be closed before pretty_midi reopens it, and deleted afterwards:
    # this runs once per chunk, so leaving it behind litters /tmp with a MIDI
    # per chunk per request.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        try:
            result.save(path)
        except AttributeError:
            result.write(path)
        return pretty_midi.PrettyMIDI(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _chunks(y: np.ndarray, sr: int, fallback: bool = False):
    """Split the signal into blocks of whole 2.048 s segments.

    Yields ``(samples, offset_seconds)``. The block length is a segment
    multiple, so every segment boundary lands where it would have with the whole
    file in one call. It still costs something the batch cap does not: notes are
    tied back together within one decode call, so a note sustained across a
    block boundary comes out as two. Capping the batch is therefore preferred,
    and this runs when ``fallback`` says the cap did not take.
    """
    n = (CHUNK_SEGMENTS if CHUNK_SEGMENTS is not None
         else (6 if fallback and DEVICE == "cpu" else 0))
    if n <= 0:
        yield y, 0.0
        return
    step = int(round(n * SEGMENT_SECONDS * sr))
    if step <= 0 or len(y) <= step:
        yield y, 0.0
        return
    for i in range(0, len(y), step):
        block = y[i:i + step]
        if block.size:
            yield block, i / float(sr)


def transcribe(wav_path: str, model_name: str, shift: float = 0.0) -> dict:
    """Transcribe one file.

    ``shift`` prepends that many seconds of silence before inference and
    subtracts it from the emitted times. YourMT3 consumes non-overlapping
    2.048 s segments (``input_frames`` = 32767 at 16 kHz), so an attack landing
    on a segment boundary can be lost. Re-running with a half-segment shift
    (1.024 s) puts those boundaries at a segment centre, and comparing the two
    runs locates real omissions far more reliably than a spectral heuristic.
    """
    y, sr = sf.read(wav_path)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    dur = len(y) / sr
    # A note beginning just before EOF has no trailing context, making MT3
    # under-detect final attacks. Silent right padding gives the encoder/decoder
    # room to close that event; emitted times are clamped back to source length.
    if END_PADDING_SECONDS > 0:
        y = np.pad(y, (0, int(round(END_PADDING_SECONDS * sr))))
    shift = max(0.0, float(shift))
    if shift > 0:
        y = np.pad(y, (int(round(shift * sr)), 0))

    model = _get_model(model_name)
    t0 = time.time()
    parts = []           # (audio_offset_seconds, PrettyMIDI)
    for y_chunk, offset in _chunks(y, sr, fallback=not _batch_capped):
        with torch.no_grad():
            raw = model.transcribe(y_chunk, sr=sr)
        parts.append((offset, _to_pretty_midi(raw)))
        # The backend holds its prediction batches until something drops them;
        # without this the peak of chunk N is added to the peak of chunk N-1 and
        # a long file walks straight into the same OOM chunking was meant to fix.
        del raw
        gc.collect()
        _trim_heap()
    took = time.time() - t0
    global _last_use
    _last_use = time.time()

    # A chunk's instrument list is its own; identify a track by what it IS
    # (program + drum flag) so the same instrument keeps one id across chunks.
    notes = []
    tracks: dict[tuple[int, bool], dict] = {}
    for offset, pm in parts:
        for inst in pm.instruments:
            key = (int(inst.program), bool(inst.is_drum))
            t = tracks.get(key)
            if t is None:
                t = tracks[key] = {"track": len(tracks), "program": key[0],
                                   "is_drum": key[1], "count": 0}
            for n in inst.notes:
                # Undo the chunk offset and the analysis-time shift before
                # clamping to the source length.
                start = float(n.start) + offset - shift
                end = min(float(n.end) + offset - shift, dur)
                if start < -0.05:
                    continue      # inside the silent lead-in: not a real event
                start = max(start, 0.0)
                if start >= dur or end <= start:
                    continue
                notes.append({
                    "start": round(start, 4),
                    "end": round(end, 4),
                    "pitch": int(n.pitch),
                    "velocity": int(max(1, min(127, n.velocity))),
                    "program": key[0],
                    "is_drum": key[1],
                    "track": t["track"],
                })
                t["count"] += 1
    tracks = list(tracks.values())
    notes.sort(key=lambda n: (n["start"], n["track"], n["pitch"]))
    return {"ok": True, "notes": notes, "tracks": tracks, "shift": round(shift, 4),
            "model": model_name, "seconds": round(took, 1), "duration": round(dur, 2)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quieter
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            # `batch_capped` is worth reporting: if the cap silently stopped
            # applying (an upstream rename, say) the worker is one long clip
            # away from being OOM-killed again, and nothing else would show it.
            self._send(200, {"ok": True, "model": MODEL_NAME, "device": DEVICE,
                             "loaded": _model_name,
                             "batch": MT3_BATCH, "batch_capped": _batch_capped,
                             "chunk_segments": CHUNK_SEGMENTS,
                             "models": ["mr_mt3", "mt3_pytorch", "yourmt3"]})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/transcribe":
            self._send(404, {"ok": False, "error": "not found"})
            return
        tmp = None
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            name = req.get("model") or MODEL_NAME
            # A remote worker has no access to the caller's filesystem, so the
            # audio travels in the request. Local runs keep using a path so a
            # same-host job does not copy megabytes through JSON.
            if req.get("audio_b64"):
                import base64
                import tempfile
                raw = base64.b64decode(req["audio_b64"])
                if len(raw) > MAX_AUDIO_BYTES:
                    self._send(413, {"ok": False, "error": "audio too large"})
                    return
                fd, tmp = tempfile.mkstemp(suffix=req.get("audio_ext", ".wav"))
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                wav = tmp
            else:
                wav = req.get("wav_path")
                if not wav or not os.path.exists(wav):
                    self._send(400, {"ok": False, "error": f"no such file: {wav}"})
                    return
            with _inference_lock:
                out = transcribe(wav, name, float(req.get("shift", 0.0)))
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-1500:]})
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


def main():
    threading.Thread(target=_maybe_unload, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"mt3_worker: listening on {HOST}:{PORT}  default model={MODEL_NAME} "
          f"device={DEVICE} threads={THREADS} idle_unload={IDLE_UNLOAD}s", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
