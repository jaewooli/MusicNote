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
idle period, because YourMT3 peaks ~7.5 GB on CPU.

Env:
  MT3_MODEL           default model name                 (default "mr_mt3")
  MT3_DEVICE         torch device: cpu|cuda|auto        (default "cpu")
  MT3_PORT            listen port                        (default 8732)
  MT3_THREADS        torch intra-op threads             (default 2)
  MT3_IDLE_UNLOAD    seconds of idle before unload; 0=never (default 600)
  MT3_CHECKPOINT_DIR passed through to mt3-infer
"""
from __future__ import annotations

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
THREADS = int(os.environ.get("MT3_THREADS", "2"))
# "auto" lets mt3_infer pick CUDA when present. Kept at cpu by default so the
# local pm2 worker never silently changes behaviour.
DEVICE = os.environ.get("MT3_DEVICE", "cpu")
IDLE_UNLOAD = float(os.environ.get("MT3_IDLE_UNLOAD", "600"))
TARGET_SR = 16000
END_PADDING_SECONDS = float(os.environ.get("MT3_END_PADDING_SECONDS", "0.75"))
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


def _get_model(name: str):
    global _model, _model_name, _last_use
    with _lock:
        if _model is not None and _model_name != name:
            _model = None
            _model_name = None
        if _model is None:
            from mt3_infer import load_model
            _model = load_model(name, device=DEVICE)
            _model_name = name
        _last_use = time.time()
        return _model


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


def _to_pretty_midi(result):
    """mt3-infer backends return either a pretty_midi.PrettyMIDI or a
    mido.MidiFile (YourMT3). Normalise to pretty_midi."""
    import pretty_midi
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, pretty_midi.PrettyMIDI):
        return result
    # mido.MidiFile -> write to a temp file, reload with pretty_midi
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        try:
            result.save(f.name)
        except AttributeError:
            result.write(f.name)
        return pretty_midi.PrettyMIDI(f.name)


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
    with torch.no_grad():
        raw = model.transcribe(y, sr=sr)
    pm = _to_pretty_midi(raw)
    took = time.time() - t0
    global _last_use
    _last_use = time.time()

    notes = []
    tracks = []
    for ti, inst in enumerate(pm.instruments):
        prog = int(inst.program)
        drum = bool(inst.is_drum)
        cnt = 0
        for n in inst.notes:
            # Undo the analysis-time shift before clamping to the source length.
            start = float(n.start) - shift
            end = min(float(n.end) - shift, dur)
            if start < -0.05:
                continue          # inside the silent lead-in: not a real event
            start = max(start, 0.0)
            if start >= dur or end <= start:
                continue
            notes.append({
                "start": round(start, 4),
                "end": round(end, 4),
                "pitch": int(n.pitch),
                "velocity": int(max(1, min(127, n.velocity))),
                "program": prog,
                "is_drum": drum,
                "track": ti,
            })
            cnt += 1
        tracks.append({"track": ti, "program": prog, "is_drum": drum, "count": cnt})
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
            self._send(200, {"ok": True, "model": MODEL_NAME, "device": DEVICE,
                             "loaded": _model_name, "models": ["mr_mt3", "mt3_pytorch", "yourmt3"]})
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
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"mt3_worker: listening on 127.0.0.1:{PORT}  default model={MODEL_NAME} "
          f"device={DEVICE} threads={THREADS} idle_unload={IDLE_UNLOAD}s", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
