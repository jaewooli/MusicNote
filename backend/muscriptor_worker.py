#!/usr/bin/env python3
"""
MuScriptor transcription worker — runs in its OWN venv (~/muscriptor-venv,
PyTorch-only) as a pm2 app, so its dependency tree (a different torch than
mt3-infer needs) never touches the main MusicNote venv or mt3_worker's.

  GET  /health           -> {ok, model, loaded, device}
  POST /transcribe       body {"model": "small"|"medium"|"large"} plus ONE
                         audio source: "wav_path" (local file, same-host
                         worker) or "audio_b64" (base64 of an audio file,
                         remote worker)
                         -> {ok, notes:[{start,end,pitch,velocity,program,
                             is_drum,track,instrument}], model, seconds}

Notes carry an ``instrument`` field (MuScriptor's own timbre label, e.g.
"electric_bass", "distorted_electric_guitar") alongside a GM ``program`` number
(the group's representative program, via get_group_program_map) so downstream
code that only understands GM programs — voices.py, musicxml.py, mt3_bridge's
family table — keeps working unchanged, while app.py's timbre rescue can key
on the finer label MT3's own vocabulary collapses away.

MuScriptor never estimates velocity (see muscriptor.events.NoteStartEvent) —
every note comes back at a flat 100, same as MT3's own constant velocity, so
downstream code that reads real loudness from the audio's CQT (app._mt3_dynamics)
already ignores it.

Env:
  MUSCRIPTOR_MODEL        default size: small|medium|large   (default "small")
  MUSCRIPTOR_DEVICE       torch device: cpu|cuda|auto        (default "cpu")
  MUSCRIPTOR_PORT         listen port                        (default 8733)
  MUSCRIPTOR_HOST         listen address                     (default 127.0.0.1)
  MUSCRIPTOR_IDLE_UNLOAD  seconds idle before unload; 0=never (default 600)
  HF_TOKEN                HuggingFace read token (weights are gated, CC BY-NC)
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get("MUSCRIPTOR_MODEL", "small")
PORT = int(os.environ.get("MUSCRIPTOR_PORT", "8733"))
HOST = os.environ.get("MUSCRIPTOR_HOST", "127.0.0.1")
DEVICE = os.environ.get("MUSCRIPTOR_DEVICE", "cpu")
IDLE_UNLOAD = int(os.environ.get("MUSCRIPTOR_IDLE_UNLOAD", "600"))
MAX_AUDIO_BYTES = 60 * 1024 * 1024

_model = None
_model_name: str | None = None
_group_program: dict[str, int] | None = None
_last_use = 0.0
_lock = threading.Lock()
_inference_lock = threading.Lock()


def _get_group_program_map() -> dict[str, int]:
    """MuScriptor timbre label -> representative GM program (first in its
    group's list). Built once from the package's own MT3_FULL_PLUS vocabulary
    so this stays in sync with whatever the installed version ships."""
    global _group_program
    if _group_program is None:
        from muscriptor.tokenizer.mt3 import (
            MT3_FULL_PLUS_GROUP_NAMES, get_group_program_map)
        idx_to_progs = get_group_program_map(
            "MT3_FULL_PLUS", misc_programs="OMIT", is_mt3=True, include_drums=True)
        _group_program = {
            name: idx_to_progs[idx][0]
            for name, idx in MT3_FULL_PLUS_GROUP_NAMES.items()
            if idx in idx_to_progs and idx_to_progs[idx]
        }
    return _group_program


def _get_model(name: str):
    global _model, _model_name, _last_use
    with _lock:
        if _model is not None and _model_name != name:
            _model = None
            _model_name = None
        if _model is None:
            from muscriptor import TranscriptionModel
            _model = TranscriptionModel.load_model(weights_path=name, device=DEVICE)
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
                gc.collect()


def transcribe(wav_path: str, model_name: str) -> dict:
    """Transcribe one file. Returns the same note schema mt3_worker does, plus
    ``instrument`` (MuScriptor's own timbre label)."""
    model = _get_model(model_name)
    group_program = _get_group_program_map()

    t0 = time.time()
    open_by_idx: dict[int, object] = {}
    notes = []
    tracks: dict[str, dict] = {}
    dur = 0.0
    for e in model.transcribe(wav_path):
        cls = type(e).__name__
        if cls == "NoteStartEvent":
            open_by_idx[e.index] = e
        elif cls == "NoteEndEvent":
            s = e.start_event
            start, end = float(s.start_time), float(e.end_time)
            if end <= start:
                continue      # a handful of chunk-boundary zero/negative durations
            is_drum = s.instrument == "drums"
            program = 0 if is_drum else int(group_program.get(s.instrument, 0))
            key = s.instrument
            t = tracks.get(key)
            if t is None:
                t = tracks[key] = {"track": len(tracks), "program": program,
                                   "is_drum": is_drum, "instrument": key, "count": 0}
            notes.append({
                "start": round(start, 4), "end": round(end, 4),
                "pitch": int(s.pitch), "velocity": 100,
                "program": program, "is_drum": is_drum,
                "track": t["track"], "instrument": key,
            })
            t["count"] += 1
            dur = max(dur, end)
    took = time.time() - t0
    notes.sort(key=lambda n: (n["start"], n["track"], n["pitch"]))
    return {"ok": True, "notes": notes, "tracks": list(tracks.values()),
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
                             "loaded": _model_name,
                             "models": ["small", "medium", "large"]})
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
                out = transcribe(wav, name)
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


def main() -> None:
    threading.Thread(target=_maybe_unload, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"muscriptor_worker: listening on {HOST}:{PORT}  default model={MODEL_NAME} "
          f"device={DEVICE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
