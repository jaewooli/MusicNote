"""
Client for the MuScriptor transcription worker (backend/muscriptor_worker.py,
pm2 app `muscriptor-worker`, its own venv). Keeps muscriptor's dependency tree
— a different torch than mt3-infer needs — out of both the main MusicNote
process and mt3_worker's venv.

Unlike mt3_bridge, this has no remote-GPU or vast.ai path yet: MuScriptor
small runs close to real-time on this box's CPU (measured ~1-2x a clip's own
length), so there has been no need for one. Add one the same way if that
changes — the note schema and the `/health`, `/transcribe` shape already match
mt3_worker's, on purpose.

`available()` — is the worker up?
`transcribe(wav_path, model=None, timeout=...)` — returns the worker's note
list, each carrying an ``instrument`` timbre label alongside a GM ``program``.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

MUSCRIPTOR_URL = os.environ.get("MUSICNOTE_MUSCRIPTOR_URL", "http://127.0.0.1:8733").rstrip("/")
DEFAULT_MODEL = os.environ.get("MUSICNOTE_MUSCRIPTOR_MODEL") or None  # unset -> worker's own default
DEFAULT_TIMEOUT = int(os.environ.get("MUSICNOTE_MUSCRIPTOR_TIMEOUT", "1200"))

_health_cache = {"t": 0.0, "ok": False, "info": {}}


def _get(path: str, timeout: float = 4.0) -> dict:
    with urllib.request.urlopen(MUSCRIPTOR_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def available(ttl: float = 15.0) -> bool:
    now = time.time()
    if now - _health_cache["t"] < ttl:
        return _health_cache["ok"]
    ok, info = False, {}
    try:
        info = _get("/health")
        ok = bool(info.get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    _health_cache.update(t=now, ok=ok, info=info)
    return ok


def health() -> dict:
    available()
    return _health_cache["info"]


def transcribe(wav_path: str, model: str | None = None,
              timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Transcribe via the local worker. Returns {notes:[...], tracks:[...],
    model, seconds}; each note additionally carries ``instrument``."""
    payload: dict = {"wav_path": str(wav_path)}
    m = model or DEFAULT_MODEL
    if m:
        payload["model"] = m
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        MUSCRIPTOR_URL + "/transcribe", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:          # 500 -> read the JSON error body
        try:
            out = json.loads(e.read())
        except Exception:
            raise RuntimeError(f"MuScriptor worker HTTP {e.code}") from e
    except Exception as e:  # worker restart / socket close / local connection
        raise RuntimeError(
            "음색 보강 워커가 재시작되었거나 응답하지 않습니다. 잠시 후 다시 시도하세요.") from e
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "MuScriptor worker error"))
    return out
