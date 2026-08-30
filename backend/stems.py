"""
Multi-instrument stem separation (Demucs) grouped into MuseScore-style
instrument families, plus per-stem active spans.

Pipeline (see app._run_job for orchestration):
  1. separate(path) -> list of stems: {id, family, label, program, pitched,
     spans, presence, peak, wav_path (22 kHz mono)}
  2. app runs transcribe.analyze() on each pitched stem's wav and caches it,
     so the sensitivity / instrument knobs still work per stem.

Demucs htdemucs_6s sources: drums, bass, other, vocals, guitar, piano.
The "other" bucket is sub-classified (bowed strings / winds / brass / synth)
with transcribe.classify_instrument().
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = os.environ.get("MUSICNOTE_DEMUCS_MODEL", "htdemucs_6s")
STEMS_MAX_DURATION = int(os.environ.get("MUSICNOTE_STEMS_MAX_DURATION", "480"))  # 8 min
ANALYSIS_SR = 22050

# raw demucs stem -> (family key, KO label, General-MIDI program, pitched?)
_STEM_FAMILY: dict[str, tuple[str, str, int, bool]] = {
    "drums":  ("percussion", "타악 (드럼)",        0,  False),
    "bass":   ("bass",       "베이스",             33, True),
    "vocals": ("voice",      "성악",               52, True),
    "guitar": ("plucked",    "발현 현악 (기타)",    25, True),
    "piano":  ("keyboard",   "건반 (피아노)",       0,  True),
    "other":  ("other",      "기타 악기",           48, True),   # sub-classified below
}
# classify_instrument family -> (KO label, GM program) for the "other" stem
_OTHER_MAP: dict[str, tuple[str, int]] = {
    "strings": ("찰현 현악 (바이올린 등)", 48),
    "winds":   ("목관",                    73),
    "voice":   ("성악·현악",               48),
    "piano":   ("건반·기타",               0),
    "guitar":  ("발현 현악",               25),
    "drums":   ("타악·기타",               47),
    "other":   ("기타 악기 (혼합)",        48),
}

_model = None
_model_lock = threading.Lock()


def available() -> bool:
    try:
        import demucs.pretrained  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from demucs.pretrained import get_model
                torch.set_num_threads(max(1, (os.cpu_count() or 4)))
                m = get_model(MODEL_NAME)
                m.eval()
                _model = m
    return _model


def _active_spans(env: np.ndarray, times: np.ndarray, peak: float,
                  rel: float = 0.10, min_gap: float = 0.6,
                  min_len: float = 0.7) -> list[list[float]]:
    """Contiguous [start, end] ranges where the stem's RMS envelope is above
    `rel` * its own peak. Small gaps are bridged, tiny blips dropped."""
    if peak <= 1e-6 or times.size == 0:
        return []
    on = env > (peak * rel)
    spans: list[list[float]] = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            spans.append([float(times[i]), float(times[min(j, n - 1)])])
            i = j
        else:
            i += 1
    # bridge short gaps
    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [[round(a, 2), round(b, 2)] for a, b in merged if b - a >= min_len]


def separate(path: str, dest_dir: Path, job_id: str,
             progress=None) -> list[dict[str, Any]]:
    """Run Demucs, write one 22 kHz-mono wav per non-silent stem, return metadata."""
    import torch
    import soundfile as sf
    import librosa
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, convert_audio
    import transcribe as T

    model = _get_model()
    sr_model = model.samplerate

    wav = AudioFile(path).read(streams=0, samplerate=sr_model, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    if progress:
        progress(0.05, "Demucs 모델 실행 중…")
    with torch.no_grad():
        sources = apply_model(model, wav[None], split=True, overlap=0.10,
                              progress=False, num_workers=0)[0]
    sources = sources * ref.std() + ref.mean()          # de-normalise
    dur = sources.shape[-1] / sr_model

    out: list[dict[str, Any]] = []
    names = list(model.sources)
    for k, name in enumerate(names):
        if progress:
            progress(0.5 + 0.45 * (k + 1) / len(names), f"스템 정리: {name}")
        stereo = sources[k].cpu().numpy()               # (channels, samples) @ sr_model
        mono = stereo.mean(0)
        y = librosa.resample(mono, orig_sr=sr_model, target_sr=ANALYSIS_SR)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        rms = librosa.feature.rms(y=y, hop_length=1024)[0]
        env = rms
        env_peak = float(np.max(env)) if env.size else 0.0
        times = librosa.frames_to_time(np.arange(env.size), sr=ANALYSIS_SR, hop_length=1024)
        spans = _active_spans(env, times, env_peak)
        presence = round(float(np.mean(env > env_peak * 0.10)) if env_peak > 0 else 0.0, 3)

        # skip stems that are basically silent or only bleed
        span_cov = sum(b - a for a, b in spans) / max(dur, 1e-6)
        if peak < 0.02 or presence < 0.06 or span_cov < 0.05 or not spans:
            continue

        fam, label, program, pitched = _STEM_FAMILY.get(
            name, ("other", name, 48, True))
        if name == "other":
            det = T.classify_instrument(y, ANALYSIS_SR)
            lb, program = _OTHER_MAP.get(det["family"], _OTHER_MAP["other"])
            label = f"기타 파트 — {lb}"
            fam = det["family"]

        wav_path = dest_dir / f"{job_id}_{name}.wav"
        sf.write(str(wav_path), y.astype("float32"), ANALYSIS_SR)

        out.append({
            "id": name,
            "family": fam,
            "label": label,
            "program": program,
            "pitched": pitched,
            "presence": presence,
            "peak": round(peak, 3),
            "spans": spans,
            "duration": round(dur, 2),
            "wav_path": str(wav_path),
            "audio_url": f"/api/audio/{job_id}_{name}.wav",
        })

    # loudest / most-present pitched stems first; percussion last
    out.sort(key=lambda s: (not s["pitched"], -s["presence"] * s["peak"]))
    return out
