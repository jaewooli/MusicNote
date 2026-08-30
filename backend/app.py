"""
MusicNote -- 음원에서 멜로디/음을 추출하는 웹 서비스.

  GET  /                        정적 프론트엔드
  GET  /api/health              상태 확인
  POST /api/transcribe          오디오(파일 또는 YouTube URL) 제출 -> job_id (202)
  GET  /api/progress/{job_id}   진행률 폴링 (완료 시 result 포함)
  POST /api/refine/{job_id}     민감도/악기/스템만 바꿔 재-세그먼트 (재분석 없음, 실시간)
  GET  /api/download/{id}.mid   추출 결과 MIDI
  GET  /api/audio/{id}          분석에 쓴 원본/스템 오디오 (브라우저 재생용)

mode "stems": Demucs 로 악기군별 스템 분리 후 각 스템을 채보 (느림).
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import fetch
import mt3_bridge as MT3
import musicxml as MX
import quality as Q
import stems as S
import transcribe as T
import voices as VO

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
WORK_DIR = Path(os.environ.get("MUSICNOTE_WORKDIR", BASE_DIR / "uploads"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES = int(os.environ.get("MUSICNOTE_MAX_MB", "40")) * 1024 * 1024
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
RESULT_TTL = 3600  # seconds to keep generated MIDI / audio / job state
MAX_JOBS = 4       # concurrent transcription jobs

_AUDIO_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".webm": "audio/webm",
}

_STEM_MODE = {          # per-family transcription engine for stems mode
    "keyboard": "polyphonic", "plucked": "polyphonic", "other": "polyphonic",
    "bass": "melody", "voice": "melody", "strings": "melody", "winds": "melody",
}
_FAMILY_RANGE = {       # pYIN pitch range (Hz) per family — matters a lot for bass
    "bass":    (31.0, 400.0),     # B0 .. G4
    "voice":   (75.0, 1200.0),    # D2 .. D6
    "strings": (150.0, 3200.0),   # violin/viola/cello span
    "winds":   (150.0, 2500.0),
}

app = FastAPI(title="MusicNote", version="0.3.0")

# job_id -> dict(status, stage, pct, message, [result], [error], [http], created)
JOBS: dict[str, dict] = {}
_JOB_SLOTS = threading.Semaphore(MAX_JOBS)
_STEMS_SLOT = threading.Semaphore(1)   # Demucs is heavy: one at a time
# MT3 and Demucs both peak RAM hard — an MT3 job also holds _STEMS_SLOT so they
# can never run together on this box.
_MT3_SLOT = threading.Semaphore(1)


def _cleanup() -> None:
    now = time.time()
    for p in WORK_DIR.glob("*"):
        try:
            if now - p.stat().st_mtime > RESULT_TTL:
                p.unlink()
        except OSError:
            pass
    for jid, j in list(JOBS.items()):
        if now - j.get("created", now) > RESULT_TTL:
            JOBS.pop(jid, None)


def _set(jid: str, **kw) -> None:
    j = JOBS.get(jid)
    if j is not None:
        j.update(kw)


def _write_midi(job_id: str, result: dict) -> None:
    """(Re)write {job_id}.mid from result['notes']; set result['midi_url']."""
    try:
        pm = T.notes_to_midi(result["notes"], tempo=result.get("tempo") or 120.0)
        pm.write(str(WORK_DIR / f"{job_id}.mid"))
        result["midi_url"] = f"/api/download/{job_id}.mid"
    except Exception as e:  # noqa: BLE001
        result["midi_url"] = None
        result["warning"] = (result.get("warning", "") + f" (MIDI 생성 실패: {e})").strip()


def _keep_alive(job_id: str) -> None:
    """Refresh a job + its on-disk files so an active edit/refine session does
    not get reaped by _cleanup() mid-use."""
    j = JOBS.get(job_id)
    if not j:
        return
    j["created"] = time.time()
    now = time.time()
    for p in WORK_DIR.glob(f"{job_id}*"):
        try:
            os.utime(p, (now, now))
        except OSError:
            pass


def _probe_duration(path: Path) -> float:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "basic_pitch": T._has_basic_pitch(),
        "engines": ["polyphonic"],
        "url_input": True,
        "yt_cookies": fetch.cookies_available(),
        "pot_server": fetch.pot_server_up(),
        "max_duration": fetch.MAX_DURATION,
        "stems": S.available(),
        "stems_max_duration": S.STEMS_MAX_DURATION,
        "melody_engine": "crepe" if T._has_crepe() else "pyin",
        "piano_model": T._has_piano_model(),
        "mt3": MT3.available(),
        "mt3_model": MT3.health().get("model"),
    }


def _stem_midi_url(job_id: str, stem: dict, tempo: float) -> Optional[str]:
    try:
        pm = T.notes_to_midi(stem.get("notes", []), tempo=tempo or 120.0,
                             program=stem.get("program", 0), name=str(stem["id"]),
                             is_drum=not stem.get("pitched", True))
        pm.write(str(WORK_DIR / f"{job_id}_{stem['id']}.mid"))
        return f"/api/download/{job_id}_{stem['id']}.mid"
    except Exception:
        return None


def _tag_notes(notes: list, stem_id: str, label: str) -> list:
    return [{**n, "stem": stem_id, "inst": label} for n in notes]


def _merge_stems(stems_out: list[dict]) -> tuple[list, list]:
    """Merge every pitched stem's notes into one list (each note tagged with its
    stem) + a contour that follows the highest sounding pitch."""
    notes: list = []
    for s in stems_out:
        if s.get("notes"):
            notes += _tag_notes(s["notes"], s["id"], s["label"])
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    contour: list = []
    if notes:
        import numpy as _np
        end = max(n["end"] for n in notes)
        for t in _np.arange(0.0, end, 0.05):
            top = [n["pitch"] for n in notes if n["start"] <= t < n["end"]]
            if top:
                p = max(top)
                contour.append({"t": round(float(t), 3), "midi": float(p),
                                "freq": round(440.0 * 2 ** ((p - 69) / 12), 2)})
    return notes, contour


def _lead_stem(stems_out: list[dict]) -> Optional[dict]:
    """Best single 'the tune' stem: vocals if present, else score the rest by
    register + note count, de-prioritising the bass."""
    pitched = [s for s in stems_out if s.get("notes")]
    if not pitched:
        return None
    for s in pitched:
        if s["id"] == "vocals":
            return s

    def score(s: dict) -> tuple:
        ps = [n["pitch"] for n in s["notes"]]
        mean_p = sum(ps) / len(ps) if ps else 0.0
        return (s["id"] != "bass", 0.4 * s["note_count"] + mean_p)

    return max(pitched, key=score)


def _write_merged_midi(job_id: str, stems_out: list[dict], tempo: float) -> Optional[str]:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo or 120.0)
        for s in stems_out:
            if not s.get("notes"):
                continue
            one = T.notes_to_midi(s["notes"], tempo=tempo or 120.0,
                                  program=s.get("program", 0), name=str(s["id"]),
                                  is_drum=not s.get("pitched", True))
            pm.instruments.extend(one.instruments)
        if not pm.instruments:
            return None
        pm.write(str(WORK_DIR / f"{job_id}.mid"))
        return f"/api/download/{job_id}.mid"
    except Exception:
        return None


def _lead_meta_id(pitched_meta: list[dict]) -> Optional[str]:
    """Lead stem chosen from separation metadata (no notes yet)."""
    if not pitched_meta:
        return None
    for m in pitched_meta:
        if m["id"] == "vocals":
            return "vocals"
    return max(pitched_meta,
              key=lambda m: (m["id"] != "bass", m["presence"] * m["peak"]))["id"]


def _analyze_stem(job_id: str, meta: dict, mix_beats, force_melody: bool = False):
    """Run the right engine on one stem wav -> (analysis, refined-result)."""
    eng_mode = "melody" if force_melody else _STEM_MODE.get(meta["family"], "polyphonic")
    lo, hi = _FAMILY_RANGE.get(meta["family"], (None, None))
    a = T.analyze(meta["wav_path"], eng_mode, fmin=lo, fmax=hi,
                  family=meta["family"], hq=force_melody,
                  instrument_hint=meta["family"] if eng_mode == "melody" else None)
    if mix_beats:
        a["beats"] = mix_beats
    # this is a separated stem: turn on the bleed / ghost-note gate
    a["gate"] = True
    a["spans"] = meta.get("spans") or []
    return a, T.refine(a, T.DEFAULT_SENSITIVITY)


def _stems_pipeline(job_id: str, src_path: Path, audio_dur: float,
                    dl_steps: int, _phase, mode: str = "stems") -> tuple[list[dict], dict]:
    """Demucs separation + per-stem analysis. Every mode goes through here.
    For `mode="melody"` the lead stem is (re-)transcribed monophonically so the
    default view is a single clean line even if that stem is chordal-family.
    Returns (stems_out, stem_analyses); caches analyses on the job."""
    if audio_dur > S.STEMS_MAX_DURATION:
        raise fetch.TooLong(
            f"악기 분리 채보는 {S.STEMS_MAX_DURATION // 60}분 이하만 지원합니다 "
            f"(입력 {int(audio_dur // 60)}분). 더 짧은 구간을 주세요.")
    if not _STEMS_SLOT.acquire(timeout=15):
        raise RuntimeError("다른 분리 작업이 진행 중입니다. 잠시 후 다시 시도하세요.")
    try:
        _phase(dl_steps + 1, "separate", "악기 스템 분리 중 (Demucs)…",
               max(20.0, audio_dur * 3.0))

        def sep_progress(frac: float, msg: str) -> None:
            _set(job_id, pct=min(0.98, frac), message=f"악기 스템 분리 — {msg}")

        stem_meta = S.separate(str(src_path), WORK_DIR, job_id, progress=sep_progress)
        if not stem_meta:
            raise RuntimeError("분리된 악기 스템이 없습니다 (무음이거나 인식 실패).")

        _phase(dl_steps + 2, "analyze", "스템별 채보 중…",
               max(6.0, len(stem_meta) * audio_dur * 0.5))

        try:                        # song-level beat grid + tempo, shared by every stem
            import librosa
            _y, _sr = librosa.load(str(src_path), sr=22050, mono=True)
            mix_beats, mix_tempo = T._beat_grid(_y, _sr)
        except Exception:
            mix_beats, mix_tempo = [], 0.0
        JOBS.get(job_id, {})["mix_beats"] = mix_beats
        JOBS.get(job_id, {})["mix_tempo"] = mix_tempo

        stem_analyses: dict[str, dict] = {}
        stems_out: list[dict] = []
        pitched_meta = [m for m in stem_meta if m["pitched"]]
        lead_id = _lead_meta_id(pitched_meta) if mode == "melody" else None
        n = len(stem_meta)
        for k, meta in enumerate(stem_meta):
            _set(job_id, pct=min(0.98, (k + 0.1) / n),
                 message=f"스템 채보: {meta['label']}")
            stem = dict(meta)
            stem.pop("wav_path", None)
            if meta["pitched"]:
                try:
                    a, r = _analyze_stem(job_id, meta, mix_beats,
                                         force_melody=(meta["id"] == lead_id))
                    stem_analyses[meta["id"]] = a
                    stem.update(engine=r["engine"], notes=r["notes"],
                                contour=r["contour"], tempo=r["tempo"],
                                sensitivity=r["sensitivity"], instrument=r["instrument"],
                                quantized=r.get("quantized", False),
                                beat_count=r.get("beat_count", 0),
                                key=r.get("key"), low_conf=r.get("low_conf", 0))
                except Exception as e:  # noqa: BLE001
                    stem.update(engine="failed", notes=[], contour=[],
                                warning=f"채보 실패: {e}")
            else:
                stem.update(engine="none", notes=[], contour=[])
            stem["note_count"] = len(stem.get("notes", []))
            stem["midi_url"] = _stem_midi_url(job_id, stem, stem.get("tempo") or 0.0)
            stems_out.append(stem)

        JOBS.get(job_id, {})["stem_analyses"] = stem_analyses
        return stems_out, stem_analyses
    finally:
        _STEMS_SLOT.release()


def _assemble(job_id: str, mode: str, stems_out: list[dict],
              audio_dur: float) -> dict:
    """Shape the final result for the requested mode. `stems` is always present
    so the UI can drill into any instrument."""
    lead = _lead_stem(stems_out)
    _job = JOBS.get(job_id, {})
    mix_beats = _job.get("mix_beats", []) or []
    tempo = _job.get("mix_tempo") or (lead or {}).get("tempo", 0.0)
    result: dict = {
        "engine": "demucs:" + S.MODEL_NAME,
        "mode": mode,
        "duration": round(audio_dur, 3),
        "tempo": tempo,
        "stems": stems_out,
        "active_stem": None,
        "beat_count": len(mix_beats) or next(
            (s.get("beat_count", 0) for s in stems_out if s.get("beat_count")), 0),
        "beats": mix_beats,
        "quantized": False,
        "musicxml_url": None,
        "edited": False,
        "key": (lead or {}).get("key"),
        "low_conf": 0,
    }
    if mode == "polyphonic":
        notes, contour = _merge_stems(stems_out)
        result.update(notes=notes, contour=contour,
                      low_conf=sum(1 for n in notes if n.get("conf", 1.0) < 0.5),
                      midi_url=_write_merged_midi(job_id, stems_out, tempo))
    else:  # melody + stems: default view = the lead line
        result.update(
            notes=(lead or {}).get("notes", []),
            contour=(lead or {}).get("contour", []),
            midi_url=(lead or {}).get("midi_url"),
            low_conf=(lead or {}).get("low_conf", 0),
            active_stem=(lead or (stems_out[0] if stems_out else {})).get("id"),
        )
    return result


def _note_spans(notes: list[dict], gap: float = 0.6, min_len: float = 0.4) -> list[list[float]]:
    """Coarse [start,end] ranges a track is active, for the stem timeline bars."""
    if not notes:
        return []
    ns = sorted(notes, key=lambda n: n["start"])
    spans = [[ns[0]["start"], ns[0]["end"]]]
    for n in ns[1:]:
        if n["start"] - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], n["end"])
        else:
            spans.append([n["start"], n["end"]])
    return [[round(a, 2), round(b, 2)] for a, b in spans if b - a >= min_len]


def _mt3_stems(job_id: str, raw: list[dict], sensitivity: float,
               split_voices: bool = True, max_voices: int | None = None) -> list[dict]:
    """Group MT3 notes by model track and infer notation lines when needed.

    Polyphonic lines are inferred from actual overlap, pitch continuity and
    register history. They are not claimed to be separate performers.
    """
    s = max(0.0, min(1.0, sensitivity))
    amp_thr = int(round(18 * (1.0 - s)))          # velocity floor
    min_len = 0.05 + 0.06 * (1.0 - s)             # seconds
    by_track: dict[int, dict] = {}
    for n in raw:
        if n["velocity"] < amp_thr or (n["end"] - n["start"]) < min_len:
            continue
        t = n["track"]
        d = by_track.setdefault(t, {"program": n["program"], "is_drum": n["is_drum"], "notes": []})
        p = int(n["pitch"])
        d["notes"].append({
            "start": round(float(n["start"]), 3), "end": round(float(n["end"]), 3),
            "pitch": p, "name": T.midi_to_name(p),
            "freq": round(440.0 * 2 ** ((p - 69) / 12.0), 2),
            "velocity": int(n["velocity"]),
            "conf": round(0.55 + 0.4 * (n["velocity"] / 127.0), 2),
        })

    stems: list[dict] = []
    for t, d in sorted(by_track.items()):
        if not d["notes"]:
            continue
        fam, _ = MT3.map_family(d["program"], d["is_drum"])
        label = MT3.program_label(d["program"], d["is_drum"])
        d["notes"].sort(key=lambda n: (n["start"], n["pitch"]))

        # A simultaneous onset is initially treated as a chord. It is split
        # only when neighbouring onset groups establish independent sequences.
        groups: list[tuple[str, list[dict]]] = [("", d["notes"])]
        if split_voices and not d["is_drum"] and len(d["notes"]) >= 8:
            if VO.poly_fraction(d["notes"]) >= 0.18:
                parts = VO.separate_sequences(d["notes"])
                if len(parts) > 1:
                    groups = [(f"시퀀스 {k + 1}", p) for k, p in enumerate(parts)]

        for voice_index, (suffix, notes) in enumerate(groups):
            if not notes:
                continue
            vlabel = f"{label} · {suffix}성부" if suffix else label
            # Preserve the source model's track identity. Multiple violin or
            # piano tracks must remain separate, rather than becoming one
            # generic "strings" / "keyboard" bucket.
            sid = f"track{t}" + (f"_voice{voice_index + 1}" if suffix else "")
            stem = {
                "id": sid, "family": fam, "label": vlabel,
                "program": int(d["program"]), "pitched": not d["is_drum"],
                "presence": round(min(1.0, len(notes) / 120.0), 3),
                "peak": 1.0, "spans": _note_spans(notes),
                "duration": 0.0, "engine": "mt3",
                "notes": notes, "contour": T._poly_contour(notes),
                "tempo": 0.0, "sensitivity": round(s, 3),
                "voice": suffix or None,
                "instrument": {"selected": "auto", "detected": fam, "detected_label": vlabel,
                               "preset": "neutral", "features": {}, "options": []},
                "quantized": False, "beat_count": 0,
                "note_count": len(notes),
                "low_conf": sum(1 for n in notes if n["conf"] < 0.5),
                "audio_url": None,
            }
            stem["midi_url"] = _stem_midi_url(job_id, stem, 0.0)
            stems.append(stem)
    stems.sort(key=lambda s: (not s["pitched"], -s["presence"]))
    return stems


def _assemble_mt3(job_id: str, stems_out: list[dict], audio_dur: float,
                  model: str, tempo: float, beats: list | None = None) -> dict:
    notes, contour = _merge_stems(stems_out)
    beats = beats or []
    return {
        "engine": f"mt3:{model}",
        "mode": "mt3",
        "duration": round(audio_dur, 3),
        "tempo": tempo,
        "stems": stems_out,
        "active_stem": None,
        "beats": beats,
        "beat_count": len(beats),
        "quantized": False,
        "musicxml_url": None,
        "edited": False,
        "key": None,
        "low_conf": sum(1 for n in notes if n.get("conf", 1.0) < 0.5),
        "notes": notes,
        "contour": contour,
        "midi_url": _write_merged_midi(job_id, stems_out, tempo),
    }


def _mt3_pipeline(job_id: str, src_path: Path, audio_dur: float,
                  dl_steps: int, _phase) -> dict:
    """MT3 multi-instrument transcription (no Demucs). Holds both slots so it
    never runs alongside a Demucs job."""
    if not _STEMS_SLOT.acquire(timeout=15) or not _MT3_SLOT.acquire(timeout=15):
        raise RuntimeError("다른 정밀 채보/분리 작업이 진행 중입니다. 잠시 후 다시 시도하세요.")
    try:
        _phase(dl_steps + 1, "mt3",
               "MT3 다악기 정밀 채보 중… (2-CPU 서버라 곡 길이의 약 8배 소요)",
               max(30.0, audio_dur * 8.0))
        out = MT3.transcribe(str(src_path))
        raw = out.get("notes", [])
        if not raw:
            raise RuntimeError("MT3 가 음을 찾지 못했습니다.")
        JOBS.get(job_id, {})["mt3_raw"] = raw
        JOBS.get(job_id, {})["mt3_model"] = out.get("model", "mr_mt3")
        try:
            import librosa
            y, sr = librosa.load(str(src_path), sr=22050, mono=True)
            beats, tempo = T._beat_grid(y, sr)
        except Exception:
            beats, tempo = [], 0.0
        JOBS.get(job_id, {})["mt3_tempo"] = tempo
        JOBS.get(job_id, {})["mt3_beats"] = beats
        _phase(dl_steps + 2, "assemble", "악기별·성부별(1st/2nd) 정리 중…", 5.0)
        JOBS.get(job_id, {})["mt3_split_voices"] = True
        JOBS.get(job_id, {})["mt3_max_voices"] = None
        stems_out = _mt3_stems(job_id, raw, T.DEFAULT_SENSITIVITY)
        result = _assemble_mt3(job_id, stems_out, audio_dur,
                               out.get("model", "mr_mt3"), tempo, beats)
        # Validation is advisory: it surfaces likely misses but never invents
        # notes in the delivered score without a user review.
        result["validation"] = Q.audit(str(src_path), stems_out, result["notes"])
        return result
    finally:
        _MT3_SLOT.release()
        _STEMS_SLOT.release()


def _run_job(job_id: str, src: tuple, mode: str) -> None:
    """Background worker. src is ("url", url) or ("file", path, original_name)."""
    acquired = _JOB_SLOTS.acquire(timeout=120)
    if not acquired:
        _set(job_id, status="error", http=503, error="서버가 바쁩니다. 잠시 후 다시 시도하세요.",
             message="서버가 바쁩니다.")
        return

    # progress is PHASE-LOCAL: each named phase runs its own 0..100%. A ticker
    # eases the bar forward while a blocking call runs; real signals (download
    # byte hook) override it. `steps` = how many phases this job has.
    dl_steps = 1 if src[0] == "url" else 0
    # A complete arrangement needs joint multi-instrument transcription, not a
    # bag of notes from a generic polyphonic detector.  "polyphonic" is the
    # public all-instruments mode; keep "mt3" as a backwards-compatible alias.
    is_mt3 = mode in ("polyphonic", "mt3")
    # Source separation is useful when the user explicitly asks for stems, but
    # it is not a neutral pre-processing step for transcription: it smears
    # attacks and leaves cross-instrument bleed.  Direct models must see the
    # original audio for the normal melody/polyphonic/piano workflows.
    use_demucs = mode == "stems"
    total_steps = dl_steps + 2 if (use_demucs or is_mt3) else dl_steps + 1
    creep = {"eta": 8.0, "t0": time.time(), "cap": 0.97}
    stop_tick = threading.Event()

    def _phase(step: int, stage: str, label: str, eta: float) -> None:
        creep.update(eta=max(1.0, eta), t0=time.time())
        _set(job_id, stage=stage, message=label, pct=0.0,
             step=step, steps=total_steps)

    def _ticker() -> None:
        while not stop_tick.wait(0.35):
            frac = min(1.0, (time.time() - creep["t0"]) / creep["eta"])
            eased = creep["cap"] * (1 - (1 - frac) ** 2)
            if eased > JOBS.get(job_id, {}).get("pct", 0.0):
                _set(job_id, pct=eased)

    threading.Thread(target=_ticker, daemon=True).start()

    src_path: Optional[Path] = None
    keep_audio: Optional[Path] = None
    try:
        if src[0] == "url":
            url = src[1]
            _phase(1, "download", "YouTube에서 오디오 받는 중…", 25.0)

            def hook(d: dict) -> None:
                st = d.get("status")
                if st == "downloading":
                    tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    got = d.get("downloaded_bytes") or 0
                    if tot:
                        _set(job_id, pct=min(0.99, got / tot),
                             message="YouTube에서 오디오 받는 중…")
                elif st == "finished":
                    _set(job_id, pct=0.99, message="오디오 변환 중… (ffmpeg)")

            dl_path, title, _dur = fetch.download_audio(url, WORK_DIR, progress_hook=hook)
            src_path = dl_path
            source_name = f"{title} (YouTube)"
            _set(job_id, pct=1.0)
        else:
            src_path = Path(src[1])
            source_name = src[2]

        audio_dur = _probe_duration(src_path) or 45.0

        if is_mt3:
            try:
                result = _mt3_pipeline(job_id, src_path, audio_dur, dl_steps, _phase)
            finally:
                stop_tick.set()
            JOBS.get(job_id, {})["mode"] = "mt3"
        elif use_demucs:
            # every mode: separate first, transcribe each stem, then shape the
            # result for the requested mode (melody = lead line, polyphonic =
            # all stems merged, stems = per-stem breakdown).
            stems_out, _ = _stems_pipeline(job_id, src_path, audio_dur, dl_steps,
                                           _phase, mode=mode)
            stop_tick.set()
            JOBS.get(job_id, {})["mode"] = mode
            result = _assemble(job_id, mode, stems_out, audio_dur)
        else:
            label = "멜로디 분석 중…" if mode == "melody" else "다성(화음) 채보 중…"
            speed = 0.55 if mode == "melody" else 1.4
            _phase(total_steps, "analyze", label, max(4.0, audio_dur * speed))
            try:
                m = "melody" if mode == "melody" else "polyphonic"
                result, analysis = T.transcribe(str(src_path), mode=m)
                JOBS.get(job_id, {})["analysis"] = analysis
                JOBS.get(job_id, {})["mode"] = mode
            finally:
                stop_tick.set()
            _set(job_id, message="MIDI 파일 생성 중…", pct=0.99)
            _write_midi(job_id, result)

        # keep the decoded/downloaded audio around for in-browser playback
        ext = src_path.suffix.lower() or ".wav"
        keep_audio = WORK_DIR / f"{job_id}.audio{ext}"
        try:
            src_path.rename(keep_audio)
            src_path = None
            result["audio_url"] = f"/api/audio/{job_id}{ext}"
        except OSError:
            result["audio_url"] = None

        result["job_id"] = job_id
        result["note_count"] = len(result["notes"])
        result["filename"] = source_name
        _set(job_id, status="done", stage="done", pct=1.0, message="완료", result=result)

    except fetch.TooLong as e:
        _set(job_id, status="error", http=413, error=str(e), message=str(e))
    except fetch.NeedsCookies as e:
        _set(job_id, status="error", http=503, error=str(e), message="YouTube 쿠키 필요")
    except Exception as e:  # noqa: BLE001
        _set(job_id, status="error", http=502, error=str(e), message=f"실패: {e}")
    finally:
        stop_tick.set()
        _JOB_SLOTS.release()
        if src_path is not None:
            try:
                src_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/transcribe", status_code=202)
async def do_transcribe(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    mode: str = Form("polyphonic"),
) -> JSONResponse:
    if mode not in ("polyphonic", "mt3"):
        raise HTTPException(400, "MusicNote는 전체 악기 채보만 지원합니다.")
    if mode in ("polyphonic", "mt3") and not MT3.available():
        raise HTTPException(501, "이 서버에는 MT3 워커가 실행되고 있지 않습니다.")

    url = (url or "").strip()
    has_file = file is not None and (file.filename or "")
    if not has_file and not url:
        raise HTTPException(400, "오디오 파일 또는 YouTube URL 중 하나가 필요합니다.")

    _cleanup()
    job_id = uuid.uuid4().hex

    if url:
        if not fetch.is_supported_url(url):
            raise HTTPException(400, "지원하지 않는 URL 입니다. 현재는 YouTube 링크만 지원합니다.")
        src: tuple = ("url", url)
    else:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(
                415, f"지원하지 않는 형식입니다: {ext or '알 수 없음'} "
                     f"(허용: {', '.join(sorted(ALLOWED_EXT))})")
        data = await file.read()
        if not data:
            raise HTTPException(400, "빈 파일입니다.")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"파일이 너무 큽니다 (최대 {MAX_BYTES // 1024 // 1024} MB).")
        p = WORK_DIR / f"{job_id}{ext}"
        p.write_bytes(data)
        src = ("file", str(p), file.filename)

    JOBS[job_id] = {
        "status": "running", "stage": "queued", "pct": 0.0,
        "message": "대기 중…", "created": time.time(),
    }
    threading.Thread(target=_run_job, args=(job_id, src, mode), daemon=True).start()
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/api/progress/{job_id}")
def progress(job_id: str) -> dict:
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, "알 수 없거나 만료된 작업입니다.")
    out = {
        "status": j["status"],
        "stage": j.get("stage"),
        "pct": round(j.get("pct", 0.0), 4),
        "message": j.get("message", ""),
        "step": j.get("step", 1),
        "steps": j.get("steps", 1),
    }
    if j["status"] == "done":
        out["result"] = j["result"]
    elif j["status"] == "error":
        out["error"] = j.get("error", "알 수 없는 오류")
        out["http"] = j.get("http", 500)
    return out


@app.post("/api/refine/{job_id}")
def refine(job_id: str,
           sensitivity: float = Body(..., embed=True),
           instrument: Optional[str] = Body(None, embed=True),
           stem: Optional[str] = Body(None, embed=True),
           quantize: bool = Body(False, embed=True),
           split_voices: Optional[bool] = Body(None, embed=True),
           max_voices: Optional[int] = Body(None, embed=True)) -> dict:
    """Re-segment a finished job at a new sensitivity (0..1) / instrument preset /
    beat quantisation. For stems mode, `stem` picks which stem. No re-analysis.
    MT3 jobs also accept `split_voices`; voice count is inferred automatically.
    `max_voices` is accepted only for compatibility with older clients."""
    j = JOBS.get(job_id)
    if j is None or j.get("status") != "done":
        raise HTTPException(404, "이 작업은 더 이상 조정할 수 없습니다 (만료되었거나 없음).")
    _keep_alive(job_id)

    s = float(sensitivity)
    inst = instrument or None
    q = bool(quantize)

    # MT3 jobs: re-filter the cached raw note list (velocity/length) and re-split
    # voices — no model re-run, so the 1st/2nd knobs are instant.
    if j.get("mt3_raw") is not None:
        sv = j.get("mt3_split_voices", True) if split_voices is None else bool(split_voices)
        # Omitted or zero means automatic / uncapped. A positive legacy
        # value is still honoured for old API clients.
        mv = int(max_voices) if max_voices and int(max_voices) > 0 else None
        j["mt3_split_voices"], j["mt3_max_voices"] = sv, mv
        try:
            stems_out = _mt3_stems(job_id, j["mt3_raw"], s,
                                   split_voices=sv, max_voices=mv)
            res = _assemble_mt3(job_id, stems_out, j["result"].get("duration", 0.0),
                                j.get("mt3_model", "mr_mt3"), j.get("mt3_tempo", 0.0),
                                j.get("mt3_beats", []))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"재분석 실패: {e}")
        prev = j["result"]
        res["audio_url"] = prev.get("audio_url")
        res["job_id"] = job_id
        res["filename"] = prev.get("filename")
        res["note_count"] = len(res["notes"])
        j["result"] = res
        return res

    cache = j.get("stem_analyses")

    def _reseg_stem(sid: str) -> Optional[dict]:
        r = T.refine(cache[sid], s, inst, quantize=q)
        res = j["result"]
        for i, sd in enumerate(res.get("stems", [])):
            if sd["id"] == sid:
                sd.update(engine=r["engine"], notes=r["notes"], contour=r["contour"],
                          tempo=r["tempo"], sensitivity=r["sensitivity"],
                          instrument=r["instrument"], note_count=len(r["notes"]),
                          quantized=r.get("quantized", False),
                          beat_count=r.get("beat_count", 0),
                          key=r.get("key"), low_conf=r.get("low_conf", 0))
                sd["midi_url"] = _stem_midi_url(job_id, sd, sd.get("tempo") or 0.0)
                res["stems"][i] = sd
                return sd
        return None

    # -- Demucs jobs (all modes): re-segment stem(s) from cached per-stem analyses
    if cache is not None:
        res = j["result"]
        res["quantized"] = q
        mode = j.get("mode", res.get("mode", "stems"))
        try:
            if stem:                                   # one specific stem
                if stem not in cache:
                    raise HTTPException(404, "이 스템은 조정할 수 없습니다.")
                sd = _reseg_stem(stem)
                if res.get("active_stem") == stem:
                    res.update(notes=sd["notes"], contour=sd["contour"],
                               midi_url=sd["midi_url"], tempo=sd["tempo"],
                               note_count=sd["note_count"])
                if mode == "polyphonic":               # keep the merged view fresh
                    notes, contour = _merge_stems(res["stems"])
                    res.update(notes=notes, contour=contour, note_count=len(notes),
                               midi_url=_write_merged_midi(job_id, res["stems"], res["tempo"]))
                return {**res, "changed_stem": sd}

            if mode == "polyphonic":                   # re-segment every stem
                for sid in list(cache):
                    _reseg_stem(sid)
                notes, contour = _merge_stems(res["stems"])
                res.update(notes=notes, contour=contour, note_count=len(notes),
                           midi_url=_write_merged_midi(job_id, res["stems"], res["tempo"]))
                return res

            # melody: re-segment the lead line
            lead = _lead_stem(res["stems"])
            if not lead or lead["id"] not in cache:
                raise HTTPException(404, "조정할 선율 스템이 없습니다.")
            sd = _reseg_stem(lead["id"])
            res.update(notes=sd["notes"], contour=sd["contour"], midi_url=sd["midi_url"],
                       tempo=sd["tempo"], note_count=sd["note_count"],
                       active_stem=lead["id"])
            return {**res, "changed_stem": sd}
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"재분석 실패: {e}")

    # -- non-Demucs fallback jobs: single cached analysis
    if "analysis" not in j:
        raise HTTPException(404, "이 작업은 더 이상 조정할 수 없습니다 (만료되었거나 없음).")
    try:
        res = T.refine(j["analysis"], s, inst, quantize=q)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"재분석 실패: {e}")
    _write_midi(job_id, res)
    prev = j["result"]
    res["audio_url"] = prev.get("audio_url")
    res["job_id"] = job_id
    res["note_count"] = len(res["notes"])
    res["filename"] = prev.get("filename")
    j["result"] = res
    return res


@app.post("/api/edit/{job_id}")
def edit(job_id: str,
         notes: list = Body(..., embed=True),
         tempo: float = Body(120.0, embed=True),
         time_sig: Optional[list] = Body(None, embed=True),
         title: Optional[str] = Body(None, embed=True)) -> dict:
    """Persist a hand-edited note list: rewrite {job_id}.mid and generate
    {job_id}.musicxml. Used by the in-browser score / piano-roll editor."""
    j = JOBS.get(job_id)
    if j is None or j.get("status") != "done":
        raise HTTPException(404, "이 작업은 만료되었거나 없습니다.")
    _keep_alive(job_id)
    if not isinstance(notes, list) or len(notes) > 5000:
        raise HTTPException(400, "음표 목록이 올바르지 않습니다 (최대 5000개).")

    clean: list[dict] = []
    for n in notes:
        try:
            st, en, p = float(n["start"]), float(n["end"]), int(round(float(n["pitch"])))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= p <= 127) or en <= st or st < 0 or en > 36000:
            continue
        vel = n.get("velocity", 90)
        try:
            vel = int(max(1, min(127, int(vel))))
        except (TypeError, ValueError):
            vel = 90
        clean.append({
            "start": round(st, 4), "end": round(en, 4), "pitch": p,
            "name": T.midi_to_name(p),
            "freq": round(440.0 * 2 ** ((p - 69) / 12.0), 2),
            "velocity": vel,
        })
    clean.sort(key=lambda n: (n["start"], n["pitch"]))

    if (isinstance(time_sig, list) and len(time_sig) == 2):
        try:
            ts = (int(time_sig[0]), int(time_sig[1]))
        except (TypeError, ValueError):
            ts = (4, 4)
    else:
        ts = (4, 4)
    tempo = float(tempo) or 120.0
    ttl = (title or j["result"].get("filename") or "MusicNote")

    try:
        pm = T.notes_to_midi(clean, tempo=tempo, name=ttl)
        pm.write(str(WORK_DIR / f"{job_id}.mid"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"MIDI 생성 실패: {e}")
    # the detected beat grid is only valid at the tempo it was detected at; if the
    # user overrode the tempo, fall back to a grid derived from that tempo
    _bt = j["result"].get("beats") or None
    if _bt and abs(float(j["result"].get("tempo") or 0) - tempo) > 1.0:
        _bt = None
    try:
        xml = MX.build(clean, tempo=tempo, time_sig=ts, title=ttl, beats=_bt)
        (WORK_DIR / f"{job_id}.musicxml").write_text(xml, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"MusicXML 생성 실패: {e}")

    res = j["result"]
    res["notes"] = clean
    res["note_count"] = len(clean)
    res["tempo"] = tempo
    res["edited"] = True
    res["midi_url"] = f"/api/download/{job_id}.mid"
    res["musicxml_url"] = f"/api/download/{job_id}.musicxml"
    return {
        "midi_url": res["midi_url"],
        "musicxml_url": res["musicxml_url"],
        "note_count": len(clean),
        "tempo": tempo,
    }


def _score_parts(res: dict, stem: Optional[str] = None) -> list[dict]:
    """Result → parts for score_build (one entry per instrument/voice)."""
    stems = res.get("stems") or []
    if stem and stem != "__all__":
        sd = next((s for s in stems if s["id"] == stem), None)
        if sd:
            return [{"name": sd.get("label") or sd["id"], "notes": sd.get("notes") or [],
                     "program": sd.get("program", 0), "is_drum": not sd.get("pitched", True)}]
    if stems:
        out = [{"name": s.get("label") or s["id"], "notes": s.get("notes") or [],
                "program": s.get("program", 0), "is_drum": not s.get("pitched", True)}
               for s in stems if s.get("notes")]
        if out:
            return out
    return [{"name": res.get("filename") or "Music", "notes": res.get("notes") or []}]


@app.get("/api/score/{job_id}")
def score(job_id: str, stem: Optional[str] = None,
          num: int = 4, den: int = 4, tempo: Optional[float] = None,
          fifths: Optional[int] = None) -> dict:
    """The ScoreDoc as JSON — the single notation source the browser renders,
    so screen and MusicXML can no longer drift apart."""
    j = JOBS.get(job_id)
    if j is None or j.get("status") != "done":
        raise HTTPException(404, "이 작업은 만료되었거나 없습니다.")
    _keep_alive(job_id)
    res = j["result"]
    from dataclasses import asdict
    from score_build import build_score
    tmp = float(tempo or res.get("tempo") or 120.0)
    bts = res.get("beats") or None
    if bts and abs(float(res.get("tempo") or 0) - tmp) > 1.0:
        bts = None            # tempo overridden → detected beats no longer apply
    try:
        doc = build_score(_score_parts(res, stem),
                          beats=bts,
                          tempo=tmp,
                          time_sig=(int(num), int(den)),
                          title=res.get("filename") or "MusicNote",
                          key_fifths=fifths)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"악보 생성 실패: {e}")
    return asdict(doc)


def _safe_stem(name: str) -> str:
    """`<hex>` or `<hex>_<stemid>` only."""
    if "/" in name or ".." in name or not name.replace("_", "").isalnum():
        raise HTTPException(400, "잘못된 이름입니다.")
    return name


@app.get("/api/download/{name}")
def download(name: str) -> FileResponse:
    if name.endswith(".mid"):
        _safe_stem(name[:-4])
        media_type = "audio/midi"
    elif name.endswith(".musicxml"):
        _safe_stem(name[: -len(".musicxml")])
        media_type = "application/vnd.recordare.musicxml+xml"
    else:
        raise HTTPException(400, "잘못된 이름입니다.")
    path = WORK_DIR / name
    if not path.exists():
        raise HTTPException(404, "만료되었거나 존재하지 않는 결과입니다.")
    return FileResponse(path, media_type=media_type, filename=f"musicnote-{name}")


@app.get("/api/audio/{name}")
def audio(name: str) -> FileResponse:
    stem, _, ext = name.partition(".")
    ext = "." + ext.lower()
    _safe_stem(stem)
    if ext not in _AUDIO_MIME:
        raise HTTPException(400, "잘못된 이름입니다.")
    # main mix is "<job>.audio.<ext>"; separated stems are "<job>_<id>.<ext>"
    path = WORK_DIR / f"{stem}.audio{ext}"
    if not path.exists():
        path = WORK_DIR / f"{stem}{ext}"
    if not path.exists():
        raise HTTPException(404, "만료되었거나 존재하지 않는 오디오입니다.")
    return FileResponse(path, media_type=_AUDIO_MIME[ext])


# static frontend (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
