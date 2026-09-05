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

import json
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
import mt3_ensemble as ME
import mt3_post as MP
import muscriptor_bridge as MU
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
# Absolute paths of source audio that a running job still needs. _cleanup()
# skips these; entries are added when a job takes ownership of a file and
# removed in _run_job's finally.
_IN_FLIGHT: set[str] = set()
_JOB_SLOTS = threading.Semaphore(MAX_JOBS)
_STEMS_SLOT = threading.Semaphore(1)   # Demucs is heavy: one at a time
# MT3 and Demucs both peak RAM hard — an MT3 job also holds _STEMS_SLOT so they
# can never run together on this box.
_MT3_SLOT = threading.Semaphore(1)

# Seconds of silent lead-in for each extra MT3 pass. YourMT3 reads
# non-overlapping 2.048 s segments, so an attack landing on a boundary can be
# dropped entirely; shifting the lead-in moves every boundary somewhere else.
# The quarter-segment offsets below put each boundary at a different place in
# all four runs. Empty disables the ensemble and with it the review queue.
#
# This used to be a single 1.024 s pass unioned with the base run. Measured
# against the reference MIDI on both eval sets (eval/replay_eval.py machinery,
# onset within 50 ms and same pitch):
#
#                                   solo piano          band
#                                  P     R    F1     P     R    F1
#   single run                   0.971 0.913 0.941  0.718 0.575 0.639
#   0,1024 union   (was shipped) 0.949 0.944 0.946  0.598 0.676 0.635
#   0,512,1024,1536 union        0.919 0.951 0.935  0.463 0.738 0.569
#   0,512,1024,1536 agree>=2     0.965 0.937 0.951  0.762 0.599 0.671
#   0,512,1024,1536 agree>=3     0.989 0.922 0.954  0.868 0.480 0.618
#
# Two things this says. Union was actively harmful on band material — worse
# than not running an ensemble at all (0.635 vs 0.639), because every extra run
# contributes its own false positives and nothing filters them. And the earlier
# "union beats vote" result was not wrong, it was measured on solo piano only,
# where the model is accurate enough that a second opinion is nearly free.
#
# Four runs at agreement 2 is the only configuration that improves BOTH sets.
# agree>=3 wins on solo piano and collapses on band, so it is not the default.
# The top of the band sweep is a plateau (0.667-0.674 across 4-6 runs at about
# half agreement), so evenly spaced quarters is a point on the plateau rather
# than a value fitted to these eight clips.
#
# Cost is four model passes instead of two. On the GPU worker that is seconds;
# on the CPU fallback it doubles an already long wait.
# Re-picked after mt3_ensemble._key stopped keying on the track, which changed
# the agreement distribution the previous choice was made under. Three runs now
# match four on F1 for both sets while printing fewer wrong notes:
#
#                          밴드 P     R     F1  |  솔로 P     R     F1
#   0,512,1024,1536 (4실행) 0.737 0.633 0.681  | 0.965 0.937 0.951
#   0,1024,1536     (3실행) 0.787 0.600 0.681  | 0.976 0.930 0.952
#
# The tie is broken on precision and cost. In a score a wrong note is visible as
# a wrong note, while a missing one only shows against the audio, and three
# passes is 25% less inference — which matters on the CPU fallback, where each
# pass runs at roughly 8x the clip length.
ENSEMBLE_SHIFTS = tuple(
    float(x) for x in os.environ.get(
        "MUSICNOTE_MT3_ENSEMBLE_SHIFTS", "1.024,1.536").split(",") if x.strip())
# Extra passes on a RESAMPLED copy, in semitones. Resampling moves pitch and
# tempo together by an exact factor, so it is perfectly invertible; the model is
# asked the same music in another register, through different segment
# boundaries, with no spectral damage. (Time-stretching is the obvious
# alternative and is much worse: on band00 a 0.8x librosa stretch held onsets up
# but dropped note F1 from 0.600 to 0.363, because the phase vocoder smears the
# harmonic structure the pitch decoder reads.)
#
# Measured over the 8 band clips, as a FOURTH run added to the three shifts
# above, at agreement 2 — with a control, because a fourth run of any kind also
# relaxes the vote from 2-of-3 to 2-of-4:
#
#   fourth run          onset F1   note F1   +offset F1   notes
#   none (3 shifts)       0.829     0.688      0.618       421
#   another shift (512)   0.841     0.692      0.620       472
#   another shift (768)   0.846     0.688      0.614       481
#   +1 semitone           0.846     0.702      0.628       472
#
# A fourth SHIFT buys +0.004 and +0.000 — the shifted ensemble is saturated,
# which matches the earlier "8 runs ~ 3 runs" result. The same inference spent
# on a transposed run buys +0.014 at the same note count, because the shifted
# runs are strongly correlated with each other and this one is not. On its own
# the transposed run is a peer of the base run (mean note F1 0.657 vs 0.649).
#
# It composes with the octave correction rather than overlapping it — the
# correction repairs notes already found (precision), the transposed run finds
# notes the shifted runs missed (recall), and recall is what binds now:
#
#   octave fix  transpose      P       R      F1
#       -           -        0.781   0.634   0.688
#       -           yes      0.749   0.676   0.702
#       yes         -        0.817   0.671   0.725
#       yes         yes      0.779   0.709   0.734
#
# Default: ON where the pass is cheap, OFF where it is not. A GPU pass is
# seconds; the CPU fallback runs at roughly 8x the clip length, so a fourth pass
# there turns a 40 minute wait into 53. Accuracy is worth a lot but not that.
# Set the env var to a semitone list to force it either way ("" disables).
_TRANSPOSE_ENV = os.environ.get("MUSICNOTE_MT3_ENSEMBLE_TRANSPOSE")
ENSEMBLE_TRANSPOSE = (None if _TRANSPOSE_ENV is None else
                      tuple(int(x) for x in _TRANSPOSE_ENV.split(",") if x.strip()))
ENSEMBLE_TRANSPOSE_ON_GPU = (1,)
# A dedicated pass for the register the ensemble above cannot fix by voting.
# Recall by octave on the band clips: C1-B1 (24-35) 6.1%, C2-B2 66.3%, C3-B3
# 64.9%, C4-B4 89.6% (home register), C5-B5 65.7%. A real note below MIDI 36
# gets at most ONE vote from every run in ENSEMBLE_SHIFTS/ENSEMBLE_TRANSPOSE —
# they are all deaf down there together — so agreement>=2 throws it away
# whether or not it is correct. Voting cannot fix a blind spot every voter
# shares.
#
# The fix is not to vote harder but to trust a different witness: a pass
# resampled UP moves that register into the model's home range (24-35 becomes
# 36-47), and below the cutoff its answer is taken alone instead of counted.
# Above the cutoff it is not consulted at all — the base ensemble is already
# good there and the transposed pass has been pushed out of ITS home register.
#
# Swept +12 and +7 semitones against cutoffs 36/42/48/54/60 on eval/refs_band,
# on top of the shifted ensemble + octave correction:
#
#   semitones   cutoff    P       R      F1   notes added
#       -         -     0.809   0.730   0.766        0
#      +7        <36    0.800   0.764   0.779      110
#      +7        <42    0.788   0.766   0.774      142
#      +7        <48    0.758   0.776   0.764      247
#     +12        <36    0.803   0.744   0.770       51
#     +12        <42    0.790   0.748   0.766       85
#
# +7 beat +12 at every cutoff (a smaller shift keeps more of the note's own
# harmonic content intact), and 36 is the best cutoff by a clear margin — past
# it the added recall stops being worth the precision it costs. On the solo
# set the same transposed pass is worse than the base run at every clip
# (mean note F1 0.917 vs 0.939) — solo piano's bass is already in range, so
# this pass is not run there at all: it costs a whole extra inference for a
# cutoff that rarely has a note under it.
BASS_RESCUE_ENV = os.environ.get("MUSICNOTE_MT3_BASS_RESCUE")
BASS_RESCUE_SEMITONES = (None if BASS_RESCUE_ENV is None else
                         (int(BASS_RESCUE_ENV) if BASS_RESCUE_ENV.strip() else None))
BASS_RESCUE_ON_GPU = 7
BASS_RESCUE_CUTOFF = int(os.environ.get("MUSICNOTE_MT3_BASS_RESCUE_CUTOFF", "36"))
# A second rescue, keyed on TIMBRE rather than register. Measured on
# eval/refs_band by splitting recall against the reference MIDI's own GM
# instrument, not by pitch: acoustic instruments (piano, acoustic bass) recall
# 84.2% through the shipped ensemble (YourMT3 + shifted runs + octave
# correction); electric/synth instruments recall only 37.8% -- AND THE GAP
# HOLDS WITHIN THE SAME REGISTER, so it is not the low-register blind spot
# above, it is a separate one: MT3 was not trained to recognise these timbres
# well, at any pitch. No pitch shift or ensemble vote fixes a model not
# knowing what an instrument sounds like.
#
# MuScriptor (Kyutai + Mirelo, arXiv 2607.08168) labels its own output by
# timbre (its vocabulary distinguishes acoustic_bass/electric_bass,
# clean/distorted_electric_guitar, synth_lead/synth_pad...) and was measured
# on the same 8 clips, single pass, no ensemble, no post-processing:
#
#   family          YourMT3 ensemble (shipped)   MuScriptor small (single pass)
#   acoustic                84.2%                        91.9%
#   electric/synth          37.8%                        65.8%
#
# Both axes improve -- not a trade-off -- so below, notes MuScriptor tagged
# with one of these labels are trusted on their own (never touching the
# families YourMT3 already covers well) exactly like the register-gated bass
# rescue above: the base ensemble was never able to vote on what it could not
# recognise, so disagreement there is not evidence against a note.
#
# MuScriptor separately regresses hard on solo material (mean note F1 0.74 vs
# our 0.92-0.99) and its bundled beat tracker is worse than meter.detect() at
# rhythm (7/13 refs_meter clips failed outright) -- see ACCURACY.md 2026-09-05.
# So this stays a narrow, timbre-gated addition, not a wholesale replacement.
MUSCRIPTOR_TIMBRES = frozenset(
    os.environ.get("MUSICNOTE_MUSCRIPTOR_TIMBRES",
                   "electric_bass,organ,synth_lead,synth_pad,tuba").split(","))
MUSCRIPTOR_RESCUE = os.environ.get("MUSICNOTE_MUSCRIPTOR_RESCUE", "1") == "1"
# Runs a note must appear in to be delivered. 1 = union (keep everything, review
# only the disagreements); 2 = vote. Notes below the threshold are not deleted —
# they become review candidates the user can accept.
MIN_AGREEMENT = int(os.environ.get("MUSICNOTE_MT3_MIN_AGREEMENT", "2"))


# Job state that survives a restart. Everything here is plain JSON; the numpy
# analysis caches (`analysis`, `stem_analyses`, `_sal`, model outputs) are
# deliberately left out — they exist only to make a refine fast, they are large,
# and a restored job falls back to the same "cannot be re-tuned" path an expired
# one already used. `mt3_raw` IS kept, so an MT3 job stays fully re-tunable
# across a restart, which is the case that actually mattered.
_PERSIST_KEYS = (
    "status", "stage", "pct", "message", "created", "mode", "http", "error",
    "result", "mt3_raw", "mt3_review", "mt3_model", "mt3_tempo", "mt3_beats",
    "mt3_meter", "mt3_split_voices", "mt3_max_voices", "mt3_runs",
    "mix_beats", "mix_tempo",
)


def _jsonable(o):
    """numpy scalars/arrays leak into results from the analysis stages."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _persist(job_id: str) -> None:
    """Snapshot a finished job to {job_id}.job.json.

    Results used to live only in memory, so any backend restart — including a
    deploy — silently threw away whatever the user was looking at. The file goes
    in WORK_DIR under the job id so that _cleanup() expires it and _keep_alive()
    refreshes it with the job's other files, on the same TTL, with no separate
    lifetime to keep in sync.

    Only finished jobs are written: a running job's worker thread does not
    survive the restart, so persisting one would restore a job that can never
    make progress.
    """
    j = JOBS.get(job_id)
    if not j or j.get("status") == "running":
        return
    snap = {k: j[k] for k in _PERSIST_KEYS if k in j}
    path = WORK_DIR / f"{job_id}.job.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(snap, default=_jsonable, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)          # atomic: a reader never sees a half file
    except (OSError, TypeError, ValueError) as e:  # noqa: BLE001
        print(f"job {job_id[:8]} not persisted: {e}", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _restore() -> None:
    """Reload persisted jobs at startup, skipping any past their TTL.

    Age comes from the file's mtime rather than the `created` field inside it,
    because _keep_alive() extends a job's life by touching its files. Reading
    the field instead would expire a job the user is actively working on.
    """
    now = time.time()
    n = 0
    for path in WORK_DIR.glob("*.job.json"):
        jid = path.name[:-len(".job.json")]
        try:
            if now - path.stat().st_mtime > RESULT_TTL:
                continue
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if jid not in JOBS:
            snap["created"] = path.stat().st_mtime
            JOBS[jid] = snap
            n += 1
    if n:
        print(f"restored {n} job(s) from disk", flush=True)


def _cleanup() -> None:
    now = time.time()
    busy = set(_IN_FLIGHT)
    for p in WORK_DIR.glob("*"):
        try:
            # Never reap a file a running job is still reading. Age alone is not
            # enough: a long upload plus a slow four-pass transcription can
            # outlive RESULT_TTL, and then the next submission deletes the
            # source out from under the job that is mid-inference. The symptom
            # is remote, so it reads as a worker fault:
            #   mt3 remote backend failed (FileNotFoundError: ...uploads/x.wav)
            if str(p) in busy:
                continue
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


# An octave correction only fires when the harmonic evidence for the other
# register beats "leave it alone" by this factor, over a part with at least this
# many notes. Both guards exist because a harmonic comb an octave DOWN also
# covers the note itself (2nd harmonic), a fifth above it (3rd) and the octave
# above (4th), so the lower template can explain everything the correct one does
# plus more, and wins on raw activation. Taking the plain argmax scores WORSE
# than doing nothing (0.613 vs 0.688 on the band clips).
#
# Swept over the 8 band clips. Every cell of ratio 2.0-3.5 x min-notes 20-60
# gives the same 0.725 with two moves and none of them wrong, so this is the
# middle of a plateau rather than a fitted point:
#
#   band clips   note F1        moves   wrong
#   no correction  0.688          -       -
#   plain argmax   0.613         14       9
#   guarded        0.725          2       0
#   oracle         0.732          -       -
#
# 84% of the oracle ceiling, for no extra inference. On the solo set it makes
# ZERO moves and leaves F1 at 0.958 — solo piano has no systematic displacement,
# and the guards correctly find nothing to do.
OCTAVE_RATIO = float(os.environ.get("MUSICNOTE_MT3_OCTAVE_RATIO", "2.5"))
OCTAVE_MIN_NOTES = int(os.environ.get("MUSICNOTE_MT3_OCTAVE_MIN_NOTES", "40"))


def _mt3_octaves(raw: list[dict], wav_path: str) -> None:
    """Move a whole part back into its own register when the audio says so.

    MT3's largest single error on band material is octave DISPLACEMENT, and it
    is systematic per instrument: on band03 the bass came back with 86.5% of its
    104 notes a whole octave up. Because every run makes the same mistake, the
    shifted-run vote cannot see it — and it is 28% of all matched notes, larger
    than everything else in this pipeline put together.

    The decision is taken once per part, from a harmonic-comb NMF: a template at
    pitch p explains p, 2p, 3p..., so asking which of p-12 / p / p+12 the energy
    really belongs to is exactly what that decomposition answers. A per-NOTE
    version of this was tried first and lost F1; a whole part carries enough
    evidence to be decided reliably, a single note does not.
    """
    try:
        import numpy as np       # deferred like the rest of the heavy imports
        y, sr = T.load_audio(wav_path)
        sal = T._salience_cqt(y, sr)
        nmf = T._harmonic_nmf(sal["C"], sal["t"]) if sal else None
        if not nmf:
            return
        H, lo, t = nmf["H"], nmf["lo_midi"], nmf["t"]
        by_track: dict[int, list[dict]] = {}
        for n in raw:
            if not n.get("is_drum"):     # a kit slot has no register to correct
                by_track.setdefault(int(n.get("track", 0)), []).append(n)
        for track, notes in by_track.items():
            if len(notes) < OCTAVE_MIN_NOTES:
                continue
            support = {}
            for d in (-12, 0, 12):
                vals = []
                for n in notes:
                    r = int(n["pitch"]) + d - lo
                    if not (0 <= r < H.shape[0]):
                        vals.append(0.0)
                        continue
                    i0 = min(int(np.searchsorted(t, n["start"])), H.shape[1] - 1)
                    i1 = min(max(i0 + 1, int(np.searchsorted(t, n["end"]))),
                             H.shape[1])
                    vals.append(float(H[r, i0:i1].mean()))
                support[d] = float(np.mean(vals)) if vals else 0.0
            stay = support[0]
            best = max((-12, 12), key=lambda d: support[d])
            if stay <= 0 or support[best] < OCTAVE_RATIO * stay:
                continue
            for n in notes:
                n["pitch"] = int(n["pitch"]) + best
            print(f"mt3 octave: track {track} ({len(notes)} notes) moved "
                  f"{best:+d} (support {support[best]:.3f} vs {stay:.3f})",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"mt3 octave correction skipped: {e}", flush=True)


def _mt3_bass_rescue(rescue_notes: list[dict], accepted: list[dict],
                      total_runs: int, cutoff: int = BASS_RESCUE_CUTOFF
                      ) -> list[dict]:
    """Pull in low notes the voting ensemble could never have accepted.

    ``rescue_notes`` is a transposed-up pass, already mapped back to real
    pitch and time by ``mt3_bridge._untranspose``. Below ``cutoff`` every run
    in the shifted/semitone ensemble is close to deaf (see the measurement at
    BASS_RESCUE_CUTOFF above), so a real note down there was never going to
    reach agreement>=2 — not because the runs disagree, but because none of
    them could supply a second vote. So below the cutoff this pass is trusted
    on its own instead of counted; above it, nothing here is touched.

    A rescued note is given ``agreement = total_runs`` so it survives the
    sensitivity slider re-splitting at display time exactly like a note every
    run agreed on — the alternative (agreement 1) would make the slider hide
    it at the very setting most users leave the pipeline at.
    """
    have: dict[int, list[float]] = {}
    for n in accepted:
        if not n.get("is_drum"):
            have.setdefault(int(n["pitch"]), []).append(float(n["start"]))
    extra = []
    for n in rescue_notes:
        if n.get("is_drum") or int(n["pitch"]) >= cutoff:
            continue
        starts = have.get(int(n["pitch"]), ())
        if any(abs(s - float(n["start"])) <= 0.05 for s in starts):
            continue
        extra.append({**n, "agreement": total_runs, "runs": [],
                      "source": "bass_rescue"})
    return extra


def _muscriptor_timbre_rescue(wav_path: str, accepted: list[dict],
                              total_runs: int) -> list[dict]:
    """Pull in notes from instrument families YourMT3 was never trained to
    recognise well, regardless of register (see MUSCRIPTOR_TIMBRES above).

    Guarded like every other extra pass in this pipeline: a MuScriptor failure
    (worker down, timeout, bad audio) must not lose the transcription already
    in hand, so this returns an empty list rather than raising.
    """
    if not MUSCRIPTOR_RESCUE:
        return []
    try:
        if not MU.available():
            return []
        out = MU.transcribe(wav_path)
    except Exception as e:  # noqa: BLE001
        print(f"muscriptor timbre rescue skipped: {e}", flush=True)
        return []
    have: dict[int, list[float]] = {}
    for n in accepted:
        if not n.get("is_drum"):
            have.setdefault(int(n["pitch"]), []).append(float(n["start"]))
    extra = []
    for n in out.get("notes") or []:
        if n.get("instrument") not in MUSCRIPTOR_TIMBRES:
            continue
        starts = have.get(int(n["pitch"]), ())
        if any(abs(s - float(n["start"])) <= 0.05 for s in starts):
            continue
        extra.append({"start": float(n["start"]), "end": float(n["end"]),
                      "pitch": int(n["pitch"]), "velocity": int(n.get("velocity", 100)),
                      "track": int(n.get("track", 0)), "program": int(n.get("program", 0)),
                      "is_drum": False, "agreement": total_runs, "runs": [],
                      "source": "muscriptor", "instrument": n["instrument"]})
    return extra


def _mt3_dynamics(raw: list[dict], wav_path: str) -> None:
    """Stamp every MT3 note with a real loudness (``dyn``) and a 10-point
    amplitude envelope (``env``), read from the audio's constant-Q transform.

    MT3 reports a constant velocity of 100 for every note it emits, so without
    this every note in a transcription plays back at exactly the same level and
    the result sounds mechanical. Measured on eval/refs_band, loudness read this
    way does NOT tell a correct note from a wrong one (AUC 0.452, i.e. no better
    than chance), so it is deliberately not fed into ``conf`` and never removes
    anything — it only decides how loud a note is played.

    Written to ``dyn``, not ``velocity``: mt3_post.gate() thresholds on
    ``velocity``, and a real loudness there would turn the sensitivity slider
    into a filter that deletes quiet notes. Done once per job so that a refine,
    which re-runs the gate and the voice split, stays instant.
    """
    try:
        y, sr = T.load_audio(wav_path)
        sal = T._salience_cqt(y, sr)
        if sal is None:
            return
        pitched = [n for n in raw if not n.get("is_drum")]
        if pitched:
            T._note_dynamics(pitched, {"_sal": sal}, field="dyn", override=True)
        # Drum hits have no fundamental to read, so they keep a flat level.
        for n in raw:
            n.setdefault("dyn", int(n.get("velocity", 100)))
    except Exception as e:  # noqa: BLE001
        print(f"mt3 dynamics failed, playback stays flat: {e}", flush=True)


def _mt3_stems(job_id: str, raw: list[dict], sensitivity: float,
               split_voices: bool = True, max_voices: int | None = None,
               runs: int = 1) -> list[dict]:
    """Group MT3 notes by model track and infer notation lines when needed.

    Polyphonic lines are inferred from actual overlap, pitch continuity and
    register history. They are not claimed to be separate performers.
    """
    s = max(0.0, min(1.0, sensitivity))
    kept, gate_report = MP.gate(raw, s)
    # The ensemble vote is the real sensitivity control on this path; the
    # velocity floor above cannot be, because YourMT3 pins velocity at 100.
    need = MP.required_agreement(s, runs)
    if runs > 1:
        before = len(kept)
        kept = [n for n in kept if int(n.get("agreement", runs)) >= need]
        gate_report["agreement_floor"] = need
        gate_report["dropped_agreement"] = before - len(kept)
    if (gate_report["dropped_length"] or gate_report["dropped_velocity"]
            or gate_report.get("dropped_agreement")):
        print(f"mt3 gate: {gate_report}", flush=True)
    by_track: dict[int, dict] = {}
    for n in kept:
        # All percussion is one kit. MT3 can emit several drum tracks for the
        # same performance — they are channel-10 events whose "program" is a
        # guess, and the same guess we already know drifts between runs (see
        # mt3_ensemble._key). Left separate they became two drum staves for one
        # drummer, the second of which was 54% rests.
        #
        # Pitched tracks are NOT merged this way. Measured on eval/refs_band
        # against the reference instrument labels, a small track is not a
        # fragment of a big one: of tracks with 1-5 notes only 33% duplicate an
        # instrument some larger track already carries, which is the same rate
        # as tracks with 41+ notes (36%). Folding them together would erase a
        # real part two times out of three.
        # ... and pitched notes are grouped by the PROGRAM, not the track id.
        # The track id is an arbitrary slot; the program is the instrument MT3
        # named, and one track can carry several programs while two tracks can
        # carry the same one — which printed two staves both labelled "패드 1".
        # Measured on eval/refs_band against the reference instrument labels:
        #   트랙 id     파트 53  라인 62  P 0.509  R 0.718  F1 0.596
        #   프로그램     파트 54  라인 63  P 0.517  R 0.732  F1 0.606
        t = -1 if n.get("is_drum") else int(n.get("program", 0))
        d = by_track.setdefault(t, {"program": n["program"], "is_drum": n["is_drum"], "notes": []})
        p = int(n["pitch"])
        d["notes"].append({
            "start": round(float(n["start"]), 3), "end": round(float(n["end"]), 3),
            "pitch": p, "name": T.midi_to_name(p),
            "freq": round(440.0 * 2 ** ((p - 69) / 12.0), 2),
            # `velocity` is the measured loudness where we have one; `conf` stays
            # on the model's own value because loudness was measured not to
            # predict correctness (see _mt3_dynamics).
            "velocity": int(n.get("dyn", n["velocity"])),
            # Real per-note confidence: how many shifted runs found this note.
            # The old formula read MT3's velocity, which is the constant 100, so
            # every note came out at 0.865 and the UI's uncertainty highlight
            # was decorative. See mt3_post.confidence for the calibration.
            "conf": MP.confidence(int(n.get("agreement", runs)), runs),
            "agreement": int(n.get("agreement", runs)),
        })
        if n.get("env"):
            d["notes"][-1]["env"] = n["env"]

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
            if VO.poly_fraction(d["notes"]) >= VO.SPLIT_POLY_GATE:
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
                  model: str, tempo: float, beats: list | None = None,
                  met: dict | None = None) -> dict:
    notes, contour = _merge_stems(stems_out)
    beats = beats or []
    met = met or {}
    ts = tuple(met.get("time_sig") or (4, 4))
    return {
        "time_sig": [int(ts[0]), int(ts[1])],
        "downbeats": met.get("downbeats") or [],
        "meter_confidence": met.get("confidence", 0.0),
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
        # `on_gpu` is needed before the run count, because whether the
        # transposed pass runs at all depends on it.
        on_gpu = MT3.remote_url() is not None
        transposes = (ENSEMBLE_TRANSPOSE if ENSEMBLE_TRANSPOSE is not None
                      else (ENSEMBLE_TRANSPOSE_ON_GPU if on_gpu else ()))
        rescue_semis = (BASS_RESCUE_SEMITONES if BASS_RESCUE_SEMITONES is not None
                        else (BASS_RESCUE_ON_GPU if on_gpu else None))
        runs_wanted = 1 + len(ENSEMBLE_SHIFTS) + len(transposes)
        # The estimate has to follow the backend actually in use. A GPU worker
        # runs about 3x faster than realtime; the local CPU fallback is ~8x
        # slower than realtime, per pass. Quoting the CPU figure while a GPU is
        # answering told the user to expect a 32x wait for a 40 s job.
        per_run = 0.35 if on_gpu else 8.0
        total_passes = runs_wanted + (1 if rescue_semis is not None else 0)
        eta = max(20.0, audio_dur * per_run * total_passes)
        where = "GPU 워커" if on_gpu else "2-CPU 서버라 곡 길이의 약 " \
                                          f"{int(8 * total_passes)}배 소요"
        _phase(dl_steps + 1, "mt3", f"MT3 다악기 정밀 채보 중… ({where})", eta)
        out = MT3.transcribe(str(src_path))
        raw = out.get("notes", [])
        if not raw:
            raise RuntimeError("MT3 가 음을 찾지 못했습니다.")

        # Extra passes at shifted segment boundaries. A failure in any of them
        # must not lose the transcription we already have, so each is guarded
        # and the run list simply ends up shorter.
        runs = [raw]
        for k, shift in enumerate(ENSEMBLE_SHIFTS, start=2):
            _phase(dl_steps + 1, "mt3",
                   f"누락 검증용 {k}/{runs_wanted}차 채보 중… "
                   "(세그먼트 경계를 옮겨 재추론)",
                   max(20.0, audio_dur * per_run))
            try:
                shifted = MT3.transcribe(str(src_path), shift=shift)
                if shifted.get("notes"):
                    runs.append(shifted["notes"])
            except Exception as e:  # noqa: BLE001
                print(f"mt3 ensemble pass at {shift}s failed: {e}", flush=True)

        for j, semis in enumerate(transposes, start=2 + len(ENSEMBLE_SHIFTS)):
            _phase(dl_steps + 1, "mt3",
                   f"누락 검증용 {j}/{runs_wanted}차 채보 중… "
                   "(음역을 옮겨 재추론)",
                   max(20.0, audio_dur * per_run))
            try:
                moved = MT3.transcribe(str(src_path), semitones=semis)
                if moved.get("notes"):
                    runs.append(moved["notes"])
            except Exception as e:  # noqa: BLE001
                print(f"mt3 ensemble pass at {semis} semitones failed: {e}",
                      flush=True)

        # A further pass purely for the register no voting run can reach — see
        # BASS_RESCUE_CUTOFF above. Not added to `runs`: it does not vote, it
        # is trusted alone below the cutoff after the vote is already decided.
        rescue_raw = None
        if rescue_semis is not None:
            _phase(dl_steps + 1, "mt3",
                   f"{runs_wanted + 1}/{total_passes}차 베이스 보강 채보 중… "
                   "(음역을 크게 올려 재추론)",
                   max(20.0, audio_dur * per_run))
            try:
                moved = MT3.transcribe(str(src_path), semitones=rescue_semis)
                rescue_raw = moved.get("notes") or None
            except Exception as e:  # noqa: BLE001
                print(f"mt3 bass rescue pass at {rescue_semis} semitones "
                      f"failed: {e}", flush=True)

        # Agreement is a fraction of the runs that actually returned, so losing
        # a pass relaxes the threshold instead of silently rejecting good notes.
        need = MIN_AGREEMENT if len(runs) == runs_wanted else max(
            1, round(MIN_AGREEMENT * len(runs) / runs_wanted))

        merged = ME.merge(runs)
        raw, review = ME.split(merged, len(runs), need)
        if len(runs) > 1:
            print(f"mt3 ensemble: runs={len(runs)} agree>={need} "
                  f"merged={len(merged)} accepted={len(raw)} "
                  f"review={len(review)}", flush=True)
        # The whole merged list is kept, not just what the vote accepted, so
        # the sensitivity slider can loosen as well as tighten without re-running
        # the model. _mt3_stems applies the vote at display time.
        # Before dynamics: a moved part must be at its real pitch before the
        # loudness read looks for energy there.
        _mt3_octaves(merged, str(src_path))
        if rescue_raw:
            extra = _mt3_bass_rescue(rescue_raw, raw, len(runs))
            if extra:
                merged.extend(extra)
                raw = raw + extra
                print(f"mt3 bass rescue: +{len(extra)} notes below "
                      f"{BASS_RESCUE_CUTOFF} ({rescue_semis:+d} semitones)",
                      flush=True)
        # Same idea, keyed on timbre instead of register: a MuScriptor pass is
        # cheap (near real-time on CPU, unlike the extra MT3 passes above), so
        # it is not GPU-gated.
        timbre_extra = _muscriptor_timbre_rescue(str(src_path), raw, len(runs))
        if timbre_extra:
            merged.extend(timbre_extra)
            raw = raw + timbre_extra
            print(f"muscriptor timbre rescue: +{len(timbre_extra)} notes "
                  f"({sorted(set(n['instrument'] for n in timbre_extra))})",
                  flush=True)
        _mt3_dynamics(merged, str(src_path))
        JOBS.get(job_id, {})["mt3_raw"] = merged
        JOBS.get(job_id, {})["mt3_runs"] = len(runs)
        JOBS.get(job_id, {})["mt3_review"] = review
        JOBS.get(job_id, {})["mt3_model"] = out.get("model", "mr_mt3")
        # Metre comes from the transcription, not from the audio. librosa reads
        # a spectral-flux onset envelope, which on eval/refs_meter got the tempo
        # right 10 times in 13 but produced beat F1 0.818, no downbeat at all,
        # and no time signature. meter.detect() works from the notes MT3 just
        # found (onset F1 0.958): beat F1 0.868, downbeat F1 0.592, and the time
        # signature right 10 times in 13 instead of being hardcoded to 4/4.
        import meter as MET
        met = MET.detect(raw)
        if not met.get("beats"):
            try:                       # too few notes to find a pulse in
                import librosa
                y, sr = librosa.load(str(src_path), sr=22050, mono=True)
                b, tp = T._beat_grid(y, sr)
                met = {"tempo": tp, "beats": b, "downbeats": [],
                       "time_sig": (4, 4), "confidence": 0.0}
            except Exception:
                met = {"tempo": 0.0, "beats": [], "downbeats": [],
                       "time_sig": (4, 4), "confidence": 0.0}
        tempo, beats = met.get("tempo", 0.0), met.get("beats", [])
        JOBS.get(job_id, {})["mt3_tempo"] = tempo
        JOBS.get(job_id, {})["mt3_beats"] = beats
        JOBS.get(job_id, {})["mt3_meter"] = met
        _phase(dl_steps + 2, "assemble", "악기별·성부별(1st/2nd) 정리 중…", 5.0)
        JOBS.get(job_id, {})["mt3_split_voices"] = True
        JOBS.get(job_id, {})["mt3_max_voices"] = None
        stems_out = _mt3_stems(job_id, merged, T.DEFAULT_SENSITIVITY,
                               runs=len(runs))
        result = _assemble_mt3(job_id, stems_out, audio_dur,
                               out.get("model", "mr_mt3"), tempo, beats, met)
        # Validation is advisory: it surfaces likely misses but never invents
        # notes in the delivered score without a user review.
        result["validation"] = Q.audit(str(src_path), stems_out, result["notes"],
                                       ensemble_candidates=review, runs=len(runs))
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
    owned = ""          # source path this job owns; see _IN_FLIGHT
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

        owned = str(src_path)         # src_path is set to None after the
        _IN_FLIGHT.add(owned)         # rename, so keep the key separately
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
        _IN_FLIGHT.discard(owned)
        _persist(job_id)
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
                                   split_voices=sv, max_voices=mv,
                                   runs=int(j.get("mt3_runs", 1)))
            res = _assemble_mt3(job_id, stems_out, j["result"].get("duration", 0.0),
                                j.get("mt3_model", "mr_mt3"), j.get("mt3_tempo", 0.0),
                                j.get("mt3_beats", []), j.get("mt3_meter"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"재분석 실패: {e}")
        prev = j["result"]
        res["audio_url"] = prev.get("audio_url")
        res["job_id"] = job_id
        res["filename"] = prev.get("filename")
        res["note_count"] = len(res["notes"])
        j["result"] = res
        _persist(job_id)
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
    _persist(job_id)
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
    _persist(job_id)
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
        # MT3 sequence stems are interpretations *inside* one model instrument
        # track.  Keep them as notation voices in one part; rendering each as a
        # separate instrument staff was the main reason the score looked unlike
        # the piano roll. Other engines retain their real stem separation.
        mt3_groups: dict[str, list[dict]] = {}
        other = []
        for s in stems:
            if not s.get("notes"):
                continue
            if s.get("engine") == "mt3":
                source = s["id"].split("_voice", 1)[0]
                mt3_groups.setdefault(source, []).append(s)
            else:
                other.append(s)
        out = []
        for group in mt3_groups.values():
            first = group[0]
            name = (first.get("label") or first["id"]).split(" · 시퀀스", 1)[0]
            out.append({"name": name, "voices": [s.get("notes") or [] for s in group],
                        "program": first.get("program", 0),
                        "is_drum": not first.get("pitched", True)})
        out.extend({"name": s.get("label") or s["id"], "notes": s.get("notes") or [],
                    "program": s.get("program", 0), "is_drum": not s.get("pitched", True)}
                   for s in other)
        if out:
            return out
    return [{"name": res.get("filename") or "Music", "notes": res.get("notes") or []}]


@app.get("/api/score/{job_id}")
def score(job_id: str, stem: Optional[str] = None,
          num: Optional[int] = None, den: Optional[int] = None,
          tempo: Optional[float] = None, fifths: Optional[int] = None) -> dict:
    """The ScoreDoc as JSON — the single notation source the browser renders,
    so screen and MusicXML can no longer drift apart.

    `num`/`den` omitted means the detected time signature (see meter.detect);
    passing them is the editor overriding it. They used to default to 4/4, which
    drew every 3/4 and 6/8 piece in four."""
    j = JOBS.get(job_id)
    if j is None or j.get("status") != "done":
        raise HTTPException(404, "이 작업은 만료되었거나 없습니다.")
    _keep_alive(job_id)
    res = j["result"]
    from dataclasses import asdict
    from score_build import build_score
    ts_det = res.get("time_sig") or [4, 4]
    num = int(num) if num else int(ts_det[0])
    den = int(den) if den else int(ts_det[1])
    tmp = float(tempo or res.get("tempo") or 120.0)
    bts = res.get("beats") or None
    if bts and abs(float(res.get("tempo") or 0) - tmp) > 1.0:
        bts = None            # tempo overridden → detected beats no longer apply
    try:
        doc = build_score(_score_parts(res, stem),
                          beats=bts,
                          # Detected downbeats decide where the barlines go.
                          # They are only valid alongside the beats they came
                          # from, so an overridden tempo drops both.
                          downbeats=(res.get("downbeats") or None) if bts else None,
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
# Pick up any job that outlived the last process. Done at import, before the
# first request, so a reload never briefly 404s a job that is on disk.
_restore()

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
