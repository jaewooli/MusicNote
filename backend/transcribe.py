"""
Audio -> note extraction.

Two engines:
  * "melody"      : monophonic pitch tracking (librosa pYIN). Robust, always available.
  * "polyphonic"  : full note transcription. Uses Spotify's basic-pitch when it is
                    installed; otherwise falls back to a CQT peak-picking heuristic.

The heavy analysis (pYIN / basic-pitch) runs once and is cached as an "analysis"
object; turning the *sensitivity* knob only re-runs the cheap segmentation step
(:func:`refine`), so the UI can slide it in real time.

Accuracy passes that are always on (independent of the knob):
  * global tuning estimate subtracted before rounding to semitones
  * median smoothing of the pitch curve + gross octave-jump correction
  * note boundaries snapped to detected onsets (also splits repeated notes)

Every engine returns the same shape:

    {
      "engine": "pyin" | "basic-pitch" | "cqt-fallback",
      "mode": "melody" | "polyphonic",
      "duration": float seconds,
      "tempo": float BPM (estimate),
      "sensitivity": float 0..1,
      "notes": [ {start, end, pitch, name, freq, velocity}, ... ],
      "contour": [ {t, freq, midi}, ... ]   # sparse pitch curve for plotting
    }
"""
from __future__ import annotations

import os
import threading
from typing import Any

import librosa
import numpy as np

TARGET_SR = 22050
BP_SR = 22050
HOP = 256

# neural pitch detector (torchcrepe) — replaces pYIN for monophonic stems.
#   model:   'tiny' is the shipping default (0.46× realtime here). 'full' is
#            4×+ realtime on this 2-CPU box — opt in with MUSICNOTE_CREPE_MODEL=full
#            (only used for a short HQ lead, and forced to the fast decoder).
#   decoder: 'viterbi' temporally decodes the pitch posterior — far fewer octave
#            jumps than 'weighted_argmax', and essentially free with 'tiny'.
_CREPE_MODEL = os.environ.get("MUSICNOTE_CREPE_MODEL", "auto")         # auto|tiny|full
_CREPE_DECODER = os.environ.get("MUSICNOTE_CREPE_DECODER", "viterbi")  # viterbi|weighted_argmax
# 'full' measured ~8× realtime here, so 'auto' does NOT reach for it by default
# (cap 0). Raise MUSICNOTE_CREPE_FULL_MAX (seconds) to let a short HQ lead use it,
# or set MUSICNOTE_CREPE_MODEL=full to force it everywhere.
_CREPE_FULL_MAX_SEC = float(os.environ.get("MUSICNOTE_CREPE_FULL_MAX", "0"))
_CREPE_SR = 16000
_CREPE_HOP = 160   # 10 ms


def _has_crepe() -> bool:
    try:
        import torchcrepe  # noqa: F401
        return True
    except Exception:
        return False


def _has_piano_model() -> bool:
    try:
        import piano_transcription_inference  # noqa: F401
        return True
    except Exception:
        return False

DEFAULT_SENSITIVITY = 0.5
# Pitch-moving "musical" priors, off by default — measured to hurt note F1.
KEY_SNAP = os.environ.get("MUSICNOTE_KEY_SNAP") == "1"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def load_audio(path: str, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, _file_sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    if y.size == 0:
        raise ValueError("빈 오디오이거나 디코딩할 수 없는 파일입니다.")
    return y, sr


def _beats(y: np.ndarray, sr: int) -> list[float]:
    """Beat times (s) for optional rhythmic quantisation."""
    return _beat_grid(y, sr)[0]


def _beat_grid(y: np.ndarray, sr: int) -> tuple[list[float], float]:
    """One beat-tracker pass -> (beat_times, tempo_bpm). The mix's own tempo is a
    far steadier default than a single separated stem's estimate."""
    try:
        tempo, bt = librosa.beat.beat_track(y=y, sr=sr, units="time")
        t = float(np.atleast_1d(tempo).reshape(-1)[0])
        return [round(float(x), 4) for x in np.atleast_1d(bt)], round(t, 1)
    except Exception:
        return [], 0.0


# --------------------------------------------------------------------------- #
# bleed / ghost-note gate  (used for Demucs stems, where separation leaks other
# instruments into a stem as faint low-amplitude notes)
# --------------------------------------------------------------------------- #
_SAL_HOP = 1024
_SAL_BASE_MIDI = 24          # C1
_SAL_BINS = 84               # C1 .. B7


def _salience_cqt(y: np.ndarray, sr: int) -> dict | None:
    """A small constant-Q magnitude map, cached on the analysis so :func:`refine`
    can score how much of the stem's own energy actually sits at each note."""
    try:
        C = np.abs(librosa.cqt(
            y, sr=sr, hop_length=_SAL_HOP,
            fmin=float(librosa.midi_to_hz(_SAL_BASE_MIDI)),
            n_bins=_SAL_BINS, bins_per_octave=12)).astype("float32")
        t = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=_SAL_HOP)
        return {"C": C, "t": t.astype("float32")}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# harmonic-templated NMF  (a second opinion on polyphonic content)
# --------------------------------------------------------------------------- #
_NMF_LO_MIDI = 33   # A1
_NMF_HI_MIDI = 96   # C7
_NMF_HARMONICS = [(1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25),
                  (5, 0.18), (6, 0.14), (7, 0.11), (8, 0.09)]


def _harmonic_nmf(C: np.ndarray, t: np.ndarray, n_iter: int = 25) -> dict | None:
    """Decompose the cached CQT with a FIXED per-pitch harmonic-comb dictionary
    (W held constant, only activations H solved). Gives a per-pitch/per-frame
    activation map that is independent of basic-pitch — used to confirm / deny /
    recover notes in :func:`_nmf_ensemble`. Cheap: 64×64·frames·n_iter."""
    try:
        nb = C.shape[0]
        pitches = np.arange(_NMF_LO_MIDI, _NMF_HI_MIDI + 1)
        W = np.zeros((nb, pitches.size), dtype=np.float64)
        for j, p in enumerate(pitches):
            for h, amp in _NMF_HARMONICS:
                b = (p - _SAL_BASE_MIDI) + 12.0 * np.log2(h)
                b0 = int(np.floor(b))
                frac = b - b0
                for bb, w in ((b0, (1.0 - frac) * amp), (b0 + 1, frac * amp)):
                    if 0 <= bb < nb:
                        W[bb, j] += w
        W /= (W.sum(axis=0, keepdims=True) + 1e-9)
        V = np.asarray(C, dtype=np.float64)
        V = V / (V.max() + 1e-9)
        WtV = W.T @ V
        WtW = W.T @ W
        H = np.maximum(WtV, 1e-6)
        for _ in range(n_iter):
            H *= WtV / (WtW @ H + 1e-9)
        return {"H": H.astype(np.float32), "lo_midi": int(_NMF_LO_MIDI),
                "t": np.asarray(t, dtype=np.float32)}
    except Exception:
        return None


def _nmf_support(n: dict, nmf: dict) -> float:
    """Fraction of the NMF activation energy that sits at this note's pitch over
    its span (0 = not there, ~1 = the only thing sounding)."""
    try:
        H, lo, t = nmf["H"], nmf["lo_midi"], nmf["t"]
        row = int(round(n["pitch"])) - lo
        if not (0 <= row < H.shape[0]):
            return 0.5
        i0 = min(int(np.searchsorted(t, n["start"])), H.shape[1] - 1)
        i1 = min(max(i0 + 1, int(np.searchsorted(t, n["end"]))), H.shape[1])
        seg = H[:, i0:i1]
        if seg.size == 0:
            return 0.5
        local = float(np.median(seg[row]))
        tot = float(np.median(seg.sum(axis=0))) + 1e-9
        return _clip01(local / tot * 2.0)
    except Exception:
        return 0.5


def _nmf_ensemble(notes: list[dict], a: dict, sensitivity: float) -> list[dict]:
    """Cross-check a polyphonic note list against the harmonic NMF:
      * drop notes the NMF gives no support to (only the quiet ones, ≤25 %,
        and only when the knob is low)
      * add sustained NMF activations that no note covers (only when the knob is
        ≥ 0.4, capped, so a noisy NMF can't flood the result)"""
    nmf = a.get("_nmf")
    if not nmf or len(notes) < 4:
        return notes
    s = _clip01(sensitivity)

    drop_thr = 0.05 + 0.10 * (1.0 - s)
    scored = [(n, _nmf_support(n, nmf)) for n in notes]
    weak = [n for n, sup in scored if sup < drop_thr and n.get("velocity", 90) < 70]
    kept = ([n for n, sup in scored
             if not (sup < drop_thr and n.get("velocity", 90) < 70)]
            if 0 < len(weak) <= 0.25 * len(notes) else list(notes))

    if s >= 0.4:
        H, lo, t = nmf["H"], nmf["lo_midi"], nmf["t"]
        Hn = H / (H.max() + 1e-9)
        thr = 0.16 - 0.08 * s
        step = float(np.median(np.diff(t))) if t.size > 1 else 0.046
        min_frames = max(2, int((0.12 - 0.05 * s) / max(step, 1e-3)))
        occupied: dict[int, list] = {}
        for n in kept:
            occupied.setdefault(int(round(n["pitch"])), []).append((n["start"], n["end"]))
        cap = int(0.35 * len(kept)) + 3
        add: list[dict] = []
        for row in range(H.shape[0]):
            if len(add) >= cap:
                break
            pitch = lo + row
            on = Hn[row] > thr
            i = 0
            while i < on.size and len(add) < cap:
                if not on[i]:
                    i += 1
                    continue
                k = i
                while k < on.size and on[k]:
                    k += 1
                if k - i >= min_frames:
                    st = float(t[i])
                    en = float(t[min(k, t.size - 1)])
                    amp = float(np.median(Hn[row, i:k]))
                    # skip if the octave above is stronger here (this is its
                    # sub-harmonic ghost, not a real note)
                    ghost = (row + 12 < H.shape[0]
                             and float(np.median(Hn[row + 12, i:k])) > amp * 1.3)
                    if not ghost and not any(
                            st < e and en > s0 for s0, e in occupied.get(pitch, [])):
                        add.append({
                            "start": round(st, 3), "end": round(en, 3), "pitch": pitch,
                            "name": midi_to_name(pitch),
                            "freq": round(float(librosa.midi_to_hz(pitch)), 2),
                            "velocity": int(max(20, min(110, round(40 + amp * 80)))),
                            "_nmf_added": True,
                        })
                i = k
        kept = kept + add

    kept.sort(key=lambda n: (n["start"], n["pitch"]))
    return _drop_octave_doublings(kept)


def _span_overlap_frac(n: dict, spans) -> float:
    dur = n["end"] - n["start"]
    if dur <= 0:
        return 0.0
    ov = 0.0
    for a, b in spans:
        ov += max(0.0, min(n["end"], b) - max(n["start"], a))
    return ov / dur


def _gate_notes(notes: list[dict], a: dict, sensitivity: float) -> list[dict]:
    """Drop notes that look like source-separation bleed:
      1. notes sitting where the stem itself is silent (outside its active spans)
      2. notes with almost no constant-Q energy at their own pitch relative to
         everything else sounding in that instant
    Conservative: never removes more than ~30 %, never runs on sparse output,
    and relaxes as the sensitivity knob goes up."""
    if len(notes) < 6:
        return notes
    s = _clip01(sensitivity)
    out = notes

    spans = a.get("spans")
    if spans:
        need = 0.35 - 0.20 * s
        keep = [n for n in out if _span_overlap_frac(n, spans) >= need]
        if len(keep) >= max(4, int(0.55 * len(out))):
            out = keep

    sal = a.get("_sal")
    if sal is not None and len(out) >= 6:
        # polyphonic stems legitimately carry soft inner voices, so the salience
        # gate is much gentler there than for a single monophonic line.
        is_poly = a.get("kind") == "poly"
        C, ct = sal["C"], sal["t"]
        nb, nf = C.shape
        vals = np.empty(len(out), dtype=float)
        for k, n in enumerate(out):
            i0 = min(int(np.searchsorted(ct, n["start"])), nf - 1)
            i1 = min(max(i0 + 1, int(np.searchsorted(ct, n["end"]))), nf)
            seg = C[:, i0:i1]
            if seg.size == 0:
                vals[k] = 1.0
                continue
            base = int(round(n["pitch"])) - _SAL_BASE_MIDI
            rows = [r for r in (base, base + 12, base + 19) if 0 <= r < nb]
            local = float(np.median(seg[rows].max(axis=0))) if rows else 0.0
            colmax = float(np.median(seg.max(axis=0))) + 1e-9
            vals[k] = local / colmax
        floor = (0.025 if is_poly else 0.04) + (0.035 if is_poly else 0.06) * (1.0 - s)
        cut = int((0.15 if is_poly else 0.30) * len(vals))
        order = np.argsort(vals)
        drop = {int(k) for k in order[:cut] if vals[k] < floor}
        if drop:
            out = [n for k, n in enumerate(out) if k not in drop]
    return out


# --------------------------------------------------------------------------- #
# musical post-processing  (key / metre / octave priors — always on, gentle)
# --------------------------------------------------------------------------- #
_MAJ_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJ_STEPS = (0, 2, 4, 5, 7, 9, 11)
_MIN_STEPS = (0, 2, 3, 5, 7, 8, 10, 11)   # natural minor + raised 7th (harmonic)


def _key_from_sal(a: dict) -> dict:
    """Krumhansl–Schmuckle key estimate from the cached CQT (folded to chroma).
    Returns tonic pitch-class, mode, correlation strength and the allowed set."""
    sal = a.get("_sal")
    try:
        C = sal["C"]
        nb = C.shape[0] - (C.shape[0] % 12)
        chroma = C[:nb].reshape(-1, 12, C.shape[1]).sum(axis=0).mean(axis=1)
        chroma = chroma / (chroma.sum() + 1e-9)
        maj = _MAJ_PROFILE / _MAJ_PROFILE.sum()
        minp = _MIN_PROFILE / _MIN_PROFILE.sum()
        best = (-2.0, 0, "major")
        for tonic in range(12):
            for prof, name in ((maj, "major"), (minp, "minor")):
                r = float(np.corrcoef(chroma, np.roll(prof, tonic))[0, 1])
                if r > best[0]:
                    best = (r, tonic, name)
        strength, tonic, mode = best
        steps = _MAJ_STEPS if mode == "major" else _MIN_STEPS
        return {"tonic": tonic, "mode": mode, "strength": round(strength, 3),
                "pcs": sorted({(tonic + i) % 12 for i in steps})}
    except Exception:
        return {"tonic": 0, "mode": "major", "strength": 0.0, "pcs": list(range(12))}


def _running_center(pitches: np.ndarray, starts: np.ndarray, win: float = 1.5) -> np.ndarray:
    """Slowly-varying pitch centre (windowed median) for octave-coherence."""
    out = np.empty(pitches.shape, dtype=float)
    for i, t0 in enumerate(starts):
        m = np.abs(starts - t0) <= win
        out[i] = float(np.median(pitches[m])) if m.any() else float(pitches[i])
    return out


def _musical_cleanup(notes: list[dict], a: dict, sensitivity: float,
                     mono: bool) -> list[dict]:
    """Gentle priors applied after segmentation:
      * octave-coherence  (monophonic): pull ±12/±24 outliers to the local centre
      * key snap          : nudge a weak/short out-of-key note by ≤1 semitone
      * beat start-snap   : snap note starts to the 16th grid when very close
      * micro-note drop   : remove sub-32nd blips with no supporting onset
    All bounded so real chromaticism / rubato survive; relaxes as sensitivity ↑."""
    if len(notes) < 4:
        return notes
    s = _clip01(sensitivity)
    notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))

    # ---- octave coherence (melody only) ----
    if mono and len(notes) >= 5:
        P = np.array([n["pitch"] for n in notes], dtype=float)
        S = np.array([n["start"] for n in notes], dtype=float)
        centre = _running_center(P, S)
        for i, n in enumerate(notes):
            d = n["pitch"] - centre[i]
            if abs(d) >= 8:
                for k in (12, -12, 24, -24):
                    if abs(d - k) <= 3:
                        n["pitch"] = int(n["pitch"] - k)
                        break

    # ---- key snap ----
    # MEASURED 2026-08-30: moving out-of-key notes by a semitone LOWERED note F1
    # on both engines (basic-pitch 0.566 -> 0.503, piano-tx 0.560 -> 0.533).
    # Real music is full of chromaticism, so this "musical prior" corrupted
    # correct pitches. Key detection still runs (the score needs a key
    # signature); only the pitch-moving is off unless explicitly asked for.
    key = a.get("key") or _key_from_sal(a)
    a["key"] = key
    allowed = set(key.get("pcs", range(12)))
    if KEY_SNAP and key.get("strength", 0.0) >= 0.6 and len(allowed) < 12:
        snap_dur = 0.20 + 0.10 * s        # only "short" notes are eligible
        for i, n in enumerate(notes):
            if n["pitch"] % 12 in allowed:
                continue
            nb_ok = ((i == 0 or notes[i - 1]["pitch"] % 12 in allowed) and
                     (i + 1 >= len(notes) or notes[i + 1]["pitch"] % 12 in allowed))
            if not (n["end"] - n["start"] < snap_dur or nb_ok):
                continue
            for cand in (n["pitch"] - 1, n["pitch"] + 1):
                if 0 <= cand <= 127 and cand % 12 in allowed:
                    n["pitch"] = int(cand)
                    break

    # refresh names / freqs after any pitch shift, then merge equal neighbours
    for n in notes:
        n["name"] = midi_to_name(n["pitch"])
        n["freq"] = round(float(librosa.midi_to_hz(n["pitch"])), 2)
    merged: list[dict] = []
    for n in notes:
        if (merged and merged[-1]["pitch"] == n["pitch"]
                and n["start"] - merged[-1]["end"] <= 0.06):
            merged[-1]["end"] = max(merged[-1]["end"], n["end"])
        else:
            merged.append(n)
    notes = merged

    # ---- beat start-snap + micro-note drop ----
    beats = a.get("beats") or []
    if len(beats) >= 2:
        bt = np.asarray(beats, dtype=float)
        period = float(np.median(np.diff(bt)))
        step = period / 4.0                       # 16th grid
        if step > 1e-3:
            grid0 = bt[0] - step * round(bt[0] / step)
            tol = min(0.045, step * 0.4)
            for n in notes:
                q = grid0 + round((n["start"] - grid0) / step) * step
                if 0 <= q < n["end"] and abs(q - n["start"]) <= tol:
                    n["start"] = round(float(q), 3)
        floor_len = max(0.045, period / 10.0) * (1.0 - 0.5 * s)
        onset_t = np.asarray(a.get("onset_t") if a.get("onset_t") is not None else [], dtype=float)

        def near_onset(t: float) -> bool:
            return onset_t.size > 0 and bool(np.min(np.abs(onset_t - t)) <= 0.04)

        notes = [n for n in notes
                 if (n["end"] - n["start"]) >= floor_len or near_onset(n["start"])]

    return notes


def quantize_notes(notes: list[dict], beat_times: list[float],
                   subdiv: int = 4) -> list[dict]:
    """Snap note starts/ends to a beat grid subdivided `subdiv` ways."""
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2 or not notes:
        return notes
    step = float(np.median(np.diff(bt))) / max(1, subdiv)
    if step <= 1e-3:
        return notes
    grid = np.arange(bt[0] - step * subdiv, bt[-1] + step * (subdiv + 1), step)

    def snap(t: float) -> float:
        i = int(np.searchsorted(grid, t))
        lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i)]
        return float(lo if abs(t - lo) <= abs(t - hi) else hi)

    out = []
    for n in notes:
        s = snap(n["start"])
        e = snap(n["end"])
        if e <= s:
            e = s + step
        out.append({**n, "start": round(s, 3), "end": round(e, 3)})
    return out


def estimate_tempo(y: np.ndarray, sr: int) -> float:
    try:
        tempo = librosa.feature.rhythm.tempo(y=y, sr=sr, aggregate=np.median)
        return float(np.atleast_1d(tempo)[0])
    except Exception:
        return 0.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------- #
# timbre -> instrument family -> transcription preset
# --------------------------------------------------------------------------- #
# A pure-pitch tracker treats a plucked piano note and a bowed violin note the
# same way, so both come out wrong (piano: attacks lost / notes over-held;
# violin: vibrato split into many notes, soft entries dropped). We classify the
# dominant timbre from the waveform and pick a segmentation preset for it.

_MEL_PRESETS = {
    #             median kernels     onset       min note   merge gap
    "struck":     dict(k_short=5,  k_long=21, onset_bias=1.15, min_dur=0.055, gap=0.035),
    "sustained":  dict(k_short=9,  k_long=45, onset_bias=0.30, min_dur=0.120, gap=0.120),
    "percussive": dict(k_short=3,  k_long=13, onset_bias=1.5,  min_dur=0.050, gap=0.030),
    "neutral":    dict(k_short=5,  k_long=31, onset_bias=0.80, min_dur=0.090, gap=0.060),
    # bass: slow, held, simple — smooth hard, don't chop on every transient
    "bass":       dict(k_short=9,  k_long=51, onset_bias=0.45, min_dur=0.110, gap=0.090),
}
_BP_PRESETS = {
    "struck":     dict(onset_d=0.00,  frame_d=0.00,  min_ms=105, melodia=True),
    "sustained":  dict(onset_d=-0.12, frame_d=-0.07, min_ms=150, melodia=True),
    "percussive": dict(onset_d=0.12,  frame_d=0.05,  min_ms=85,  melodia=False),
    "neutral":    dict(onset_d=0.00,  frame_d=0.00,  min_ms=128, melodia=True),
}
# user-selectable instrument -> preset key ("auto" defers to detection)
_INSTRUMENT_PRESET = {
    "auto": None,
    "piano": "struck", "guitar": "struck", "pluck": "struck", "mallet": "struck",
    "keyboard": "struck", "plucked": "struck",
    "strings": "sustained", "winds": "sustained", "voice": "sustained",
    "synth": "sustained",
    "bass": "bass",
    "drums": "percussive",
    "other": "neutral",
}
_INSTRUMENT_LABEL = {
    "piano": "피아노", "guitar": "기타·발현", "strings": "현악(활)",
    "winds": "관악", "voice": "목소리", "drums": "타악", "other": "기타/불명",
    "bass": "베이스", "keyboard": "건반", "plucked": "발현 현악",
}
INSTRUMENT_OPTIONS = [
    ("piano", "피아노"), ("guitar", "기타 · 발현악기"), ("strings", "현악 (바이올린 등)"),
    ("winds", "관악"), ("voice", "목소리"), ("drums", "타악"), ("other", "기타 / 불명"),
]


def classify_instrument(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Cheap timbre classifier over the loudest ~25 s. Returns a coarse family."""
    n = y.size
    if n < sr // 2:
        return {"family": "other", "label": _INSTRUMENT_LABEL["other"],
                "confidence": 0.0, "features": {}}

    win = min(n, int(25 * sr))
    if n > win:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
        w = max(1, win // 1024)
        if rms.size > w:
            energy = np.convolve(rms ** 2, np.ones(w), "valid")
            seg = y[int(np.argmax(energy)) * 1024:][:win]
        else:
            seg = y[:win]
    else:
        seg = y
    if seg.size < sr // 2:
        seg = y[:win]

    try:
        y_h, y_p = librosa.effects.hpss(seg)
        e_h, e_p = float(np.sum(y_h ** 2)), float(np.sum(y_p ** 2))
        harm = e_h / (e_h + e_p + 1e-12)
    except Exception:
        harm = 0.6

    cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(seg)))
    rms = librosa.feature.rms(y=seg)[0]
    tf = np.arange(rms.size)
    tc = float((tf * rms).sum() / (rms.sum() + 1e-12) / max(rms.size - 1, 1))  # 0..1
    onsets = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
    orate = float(len(onsets) / max(seg.size / sr, 1e-6))

    if harm < 0.42 and flat > 0.035 and orate > 2.2:
        lab = "drums"
    elif tc < 0.42 and harm > 0.45:
        lab = "piano" if cent < 2700 else "guitar"
    elif harm > 0.5 and flat < 0.03:
        if zcr > 0.09 or cent > 2600:
            lab = "strings"
        elif flat > 0.012:
            lab = "voice"
        else:
            lab = "winds"
    else:
        lab = "other"

    return {
        "family": lab,
        "label": _INSTRUMENT_LABEL.get(lab, lab),
        "confidence": round(min(1.0, abs(tc - 0.42) * 2 + harm - 0.4), 2),
        "features": {"harmonicity": round(harm, 3), "centroid_hz": round(cent, 1),
                     "flatness": round(flat, 4), "zcr": round(zcr, 4),
                     "temporal_centroid": round(tc, 3), "onset_rate": round(orate, 2)},
    }


def _preset_key(instrument: str | None, detected_family: str) -> str:
    key = (instrument or "auto").lower()
    if key == "auto" or key not in _INSTRUMENT_PRESET:
        return _INSTRUMENT_PRESET.get(detected_family) or "neutral"
    return _INSTRUMENT_PRESET.get(key) or "neutral"


def _instrument_block(a: dict[str, Any], instrument: str | None, preset_key: str) -> dict:
    det = a.get("instrument") or {"family": "other", "label": _INSTRUMENT_LABEL["other"]}
    return {
        "selected": (instrument or "auto").lower(),
        "detected": det["family"],
        "detected_label": det["label"],
        "preset": preset_key,
        "features": det.get("features", {}),
        "options": ([{"value": "auto", "label": f"자동 감지 ({det['label']})"}]
                    + [{"value": v, "label": lb} for v, lb in INSTRUMENT_OPTIONS]),
    }


# --------------------------------------------------------------------------- #
# shared: pitch-curve smoothing + onset-aware segmentation
# --------------------------------------------------------------------------- #
def _smooth_midi_track(midi_cont: np.ndarray, k_short: int = 5,
                       k_long: int = 31) -> np.ndarray:
    """Median-smooth a frame-wise (continuous) MIDI pitch curve and pull gross
    octave jumps back to the local tessitura. NaN where unvoiced."""
    from scipy.signal import medfilt

    m = np.asarray(midi_cont, dtype=float)
    valid = np.isfinite(m) & (m > 0)
    if valid.sum() < 6:
        return m

    k_short = max(1, int(k_short) | 1)
    k_long = max(k_short, int(k_long) | 1)
    idx = np.arange(m.size)
    filled = np.interp(idx, idx[valid], m[valid])
    short = medfilt(filled, kernel_size=k_short)
    long = medfilt(filled, kernel_size=k_long)

    off = short - long
    k = np.round(off / 12.0)
    fix = (np.abs(k) >= 1) & (np.abs(off - 12.0 * k) < 3.0)
    short = np.where(fix, short - 12.0 * k, short)

    short[~valid] = np.nan
    return short


def _onsets(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        return librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=HOP, units="time", backtrack=True)
    except Exception:
        return np.array([], dtype=float)


def _segment(midi_int: np.ndarray, times: np.ndarray, voiced: np.ndarray,
             onset_t: np.ndarray, min_dur: float, max_gap: float,
             onset_bias: float = 0.8) -> list[dict]:
    """Group a frame-wise integer-MIDI track into notes, using onsets to place
    starts and to split consecutive same-pitch attacks. ``onset_bias`` scales
    how strongly onsets drive boundaries (low for legato/bowed, high for
    struck/percussive)."""
    onset_t = np.asarray(onset_t, dtype=float)
    varr = np.asarray(voiced)
    # voiced may be a plain bool mask, or a 0/1/2 level array carrying a
    # Schmitt-trigger hysteresis (2 = may start a note, 1 = may only sustain).
    has_hyst = varr.dtype != bool and int(varr.max(initial=0)) >= 2
    snap_win = 0.03 + 0.04 * onset_bias
    split_win = 0.045 * onset_bias
    do_resplit = onset_bias >= 0.7

    def snap(t: float) -> float:
        if onset_t.size == 0:
            return t
        j = int(np.argmin(np.abs(onset_t - t)))
        return float(onset_t[j]) if abs(onset_t[j] - t) <= snap_win else t

    def near_onset(t: float) -> bool:
        return (do_resplit and onset_t.size > 0
                and bool(np.min(np.abs(onset_t - t)) <= split_win))

    def onset_between(a: float, b: float) -> bool:
        return onset_t.size > 0 and bool(np.any((onset_t > a + 0.02) & (onset_t < b - 0.02)))

    notes: list[dict] = []
    cur_pitch: int | None = None
    cur_start = 0.0
    last_t = 0.0

    def flush(end_t: float) -> None:
        nonlocal cur_pitch
        if cur_pitch is not None and end_t - cur_start >= min_dur:
            notes.append({
                "start": round(cur_start, 3),
                "end": round(end_t, 3),
                "pitch": int(cur_pitch),
                "name": midi_to_name(cur_pitch),
                "freq": round(float(librosa.midi_to_hz(cur_pitch)), 2),
                "velocity": 90,
            })
        cur_pitch = None

    for i, t in enumerate(times):
        p = midi_int[i]
        lvl = int(varr[i])
        ok = lvl >= 1 and np.isfinite(p) and p > 0
        can_start = ok and (lvl >= 2 or not has_hyst)
        pr = int(p) if ok else None

        if pr is None:
            if cur_pitch is not None and t - last_t > max_gap:
                flush(last_t)
        elif cur_pitch is None:
            if can_start:
                cur_pitch, cur_start = pr, snap(t)
        elif pr != cur_pitch:
            flush(t)
            cur_pitch, cur_start = pr, snap(t)
        elif near_onset(t) and t - cur_start >= min_dur:
            flush(t)
            cur_pitch, cur_start = pr, snap(t)
        if ok:
            last_t = t
    flush(last_t)

    merged: list[dict] = []
    for n in notes:
        if (merged and merged[-1]["pitch"] == n["pitch"]
                and n["start"] - merged[-1]["end"] <= max_gap
                and not onset_between(merged[-1]["end"], n["start"])):
            merged[-1]["end"] = n["end"]
        else:
            merged.append(n)

    # drop lone short blips (usually source-separation bleed): a brief note
    # with no neighbour within ~1 s on either side.
    out: list[dict] = []
    for i, n in enumerate(merged):
        if n["end"] - n["start"] >= 2 * min_dur:
            out.append(n)
            continue
        prev_gap = n["start"] - merged[i - 1]["end"] if i > 0 else 1e9
        next_gap = merged[i + 1]["start"] - n["end"] if i + 1 < len(merged) else 1e9
        if min(prev_gap, next_gap) < 1.0:
            out.append(n)
    return out


def _salience_ratio(n: dict, sal: dict) -> float:
    """How much of the stem's own CQT energy sits at this note's pitch (+2·3
    harmonics) vs everything sounding then. 0 (buried) .. ~1 (dominant)."""
    try:
        C, ct = sal["C"], sal["t"]
        nb, nf = C.shape
        i0 = min(int(np.searchsorted(ct, n["start"])), nf - 1)
        i1 = min(max(i0 + 1, int(np.searchsorted(ct, n["end"]))), nf)
        seg = C[:, i0:i1]
        if seg.size == 0:
            return 0.5
        base = int(round(n["pitch"])) - _SAL_BASE_MIDI
        rows = [r for r in (base, base + 12, base + 19) if 0 <= r < nb]
        local = float(np.median(seg[rows].max(axis=0))) if rows else 0.0
        colmax = float(np.median(seg.max(axis=0))) + 1e-9
        return _clip01(local / colmax)
    except Exception:
        return 0.5


_ENV_PTS = 10   # per-note amplitude-envelope samples


def instrument_brightness(notes: list[dict], sal: dict, sample: int = 24) -> float | None:
    """0..1: how much of a part's own-pitch energy sits in its 2nd-4th
    harmonics rather than the fundamental, read from the mix's own CQT at
    each note's own onset — a real, per-song measurement of whether THIS
    instrument sounds bright/rich or dull/round here, not a fixed preset per
    instrument family. Uses the same fundamental+2nd+3rd(+4th) CQT-row
    convention as _note_dynamics' loudness read, just as a ratio instead of
    a sum.

    Aggregated (median) over a sample of the part's own notes for stability:
    a single note's read is noisy, and in a dense mix is sometimes
    contaminated by another simultaneous instrument at a harmonically
    related pitch — the same caveat _mt3_octaves' per-part (not per-note)
    design already lives with.
    """
    C, ct = sal.get("C"), sal.get("t")
    if C is None or ct is None or not notes:
        return None
    nb, nf = C.shape
    pick = notes if len(notes) <= sample else [
        notes[i] for i in np.linspace(0, len(notes) - 1, sample).round().astype(int)]
    ratios = []
    for n in pick:
        i0 = min(int(np.searchsorted(ct, float(n["start"]))), nf - 1)
        i1 = min(max(i0 + 1, int(np.searchsorted(ct, float(n["end"])))), nf)
        base = int(round(n["pitch"])) - _SAL_BASE_MIDI
        if not (0 <= base < nb):
            continue
        fund = float(C[base, i0:i1].mean())
        harm_rows = [r for r in (base + 12, base + 19, base + 24) if 0 <= r < nb]
        harm = float(C[harm_rows, i0:i1].mean()) if harm_rows else 0.0
        total = fund + harm
        if total > 1e-9:
            ratios.append(harm / total)
    return float(np.median(ratios)) if ratios else None


def drum_hit_profile(y: np.ndarray, sr: int, notes: list[dict],
                     sample: int = 8) -> dict[str, dict]:
    """{str(pitch): {centroid_hz, decay_s}} measured from real onsets in the
    mix, per GM kit piece — so a kick/snare/hihat's synthesised shape follows
    what THIS recording's kit actually sounds like (bright/tight vs boomy/
    loose) instead of one fixed preset. Unlike instrument_brightness this
    reads the raw waveform, not the CQT: a drum hit is mostly noise, which a
    constant-Q *pitch* salience map has nothing meaningful to say about.

    Keyed by the STRING form of the pitch, not the int: this dict rides
    through job persistence (json.dumps/loads), which is happy to write an
    int dict key but always reads it back as a string — an int-keyed dict
    would silently stop matching after a restart.
    """
    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(int(n["pitch"]), []).append(n)
    out: dict[str, dict] = {}
    win = int(0.15 * sr)
    for pitch, ns in by_pitch.items():
        pick = ns if len(ns) <= sample else [
            ns[i] for i in np.linspace(0, len(ns) - 1, sample).round().astype(int)]
        centroids, decays = [], []
        for n in pick:
            i0 = int(round(float(n["start"]) * sr))
            seg = y[i0:i0 + win]
            if seg.size < sr // 50:
                continue
            try:
                cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
            except Exception:
                continue
            # frame_length=512/hop_length=128 (~6ms) for a hit that can decay
            # in under 50ms; center=False so frame 0 is the onset itself, not
            # a padded window straddling silence before it — with the default
            # center=True every hit measured decay_s as the exact same
            # constant (the segment's own frame count), because frame 0's
            # near-silent padding made "half of frame 0" a threshold nothing
            # in the real hit ever fell back below.
            rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=128,
                                      center=False)[0]
            if rms.size < 2 or rms[0] <= 1e-9:
                continue
            below = np.where(rms <= rms[0] * 0.5)[0]
            decay_frames = int(below[0]) if below.size else rms.size
            decays.append(decay_frames * 128 / sr)
            centroids.append(cent)
        if centroids:
            out[str(pitch)] = {"centroid_hz": round(float(np.median(centroids)), 1),
                              "decay_s": round(float(np.median(decays)), 3)}
    return out


def drum_hit_sample(y: np.ndarray, sr: int, notes: list[dict], sample: int = 8,
                    clip_s: float = 0.35) -> dict[str, str]:
    """One real audio clip per GM kit piece — base64 WAV, picked from this
    recording's own cleanest onset of that piece — so playback IS this song's
    own kick/snare/hihat, not a synthesised guess at one. Measuring timbre
    (drum_hit_profile above) and then re-synthesising from the measurement
    both lose information a direct clip doesn't: this replaces that guess
    with the real thing wherever a clean-enough onset exists.

    "Cleanest" = the onset with the highest ratio of its own immediate energy
    to whatever was already sounding just before it. MT3 mode has no isolated
    drum audio, so an onset landing on top of a lot of already-present
    content is more likely to hand back a clip dominated by other
    instruments, not the drum itself — this picks the least contaminated
    instance available rather than just the first or the loudest.
    """
    import base64
    import io
    import soundfile as sf

    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(int(n["pitch"]), []).append(n)
    clip_len = int(clip_s * sr)
    pre_len = int(0.03 * sr)
    post_len = min(clip_len, int(0.05 * sr))
    out: dict[str, str] = {}
    for pitch, ns in by_pitch.items():
        pick = ns if len(ns) <= sample else [
            ns[i] for i in np.linspace(0, len(ns) - 1, sample).round().astype(int)]
        best_i0, best_score = None, -1.0
        for n in pick:
            i0 = int(round(float(n["start"]) * sr))
            if i0 < pre_len or i0 + clip_len > len(y):
                continue
            pre = y[i0 - pre_len:i0]
            post = y[i0:i0 + post_len]
            pre_e = float(np.sqrt(np.mean(pre ** 2))) + 1e-6
            post_e = float(np.sqrt(np.mean(post ** 2))) + 1e-6
            score = post_e / pre_e
            if score > best_score:
                best_score, best_i0 = score, i0
        if best_i0 is None:
            continue
        clip = y[best_i0:best_i0 + clip_len].copy()
        # A short fade at each edge so the clip itself doesn't click on loop-in
        # or cut-off — this is about the SPLICE, not the drum's own decay.
        fade = min(int(0.005 * sr), clip.size // 4)
        if fade > 1:
            clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=clip.dtype)
            clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=clip.dtype)
        buf = io.BytesIO()
        sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
        out[str(pitch)] = base64.b64encode(buf.getvalue()).decode("ascii")
    return out


def _note_dynamics(notes: list[dict], a: dict, field: str = "velocity",
                   override: bool | None = None) -> list[dict]:
    """Give every note a real loudness + a per-note amplitude envelope, read from
    its OWN constant-Q rows (fundamental + 2·3rd harmonics) so simultaneous notes
    stay separated. ``field`` becomes relative loudness; ``env`` is a 10-point
    0-127 shape (own-peak normalised) → the player renders decay / swell / accent
    instead of a flat tone. Melody engines (fake velocity 90) get a full
    override; polyphonic engines keep a blend of the model's own velocity.

    ``field`` exists because the MT3 path must not overwrite ``velocity``: that
    field is what mt3_post.gate() thresholds on, so writing a real loudness into
    it would silently turn the sensitivity slider into a "delete quiet notes"
    control. MT3 notes get the measurement in ``dyn`` instead, and ``override``
    forces a full replacement because MT3's own velocity is the constant 100 and
    carries nothing to blend with."""
    sal = a.get("_sal")
    if not sal or not notes:
        return notes
    try:
        C, ct = sal["C"], sal["t"]
        nb, nf = C.shape
        is_mel = (a.get("kind") == "melody") if override is None else bool(override)
        segs, peaks = [], []
        for n in notes:
            i0 = min(int(np.searchsorted(ct, n["start"])), nf - 1)
            i1 = min(max(i0 + 1, int(np.searchsorted(ct, n["end"]))), nf)
            base = int(round(n["pitch"])) - _SAL_BASE_MIDI
            rows = [r for r in (base, base + 12, base + 19) if 0 <= r < nb]
            seg = C[rows, i0:i1].sum(axis=0) if rows else np.zeros(1, dtype=float)
            if seg.size == 0:
                seg = np.zeros(1, dtype=float)
            segs.append(np.asarray(seg, dtype=float))
            peaks.append(float(seg.max()))
        gpeak = max(peaks) or 1.0
        xq = np.linspace(0.0, 1.0, _ENV_PTS)
        for n, seg, pk in zip(notes, segs, peaks):
            loud = (pk / gpeak) ** 0.5                          # perceptual-ish
            v_meas = int(max(6, min(127, round(16 + 111 * loud))))
            cur = int(n.get(field, n.get("velocity", 90)))
            n[field] = v_meas if is_mel else int(round(0.6 * v_meas + 0.4 * cur))
            m = seg.max() or 1.0
            xs = np.linspace(0.0, 1.0, seg.size) if seg.size > 1 else np.array([0.0, 1.0])
            ys = (seg / m) if seg.size > 1 else np.array([1.0, 1.0])
            env = np.interp(xq, xs, ys)
            n["env"] = [int(round(float(x) * 127)) for x in np.clip(env, 0.0, 1.0)]
    except Exception:
        pass
    return notes


def _annotate_confidence(notes: list[dict], a: dict, mode: str) -> list[dict]:
    """Stamp each note with ``conf`` (0..1): blend of the detector's own
    confidence over the note and its harmonic salience. Drives the "uncertain
    note" highlighting in the UI — never removes anything."""
    if not notes:
        return notes
    sal = a.get("_sal")
    nmf = a.get("_nmf")
    eng = a.get("engine")
    times = a.get("times")
    vprob = a.get("vprob")
    note_pg = None
    if eng == "basic-pitch":
        try:
            note_pg = np.asarray(a["model_output"]["note"])   # (frames, 88)
            fps = float(a["fps"])
        except Exception:
            note_pg = None

    for n in notes:
        det = None
        if mode == "melody" and times is not None and vprob is not None:
            i0 = int(np.searchsorted(times, n["start"]))
            i1 = max(i0 + 1, int(np.searchsorted(times, n["end"])))
            seg = np.asarray(vprob)[i0:i1]
            if seg.size:
                det = float(np.clip(seg.mean(), 0.0, 1.0))
        elif note_pg is not None:
            f0 = int(n["start"] * fps)
            f1 = max(f0 + 1, int(n["end"] * fps))
            pi = int(round(n["pitch"])) - 21
            if 0 <= pi < note_pg.shape[1]:
                seg = note_pg[f0:f1, pi]
                if seg.size:
                    det = float(np.clip(seg.mean(), 0.0, 1.0))
        if det is None:
            det = _clip01(n.get("velocity", 90) / 110.0)

        sr_ = _salience_ratio(n, sal) if sal is not None else None
        nm_ = _nmf_support(n, nmf) if nmf is not None else None
        if nm_ is not None and sr_ is not None:
            conf = 0.45 * det + 0.30 * sr_ + 0.25 * nm_
        elif sr_ is not None:
            conf = 0.6 * det + 0.4 * sr_
        else:
            conf = det
        if n.get("_nmf_added"):          # NMF-only recoveries start a bit unsure
            conf = min(conf, 0.6)
        n["conf"] = round(_clip01(conf), 2)
    return notes


def _poly_contour(notes: list[dict]) -> list[dict]:
    if not notes:
        return []
    end = max(n["end"] for n in notes)
    out = []
    for t in np.arange(0.0, end, 0.05):
        active = [n["pitch"] for n in notes if n["start"] <= t < n["end"]]
        if active:
            top = max(active)
            out.append({"t": round(float(t), 3),
                        "freq": round(float(librosa.midi_to_hz(top)), 2),
                        "midi": float(top)})
    return out


# --------------------------------------------------------------------------- #
# Engine 1: monophonic melody (pYIN)
# --------------------------------------------------------------------------- #
def analyze_melody(y: np.ndarray, sr: int,
                   fmin: float | None = None, fmax: float | None = None,
                   instrument_hint: str | None = None) -> dict[str, Any]:
    fmin = float(fmin) if fmin else librosa.note_to_hz("C2")
    fmax = float(fmax) if fmax else librosa.note_to_hz("C7")
    # low fundamentals need a longer analysis window for frequency resolution
    frame_length = 4096 if fmin < 60 else 2048
    f0, _vflag, vprob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=HOP,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)

    try:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        tuning = 0.0

    good = np.isfinite(f0) & (f0 > 0)
    midi_cont = np.full(f0.shape, np.nan)
    midi_cont[good] = librosa.hz_to_midi(f0[good]) - tuning

    return {
        "kind": "melody",
        "engine": "pyin",
        "duration": round(float(len(y) / sr), 3),
        "tempo": round(estimate_tempo(y, sr), 1),
        "tuning": round(tuning, 3),
        "times": times,
        "vprob": np.nan_to_num(vprob, nan=0.0),
        "midi_cont": midi_cont,      # tuning-corrected, UN-smoothed (smoothed per preset)
        "onset_t": _onsets(y, sr),
        "beats": _beats(y, sr),
        "_sal": _salience_cqt(y, sr),
        "instrument": _hint_or_classify(y, sr, instrument_hint),
    }


def _hint_or_classify(y: np.ndarray, sr: int, hint: str | None) -> dict:
    if hint:
        return {"family": hint, "label": _INSTRUMENT_LABEL.get(hint, hint),
                "confidence": 1.0, "features": {}}
    return classify_instrument(y, sr)


def _yin_octave_correct(f0: np.ndarray, y16: np.ndarray, sr16: int,
                        fmin: float, fmax: float) -> np.ndarray:
    """Catch CREPE octave errors with a cheap YIN estimate: where CREPE sits a
    clean octave off YIN for ≥5 consecutive frames, snap CREPE to YIN's octave.
    The run-length requirement makes it robust to YIN's own single-frame slips."""
    try:
        yf = librosa.yin(y16, fmin=max(float(fmin), 40.0),
                         fmax=min(float(fmax), sr16 / 2.0 - 50.0),
                         sr=sr16, hop_length=_CREPE_HOP)
        m = min(len(f0), len(yf))
        f0c = np.asarray(f0[:m], dtype=float).copy()
        yf = np.asarray(yf[:m], dtype=float)
        ok = np.isfinite(f0c) & (f0c > 0) & np.isfinite(yf) & (yf > 0)
        ratio = np.ones(m)
        ratio[ok] = f0c[ok] / yf[ok]
        hi = ok & (np.abs(ratio - 2.0) < 0.20)     # CREPE an octave high
        lo = ok & (np.abs(ratio - 0.5) < 0.05)     # CREPE an octave low

        def run_gate(mask: np.ndarray, k: int = 5) -> np.ndarray:
            out = np.zeros_like(mask)
            c = 0
            for i, v in enumerate(mask):
                c = c + 1 if v else 0
                if c >= k:
                    out[i - k + 1:i + 1] = True
            return out

        f0c[run_gate(hi)] /= 2.0
        f0c[run_gate(lo)] *= 2.0
        return np.concatenate([f0c, f0[m:]]) if m < len(f0) else f0c
    except Exception:
        return f0


def analyze_melody_crepe(y: np.ndarray, sr: int,
                         fmin: float | None = None, fmax: float | None = None,
                         instrument_hint: str | None = None,
                         hq: bool = False) -> dict[str, Any]:
    """Monophonic pitch via torchcrepe (a trained CNN) — much steadier than pYIN
    on octave errors and noise. Same analysis-dict shape as analyze_melody.
    ``hq`` allows the (slow) 'full' model on short clips."""
    import torch
    import torchcrepe

    fmin = max(float(fmin) if fmin else librosa.note_to_hz("C2"), 32.7)
    fmax = min(float(fmax) if fmax else librosa.note_to_hz("C7"), 1975.5)

    y16 = librosa.resample(y, orig_sr=sr, target_sr=_CREPE_SR) if sr != _CREPE_SR else y
    audio = torch.from_numpy(np.ascontiguousarray(y16, dtype=np.float32))[None]

    dur = len(y16) / _CREPE_SR
    model = _CREPE_MODEL
    if model not in ("tiny", "full"):     # 'auto': full only for an explicitly HQ short lead
        model = "full" if (hq and dur <= _CREPE_FULL_MAX_SEC) else "tiny"
    # viterbi on 'full' is pathologically slow here — force the fast decoder
    dec_name = "weighted_argmax" if model == "full" else _CREPE_DECODER
    decoder = getattr(torchcrepe.decode, dec_name, torchcrepe.decode.viterbi)
    bs = 32 if model == "full" else (512 if dec_name == "viterbi" else 256)

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    f0_t, per_t = torchcrepe.predict(
        audio, _CREPE_SR, hop_length=_CREPE_HOP, fmin=fmin, fmax=fmax,
        model=model, batch_size=bs, device="cpu",
        decoder=decoder, return_periodicity=True, pad=True)
    per_t = torchcrepe.filter.median(per_t, 3)
    f0_t = torchcrepe.filter.median(f0_t, 3)
    f0 = f0_t[0].cpu().numpy().astype(float)
    per = per_t[0].cpu().numpy().astype(float)
    f0 = _yin_octave_correct(f0, y16, _CREPE_SR, fmin, fmax)

    n = f0.shape[0]
    times = np.arange(n) * (_CREPE_HOP / _CREPE_SR)
    try:
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        tuning = 0.0
    good = np.isfinite(f0) & (f0 > 0)
    midi_cont = np.full(n, np.nan)
    midi_cont[good] = librosa.hz_to_midi(f0[good]) - tuning

    return {
        "kind": "melody",
        "engine": "crepe",
        "crepe_model": model,
        "crepe_decoder": dec_name,
        "duration": round(float(len(y) / sr), 3),
        "tempo": round(estimate_tempo(y, sr), 1),
        "tuning": round(tuning, 3),
        "times": times,
        "vprob": np.nan_to_num(per, nan=0.0),
        "midi_cont": midi_cont,
        "onset_t": _onsets(y, sr),
        "beats": _beats(y, sr),
        "_sal": _salience_cqt(y, sr),
        "instrument": _hint_or_classify(y, sr, instrument_hint),
    }


def segment_melody(a: dict[str, Any], sensitivity: float,
                   instrument: str | None = None) -> tuple[list[dict], list[dict], str]:
    s = _clip01(sensitivity)
    pk = _preset_key(instrument, a.get("instrument", {}).get("family", "other"))
    P = _MEL_PRESETS[pk]

    # crepe periodicity runs lower than pYIN voiced-prob for the same signal
    vbase = 0.50 if a.get("engine") == "crepe" else 0.70
    vthr = vbase - (vbase - 0.08) * s
    min_dur = P["min_dur"] * (1.5 - 1.0 * s)      # preset base, then sensitivity
    max_gap = P["gap"] + 0.03 * s

    m = _smooth_midi_track(a["midi_cont"], P["k_short"], P["k_long"])
    # Schmitt hysteresis: a note may only START where confidence clears a higher
    # bar, but SUSTAINS down to the lower one — fewer spurious blips, fewer
    # mid-note dropouts.
    per = np.asarray(a["vprob"], dtype=float)
    start_thr = min(0.97, vthr * 1.4 + 0.02)
    voiced = np.zeros(per.shape, dtype=np.int8)
    voiced[per >= vthr] = 1
    voiced[per >= start_thr] = 2
    midi_int = np.where(np.isfinite(m) & (m > 0), np.round(m), 0.0)

    notes = _segment(midi_int, a["times"], voiced, a["onset_t"],
                     min_dur, max_gap, onset_bias=P["onset_bias"])

    # a melody is one line: drop notes far outside its tessitura (separation
    # bleed leaves stray very-low/high notes)
    if len(notes) >= 5:
        med = float(np.median([n["pitch"] for n in notes]))
        notes = [n for n in notes if abs(n["pitch"] - med) <= 15]

    contour = []
    for i in range(0, len(m), 3):
        v = m[i]
        if np.isfinite(v) and v > 0:
            contour.append({"t": round(float(a["times"][i]), 3),
                            "freq": round(float(librosa.midi_to_hz(v)), 2),
                            "midi": round(float(v), 2)})
    return notes, contour, pk


# --------------------------------------------------------------------------- #
# Engine 2a: polyphonic via basic-pitch (optional dependency)
# --------------------------------------------------------------------------- #
def _has_basic_pitch() -> bool:
    try:
        import basic_pitch  # noqa: F401
        import basic_pitch.inference  # noqa: F401
        return True
    except Exception:
        return False


def analyze_basic_pitch(path: str, y: np.ndarray, sr: int) -> dict[str, Any]:
    from basic_pitch.inference import predict, AUDIO_SAMPLE_RATE, FFT_HOP
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH as MODEL
    except Exception:
        MODEL = None

    kwargs = {}
    if MODEL is not None:
        kwargs["model_or_model_path"] = MODEL
    model_output, _midi_data, _note_events = predict(path, **kwargs)

    # cache the raw posteriorgrams so the sensitivity knob can re-threshold
    # them without another NN pass (model_output_to_notes needs all three keys).
    mo = {k: model_output[k] for k in ("note", "onset", "contour")}
    sal = _salience_cqt(y, sr)
    return {
        "kind": "poly",
        "engine": "basic-pitch",
        "duration": round(float(len(y) / sr), 3),
        "tempo": round(estimate_tempo(y, sr), 1),
        "model_output": mo,
        "fps": AUDIO_SAMPLE_RATE / FFT_HOP,
        "beats": _beats(y, sr),
        "_sal": sal,
        "_nmf": _harmonic_nmf(sal["C"], sal["t"]) if sal else None,
        "instrument": classify_instrument(y, sr),
    }


# --------------------------------------------------------------------------- #
# Engine 2c: dedicated piano transcription (Kong et al.) — for keyboard stems
# --------------------------------------------------------------------------- #
_piano_tx = None
_piano_lock = threading.Lock()


def _get_piano_tx():
    global _piano_tx
    if _piano_tx is None:
        with _piano_lock:
            if _piano_tx is None:
                from piano_transcription_inference import PianoTranscription
                _piano_tx = PianoTranscription(device="cpu")
    return _piano_tx


def analyze_piano(path: str) -> dict[str, Any]:
    import tempfile
    try:
        from piano_transcription_inference import sample_rate as PT_SR
    except Exception:
        PT_SR = 16000

    # their bundled load_audio breaks on librosa >=0.10 — load it ourselves
    y = librosa.load(path, sr=PT_SR, mono=True)[0].astype("float32")
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        out = _get_piano_tx().transcribe(y, tmp.name)
    events = out.get("est_note_events", []) if isinstance(out, dict) else []
    raw = []
    for ev in events:
        st = float(ev.get("onset_time", ev.get("onset", 0.0)))
        en = float(ev.get("offset_time", ev.get("offset", st + 0.1)))
        p = int(ev.get("midi_note", ev.get("pitch", 60)))
        v = float(ev.get("velocity", 100)) / 127.0
        raw.append((st, en, p, max(0.05, min(1.0, v))))
    raw.sort(key=lambda r: (r[0], r[2]))
    return {
        "kind": "poly",
        "engine": "piano-tx",
        "duration": round(float(len(y) / PT_SR), 3),
        "tempo": round(estimate_tempo(y, PT_SR), 1),
        "raw": raw,
        "beats": _beats(y, PT_SR),
        "_sal": _salience_cqt(y, PT_SR),
        "instrument": {"family": "keyboard", "label": "피아노 (전용 모델)",
                       "confidence": 1.0, "features": {}},
    }


def segment_raw_notes(a: dict[str, Any], sensitivity: float,
                      instrument: str | None = None) -> list[dict]:
    """Post-filter for engines that hand back finished note events (piano-tx).

    MEASURED 2026-08-30: the old thresholds here (min_len 77 ms, amp 0.175 at
    s=0.5) threw away 75 % of the model's notes and dropped note F1 from 0.913
    to 0.363 on the MAESTRO reference set. A trained note-level model has
    already decided what a note is — do NOT second-guess it. The knob now only
    trims at genuinely low sensitivity, and octave "doublings" are kept because
    real piano writing is full of genuine octaves.
    """
    s = _clip01(sensitivity)
    amp_thr = 0.12 * (1.0 - s) ** 2          # ~0.03 at s=0.5, 0 at s=1
    min_len = 0.030 * (1.0 - s)              # ~15 ms at s=0.5, 0 at s=1
    notes = []
    for st, en, p, amp in a["raw"]:
        if en - st < min_len or amp < amp_thr:
            continue
        notes.append({
            "start": round(st, 3), "end": round(en, 3), "pitch": p,
            "name": midi_to_name(p),
            "freq": round(float(librosa.midi_to_hz(p)), 2),
            "velocity": int(max(1, min(127, round(amp * 127)))),
        })
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def _drop_octave_doublings(notes: list[dict]) -> list[dict]:
    """Drop a note that is exactly +12 above a louder note it mostly overlaps."""
    keep = [True] * len(notes)
    for i, hi in enumerate(notes):
        for lo in notes:
            if lo is hi or lo["pitch"] != hi["pitch"] - 12:
                continue
            ov = min(hi["end"], lo["end"]) - max(hi["start"], lo["start"])
            dur = hi["end"] - hi["start"]
            if dur > 0 and ov / dur > 0.8 and hi["velocity"] < lo["velocity"] * 0.6:
                keep[i] = False
                break
    return [n for n, k in zip(notes, keep) if k]


def segment_basic_pitch(a: dict[str, Any], sensitivity: float,
                        instrument: str | None = None) -> list[dict]:
    """Re-threshold basic-pitch's cached posteriorgrams. Low sensitivity =
    high thresholds (only confident notes); high = low thresholds (more notes).
    The instrument preset shifts the thresholds and minimum note length."""
    from basic_pitch.note_creation import model_output_to_notes

    s = _clip01(sensitivity)
    pk = _preset_key(instrument, a.get("instrument", {}).get("family", "other"))
    P = _BP_PRESETS[pk]

    onset_thr = min(0.95, max(0.05, 0.75 - 0.52 * s + P["onset_d"]))
    frame_thr = min(0.95, max(0.05, 0.55 - 0.44 * s + P["frame_d"]))
    min_len_ms = max(30.0, (P["min_ms"] + 32.0) - 118.0 * s)
    min_len_frames = max(1, int(round(min_len_ms / 1000.0 * a["fps"])))

    _pm, note_events = model_output_to_notes(
        a["model_output"], onset_thresh=onset_thr, frame_thresh=frame_thr,
        infer_onsets=True, min_note_len=min_len_frames,
        include_pitch_bends=False, melodia_trick=P["melodia"],
    )

    notes = []
    for ev in note_events:
        start, end, pitch, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        notes.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "pitch": pitch,
            "name": midi_to_name(pitch),
            "freq": round(float(librosa.midi_to_hz(pitch)), 2),
            "velocity": int(max(1, min(127, round(amp * 127)))),
        })
    notes = _drop_octave_doublings(notes)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


# --------------------------------------------------------------------------- #
# Engine 2b: polyphonic fallback -- CQT peak picking
# --------------------------------------------------------------------------- #
def analyze_cqt(y: np.ndarray, sr: int) -> dict[str, Any]:
    hop = 512
    fmin = librosa.note_to_hz("C2")
    n_bins = 60
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin,
                           n_bins=n_bins, bins_per_octave=12))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
    return {
        "kind": "poly",
        "engine": "cqt-fallback",
        "duration": round(float(len(y) / sr), 3),
        "tempo": round(estimate_tempo(y, sr), 1),
        "C": C, "C_db": C_db, "times": times, "hop": hop,
        "base_midi": int(round(librosa.hz_to_midi(fmin))), "n_bins": n_bins,
        "sr": sr,
        "_sal": (_sal := _salience_cqt(y, sr)),
        "_nmf": _harmonic_nmf(_sal["C"], _sal["t"]) if _sal else None,
        "instrument": classify_instrument(y, sr),
    }


def segment_cqt(a: dict[str, Any], sensitivity: float,
                instrument: str | None = None) -> list[dict]:
    s = _clip01(sensitivity)
    C, C_db, times = a["C"], a["C_db"], a["times"]
    base_midi, n_bins, hop, sr = a["base_midi"], a["n_bins"], a["hop"], a["sr"]

    pk = _preset_key(instrument, a.get("instrument", {}).get("family", "other"))
    min_scale = {"struck": 0.6, "sustained": 1.3, "percussive": 0.5, "neutral": 1.0}[pk]

    drop_db = 10.0 + 22.0 * (1.0 - s)      # within 10..32 dB of the frame peak
    floor_db = -38.0 - 12.0 * s
    min_frames = max(1, int((0.12 - 0.08 * s) * min_scale * sr / hop))

    active = np.zeros_like(C_db, dtype=bool)
    for j in range(C_db.shape[1]):
        col = C_db[:, j]
        thr = max(col.max() - drop_db, floor_db)
        active[:, j] = col >= thr

    notes = []
    for row in range(n_bins):
        pitch = base_midi + row
        on = active[row]
        i = 0
        while i < len(on):
            if on[i]:
                k = i
                while k < len(on) and on[k]:
                    k += 1
                if k - i >= min_frames:
                    seg = C[row, i:k]
                    vel = int(max(1, min(127, round(
                        (seg.mean() / (C.max() + 1e-9)) * 127))))
                    notes.append({
                        "start": round(float(times[i]), 3),
                        "end": round(float(times[min(k, len(times) - 1)]), 3),
                        "pitch": pitch,
                        "name": midi_to_name(pitch),
                        "freq": round(float(librosa.midi_to_hz(pitch)), 2),
                        "velocity": vel,
                    })
                i = k
            else:
                i += 1
    notes = _drop_octave_doublings(notes)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def analyze(path: str, mode: str = "melody",
            fmin: float | None = None, fmax: float | None = None,
            instrument_hint: str | None = None,
            family: str | None = None, hq: bool = False) -> dict[str, Any]:
    """Run the heavy analysis once. Returns a cache dict (holds numpy arrays;
    not JSON-serialisable). Feed it to :func:`refine`.

    Engine selection:
      * melody       -> torchcrepe (CNN) if available, else pYIN
      * polyphonic + family "keyboard" -> dedicated piano model if available
      * polyphonic   -> basic-pitch, else CQT fallback
    ``fmin``/``fmax``/``instrument_hint`` apply to melody only. ``hq`` lets a
    short melody-lead stem use CREPE 'full' instead of 'tiny'."""
    y, sr = load_audio(path)
    if mode == "polyphonic":
        if family == "keyboard" and _has_piano_model():
            try:
                return analyze_piano(path)
            except Exception as e:  # pragma: no cover
                pass  # fall through to basic-pitch
        if _has_basic_pitch():
            try:
                return analyze_basic_pitch(path, y, sr)
            except Exception as e:  # pragma: no cover - defensive
                a = analyze_cqt(y, sr)
                a["warning"] = f"basic-pitch 실패, 대체 엔진 사용: {e}"
                return a
        a = analyze_cqt(y, sr)
        a["warning"] = ("basic-pitch 미설치 -- CQT 근사 엔진 사용. "
                        "정확도를 높이려면 setup.sh 로 basic-pitch 를 설치하세요.")
        return a

    if _has_crepe():
        try:
            return analyze_melody_crepe(y, sr, fmin=fmin, fmax=fmax,
                                        instrument_hint=instrument_hint, hq=hq)
        except Exception:  # pragma: no cover
            pass
    return analyze_melody(y, sr, fmin=fmin, fmax=fmax, instrument_hint=instrument_hint)


def refine(a: dict[str, Any], sensitivity: float = DEFAULT_SENSITIVITY,
           instrument: str | None = None, quantize: bool = False) -> dict[str, Any]:
    """Cheap: re-segment a cached analysis at a new sensitivity / instrument.
    ``quantize`` snaps note times to the estimated beat grid."""
    s = _clip01(sensitivity)
    detected = a.get("instrument", {}).get("family", "other")
    preset_key = _preset_key(instrument, detected)

    if a["kind"] == "melody":
        notes, contour, preset_key = segment_melody(a, s, instrument)
        mode = "melody"
    else:
        seg = {"basic-pitch": segment_basic_pitch,
               "piano-tx": segment_raw_notes}.get(a["engine"], segment_cqt)
        notes = seg(a, s, instrument)
        mode = "polyphonic"
        # second opinion from the harmonic NMF (basic-pitch / CQT fallback only)
        if a.get("_nmf") and a["engine"] in ("basic-pitch", "cqt-fallback"):
            notes = _nmf_ensemble(notes, a, s)

    # bleed / ghost-note gate (Demucs stems only; app sets a["gate"])
    if a.get("gate"):
        notes = _gate_notes(notes, a, s)

    # A dedicated piano transcription model already emits note events.  The
    # generic cleanup merges close repeated pitches and snaps timing, both of
    # which erase real piano re-attacks and offsets.  Keep its events intact.
    if a["engine"] != "piano-tx":
        notes = _musical_cleanup(notes, a, s, mono=(mode == "melody"))
    notes = _note_dynamics(notes, a)              # real per-note loudness + envelope
    notes = _annotate_confidence(notes, a, mode)

    if mode == "polyphonic":
        contour = _poly_contour(notes)

    if quantize and a.get("beats"):
        notes = quantize_notes(notes, a["beats"])

    key = a.get("key") or {}
    _TONIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_name = (f"{_TONIC[key['tonic']]} {key['mode']}"
                if key.get("tonic") is not None and key.get("strength", 0) >= 0.55
                else None)

    res: dict[str, Any] = {
        "engine": a["engine"],
        "mode": mode,
        "duration": a["duration"],
        "tempo": a["tempo"],
        "sensitivity": round(s, 3),
        "quantized": bool(quantize and a.get("beats")),
        "beat_count": len(a.get("beats", [])),
        "beats": list(a.get("beats", [])),
        "key": key_name,
        "key_strength": key.get("strength", 0.0),
        "low_conf": sum(1 for n in notes if n.get("conf", 1.0) < 0.5),
        "instrument": _instrument_block(a, instrument, preset_key),
        "notes": notes,
        "contour": contour,
    }
    if a.get("warning"):
        res["warning"] = a["warning"]
    if a.get("tuning"):
        res["tuning"] = a["tuning"]
    return res


def transcribe(path: str, mode: str = "melody",
               sensitivity: float = DEFAULT_SENSITIVITY,
               instrument: str | None = None, quantize: bool = False):
    """Full run. Returns ``(result_dict, analysis_cache)``."""
    a = analyze(path, mode, hq=(mode == "melody"))
    return refine(a, sensitivity, instrument, quantize), a


def notes_to_midi(notes: list[dict], tempo: float = 120.0,
                  program: int = 0, name: str = "MusicNote",
                  is_drum: bool = False) -> "pretty_midi.PrettyMIDI":
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo or 120.0)
    safe_name = (name or "MusicNote").encode("ascii", "replace").decode()  # mido track name is latin-1
    inst = pretty_midi.Instrument(program=int(program), name=safe_name, is_drum=is_drum)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=int(n.get("velocity", 90)),
            pitch=int(n["pitch"]),
            start=float(n["start"]),
            end=float(max(n["end"], n["start"] + 0.05)),
        ))
    pm.instruments.append(inst)
    return pm
