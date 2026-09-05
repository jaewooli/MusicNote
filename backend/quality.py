"""Post-transcription quality checks.

These checks never alter a transcription. They make structural failures and
likely missed notes visible to the caller for review.

MEASURED 2026-08-30 (eval/eval_validator.py, eval/refs):
The CQT heuristic below scored precision 0.000 and recall 0.000 — of 20 reported
candidates across two clips, none corresponded to a real omission, and it found
none of the 162 true misses. Two structural reasons: it only scans MIDI 40-84,
which excludes 44 % of actual misses, and it repeatedly fires on resonances at a
single pitch. Its second-pass confirmation also ran a whole extra piano model
(piano-tx, itself note F1 0.560 vs MT3's 0.865) inside the request path.

It is therefore disabled by default. Omission candidates now come from
disagreement between two MT3 runs at different segment offsets
(`backend/mt3_ensemble.py`), which is produced by the same model on the same
audio and so carries no independent false-positive mode.
"""
from __future__ import annotations

import os

# Opt-in only; kept so the replaced approach stays measurable rather than lost.
CQT_CANDIDATES = os.environ.get("MUSICNOTE_CQT_CANDIDATES", "0") == "1"


def _overlaps(notes: list[dict]) -> int:
    ordered = sorted(notes, key=lambda n: (float(n["start"]), float(n["end"])))
    return sum(float(b["start"]) < float(a["end"]) - 1e-6
               for a, b in zip(ordered, ordered[1:]))


def structural_report(stems: list[dict]) -> dict:
    """Report inferred sequences without treating a chord as an error."""
    sequences = [s for s in stems if s.get("voice")]
    rows = [{"id": s["id"], "sequence": s.get("voice"),
             "notes": len(s.get("notes", [])),
             "chord_overlaps": _overlaps(s.get("notes", []))}
            for s in sequences]
    return {
        # Keep old field names for clients, but overlap is informational now:
        # simultaneous notes may be a valid chord in one sequence.
        "voice_count": len(rows),
        "voice_overlaps": sum(r["chord_overlaps"] for r in rows),
        "sequence_count": len(rows),
        "sequences": rows,
        "voices": rows,
        "passed": True,
    }


def missing_onset_candidates(path: str, notes: list[dict], limit: int = 12) -> list[dict]:
    """CQT audit for strong short attacks absent from MT3.

    This is a review queue, never an auto-correction. It detects a sharp local
    energy rise at a plausible piano fundamental (E4 at 8.18 s in the Canon
    regression clip is the motivating case), then excludes already transcribed
    notes at that pitch and onset.
    """
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=22050, mono=True)
        hop, lo, hi = 256, 40, 84
        C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop,
                               fmin=librosa.midi_to_hz(lo), n_bins=hi - lo + 1,
                               bins_per_octave=12))
        A = C / (np.max(C, axis=1, keepdims=True) + 1e-9)
        times = librosa.frames_to_time(np.arange(A.shape[1]), sr=sr, hop_length=hop)
        out = []
        for row, pitch in enumerate(range(lo, hi + 1)):
            last = -1.0
            for i in range(4, A.shape[1] - 2):
                level = float(A[row, i])
                rise = level - float(np.median(A[row, i - 4:i]))
                if level < 0.58 or rise < 0.35 or rise < float(A[row, i + 1] - np.median(A[row, i - 3:i + 1])):
                    continue
                onset = float(times[i])
                if onset - last < 0.08:
                    continue
                # A CQT peak may sit one bin away from the fundamental.
                # Do not flag an already-transcribed neighbouring semitone as
                # a new omission; the E4/F#4 Canon case remains two semitones
                # apart and is therefore retained for review.
                covered = any(abs(int(n["pitch"]) - pitch) <= 1 and
                              float(n["start"]) <= onset + 0.07 and
                              float(n["end"]) >= onset - 0.07 for n in notes)
                if covered:
                    continue
                k = i + 1
                while k < A.shape[1] and A[row, k] >= 0.35 and times[k] - onset <= 0.42:
                    k += 1
                dur = float(times[min(k, len(times) - 1)] - onset)
                if dur >= 0.09:
                    out.append({"start": round(onset, 3), "end": round(onset + dur, 3),
                                "pitch": pitch, "strength": round(level, 2),
                                "rise": round(rise, 2)})
                    last = onset
        out.sort(key=lambda x: (x["strength"] + x["rise"], x["start"]), reverse=True)
        return sorted(out[:limit], key=lambda x: x["start"])
    except Exception:
        return []


def confirm_candidates_with_second_pass(path: str, candidates: list[dict]) -> list[dict]:
    """Require an independent high-recall transcription to confirm a CQT hole.

    CQT alone sees harmonics as well as fundamentals.  A dedicated piano model
    (or Basic Pitch fallback) must independently place a nearby note before the
    UI calls a candidate high-confidence.  This remains review-only.
    """
    if not candidates:
        return []
    try:
        import transcribe as T
        analysis = T.analyze(path, mode="polyphonic", family="keyboard")
        second = T.refine(analysis, sensitivity=0.9).get("notes", [])
        engine = analysis.get("engine", "second-pass")
    except Exception:
        return []
    confirmed = []
    for c in candidates:
        close = [n for n in second
                 if abs(float(n["start"]) - float(c["start"])) <= 0.10
                 and abs(int(n["pitch"]) - int(c["pitch"])) <= 1]
        if not close:
            continue
        n = min(close, key=lambda x: (abs(float(x["start"]) - float(c["start"])),
                                      abs(int(x["pitch"]) - int(c["pitch"]))))
        confirmed.append({**c, "pitch": int(n["pitch"]),
                          "end": round(float(n["end"]), 3),
                          "confirmed_by": engine})
    return confirmed


def _as_candidate(note: dict) -> dict:
    """Shape an ensemble note the way the piano roll already expects.

    ``in_score`` distinguishes "already drawn, but only one run saw it — verify"
    from "withheld — approve to add".
    """
    return {"start": round(float(note["start"]), 3),
            "end": round(float(note["end"]), 3),
            "pitch": int(note["pitch"]),
            "agreement": int(note.get("agreement", 1)),
            "in_score": bool(note.get("in_score", False)),
            "confirmed_by": "mt3-ensemble"}


# Candidates the UI will draw as ghost notes on the piano roll. Four runs at
# agreement 2 put roughly 680 notes per 25 s clip into the queue, which is more
# events than the score itself has — an unusable overlay and a large payload.
#
# The cap is a plain time-ordered prefix because there is nothing better to sort
# by. Measured over 6086 review notes on both eval sets, 11.6% of which are real
# omissions: ranking by note length gives AUC 0.535, i.e. it does not rank at
# all (the top 100 by length are 7.0% real, *below* the 11.6% base rate). If a
# per-note confidence ever becomes available, sort by it here.
MAX_REVIEW_CANDIDATES = int(os.environ.get("MUSICNOTE_MAX_REVIEW", "200"))


def audit(path: str, stems: list[dict], notes: list[dict],
          ensemble_candidates: list[dict] | None = None,
          runs: int = 1) -> dict:
    """Structural summary plus a review queue of likely omissions.

    ``ensemble_candidates`` are notes that only some MT3 runs found. They are
    never inserted into the delivered score; the UI draws them for approval.
    """
    structural = structural_report(stems)
    missed: list[dict] = []
    if CQT_CANDIDATES:
        missed = missing_onset_candidates(path, notes, limit=40)
        confirmed = confirm_candidates_with_second_pass(path, missed)
    else:
        confirmed = [_as_candidate(n) for n in (ensemble_candidates or [])]
        confirmed.sort(key=lambda c: c["start"])
    total = len(confirmed)
    confirmed = confirmed[:MAX_REVIEW_CANDIDATES]
    return {"structural": structural, "missed_onset_candidates": missed,
            "confirmed_missing_notes": confirmed,
            # the true size, so the UI never implies the list is complete
            "confirmed_missing_total": total,
            "candidate_source": "cqt" if CQT_CANDIDATES else "mt3-ensemble",
            "ensemble_runs": int(runs),
            "status": "pass" if structural["passed"] else "fail"}
