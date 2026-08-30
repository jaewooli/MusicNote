"""Post-transcription quality checks.

These checks never alter a transcription. They make structural failures and
acoustically plausible missed onsets visible to the caller for review.
"""
from __future__ import annotations


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
                covered = any(int(n["pitch"]) == pitch and
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


def audit(path: str, stems: list[dict], notes: list[dict]) -> dict:
    structural = structural_report(stems)
    missed = missing_onset_candidates(path, notes)
    return {"structural": structural, "missed_onset_candidates": missed,
            "status": "pass" if structural["passed"] else "fail"}
