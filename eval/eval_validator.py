#!/usr/bin/env python3
"""Measure the missed-note validator in backend/quality.py against ground truth.

The CQT thresholds in `missing_onset_candidates` were derived from one YouTube
clip, and nothing has ever checked how often they are right. eval/refs has
ground-truth MIDI, so the validator's own precision and recall are measurable:

  true miss  = a ground-truth note the delivered transcription does NOT contain
  precision  = of the notes the validator reports, how many are true misses
  recall     = of the true misses, how many the validator finds

Run after eval/mt3_cache.py:

    python eval/eval_validator.py eval/refs
    python eval/eval_validator.py eval/refs --stage candidates   # skip 2nd pass
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import mt3_post as MP  # noqa: E402
import quality as Q  # noqa: E402

ONSET_TOL = 0.05
PITCH_TOL = 1  # semitones; the validator itself allows a 1-bin CQT error


def hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _arrays(triples):
    iv = np.array([[a, b] for a, b, _ in triples], float) if triples else np.zeros((0, 2))
    p = np.array([hz(m) for _, _, m in triples], float) if triples else np.zeros(0)
    return iv, p


def true_misses(ref: list, est: list) -> list:
    """Ground-truth notes with no matching delivered note."""
    import mir_eval
    ref_iv, ref_p = _arrays(ref)
    est_iv, est_p = _arrays(est)
    if len(est_iv) == 0:
        return list(ref)
    matched = mir_eval.transcription.match_notes(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=ONSET_TOL, pitch_tolerance=50.0, offset_ratio=None)
    hit = {i for i, _ in matched}
    return [ref[i] for i in range(len(ref)) if i not in hit]


def hits_against(reported: list, misses: list) -> tuple[int, set]:
    """Count reported notes that land on a true miss (many-to-one allowed)."""
    good, covered = 0, set()
    for c in reported:
        for k, m in enumerate(misses):
            if (abs(float(c["start"]) - m[0]) <= ONSET_TOL
                    and abs(int(c["pitch"]) - m[2]) <= PITCH_TOL):
                good += 1
                covered.add(k)
                break
    return good, covered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--stage", choices=["confirmed", "candidates"], default="confirmed",
                    help="confirmed = after the second-pass filter (what the UI would draw)")
    a = ap.parse_args()

    import pretty_midi
    tot_rep = tot_good = tot_miss = tot_cov = 0
    print(f"stage = {a.stage}\n")
    print(f"{'clip':8s} {'delivered':>9s} {'misses':>7s} {'reported':>8s} "
          f"{'right':>6s} {'prec':>6s} {'recall':>7s}")

    for mid in sorted(Path(a.refdir).glob("*.mid")):
        cache = mid.with_suffix(".mt3.json")
        wav = mid.with_suffix(".wav")
        if not cache.exists() or not wav.exists():
            continue
        pm = pretty_midi.PrettyMIDI(str(mid))
        ref = sorted((n.start, n.end, n.pitch)
                     for i in pm.instruments if not i.is_drum for n in i.notes)

        raw = json.loads(cache.read_text())["notes"]
        kept, _ = MP.gate(raw, a.sensitivity)
        est_dicts = [{"start": float(n["start"]), "end": float(n["end"]),
                      "pitch": int(n["pitch"])} for n in kept]
        est = sorted((n["start"], n["end"], n["pitch"]) for n in est_dicts)

        misses = true_misses(ref, est)
        reported = Q.missing_onset_candidates(str(wav), est_dicts, limit=40)
        if a.stage == "confirmed":
            reported = Q.confirm_candidates_with_second_pass(str(wav), reported)

        good, covered = hits_against(reported, misses)
        prec = good / len(reported) if reported else float("nan")
        rec = len(covered) / len(misses) if misses else float("nan")
        tot_rep += len(reported); tot_good += good
        tot_miss += len(misses); tot_cov += len(covered)
        print(f"{mid.stem:8s} {len(est):9d} {len(misses):7d} {len(reported):8d} "
              f"{good:6d} {prec:6.3f} {rec:7.3f}")

    if tot_rep or tot_miss:
        p = tot_good / tot_rep if tot_rep else float("nan")
        r = tot_cov / tot_miss if tot_miss else float("nan")
        print(f"\nTOTAL  reported={tot_rep}  right={tot_good}  precision={p:.3f}"
              f"   |  true misses={tot_miss}  covered={tot_cov}  recall={r:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
