#!/usr/bin/env python3
"""
Measure beat / downbeat / tempo / time-signature detection against
eval/refs_meter (built by eval/build_meter_refs.py).

    python eval/eval_meter.py eval/refs_meter                  # current: librosa on audio
    python eval/eval_meter.py eval/refs_meter --source midi    # new detector, perfect notes
    python eval/eval_meter.py eval/refs_meter --source mt3     # new detector, MT3 notes

`--source midi` feeds the detector the ground-truth notes, which separates two
questions that are easy to confuse: is the metre algorithm wrong, or is the
transcription it was handed wrong? Fix the first before blaming the second.

Metrics:
  tempo    within 4 %. Half/double are counted separately — an octave error is
           a different bug from being lost, and gets a different fix.
  beat F1  mir_eval, 70 ms tolerance.
  down F1  same, on downbeats only. This is the one that decides whether the
           barlines land in the right place.
  ts       exact (numerator, denominator) match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def _tempo_verdict(est: float, true: float) -> str:
    if est <= 0:
        return "none"
    for name, ratio in (("ok", 1.0), ("half", 0.5), ("double", 2.0),
                        ("third", 1 / 3), ("triple", 3.0)):
        if abs(est - true * ratio) <= 0.04 * true * ratio:
            return name
    return "wrong"


def _f1(ref: list[float], est: list[float]) -> float:
    import mir_eval.beat as mb
    if not len(ref) or not len(est):
        return 0.0
    return float(mb.f_measure(mb.trim_beats(np.asarray(ref, float)),
                              mb.trim_beats(np.asarray(est, float))))


def detect(source: str, wav: Path, gt: dict) -> dict:
    """-> {tempo, beats, downbeats, time_sig}"""
    if source == "audio":
        import librosa
        import transcribe as T
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        beats, tempo = T._beat_grid(y, sr)
        # This branch is the RETIRED librosa-only path, kept for comparison. It
        # reports no downbeat and always 4/4 because it never detected either —
        # the pipeline now runs `meter.detect` over the transcribed notes, which
        # does both. Reading this branch as the product's meter accuracy makes
        # it look far worse than it is: 6/13 time signatures and downbeat F1
        # 0.000 here, against 10/13 and 0.592 on the real path.
        return {"tempo": tempo, "beats": beats, "downbeats": [], "time_sig": (4, 4)}

    import meter as M
    if source == "midi":
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(wav.with_suffix(".mid")))
        notes = [{"start": float(n.start), "end": float(n.end),
                  "pitch": int(n.pitch), "velocity": int(n.velocity)}
                 for i in pm.instruments for n in i.notes]
    else:
        cache = wav.with_suffix("").with_suffix(".mt3.json")
        if not cache.exists():
            raise SystemExit(f"no MT3 cache for {wav.name}; run eval/mt3_cache.py first")
        notes = json.loads(cache.read_text())["notes"]
    return M.detect(notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="?", default="eval/refs_meter")
    # "midi" by default: it runs the detector the product runs, on clean notes,
    # so the number is the detector's own accuracy. "mt3" adds transcription
    # error on top (and needs a cached run per clip); "audio" is the retired
    # path and is kept only for comparison.
    ap.add_argument("--source", choices=["midi", "mt3", "audio"], default="midi")
    a = ap.parse_args()

    refs = sorted(Path(a.refs).glob("*.meter.json"))
    if not refs:
        raise SystemExit(f"no *.meter.json under {a.refs}")

    rows, tallies = [], {}
    for js in refs:
        gt = json.loads(js.read_text())
        wav = js.with_name(js.name.replace(".meter.json", ".wav"))
        try:
            got = detect(a.source, wav, gt)
        except Exception as e:  # noqa: BLE001
            print(f"{wav.stem}: detect failed: {type(e).__name__}: {e}")
            continue
        ts_gt = tuple(gt["time_sig"])
        ts_est = tuple(got.get("time_sig") or (0, 0))
        # The reference stores `tempo` in quarter notes per minute, but in
        # compound time the beat is a dotted quarter — comparing a detector's
        # beat tempo against the quarter tempo marks a correct 6/8 reading wrong.
        bpm_gt = 60.0 / float(gt["beat_seconds"])
        verdict = _tempo_verdict(float(got.get("tempo") or 0), bpm_gt)
        tallies[verdict] = tallies.get(verdict, 0) + 1
        rows.append({
            "stem": wav.stem,
            "ts_gt": f"{ts_gt[0]}/{ts_gt[1]}",
            "ts_est": f"{ts_est[0]}/{ts_est[1]}",
            "ts_ok": ts_gt == ts_est,
            "bpm_gt": bpm_gt,
            "bpm_est": float(got.get("tempo") or 0),
            "tempo": verdict,
            "beat_f1": _f1(gt["beats"], got.get("beats") or []),
            "down_f1": _f1(gt["downbeats"], got.get("downbeats") or []),
        })

    print(f"\nsource: {a.source}   ({len(rows)} clips)\n")
    print(f"{'clip':<8}{'ts(gt)':>8}{'ts(est)':>9}{'bpm(gt)':>9}{'bpm(est)':>10}"
          f"{'tempo':>8}{'beatF1':>8}{'downF1':>8}")
    for r in rows:
        print(f"{r['stem']:<8}{r['ts_gt']:>8}{r['ts_est']:>9}{r['bpm_gt']:>9.0f}"
              f"{r['bpm_est']:>10.1f}{r['tempo']:>8}{r['beat_f1']:>8.3f}{r['down_f1']:>8.3f}")
    n = len(rows) or 1
    print(f"\n{'':<8}{'':>8}{sum(r['ts_ok'] for r in rows):>4}/{len(rows):<4}"
          f"{'':>19}{'':>8}"
          f"{sum(r['beat_f1'] for r in rows) / n:>8.3f}"
          f"{sum(r['down_f1'] for r in rows) / n:>8.3f}")
    print(f"\ntempo: " + "  ".join(f"{k}={v}" for k, v in sorted(tallies.items())))
    print(f"time signature: {sum(r['ts_ok'] for r in rows)}/{len(rows)} exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
