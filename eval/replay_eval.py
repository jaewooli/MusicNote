#!/usr/bin/env python3
"""Score MusicNote's MT3 post-processing against ground truth, without MT3.

Reads the raw notes cached by `eval/mt3_cache.py` and replays exactly the
stages that run after inference — the velocity/length gate and sequence
separation — so a parameter change can be measured in seconds.

    python eval/replay_eval.py eval/refs
    python eval/replay_eval.py eval/refs --min-note 0.08      # old behaviour
    python eval/replay_eval.py eval/refs --sweep-min-note 0.02,0.05,0.08,0.11
    python eval/replay_eval.py eval/refs --shifts 1024 --min-agreement 1   # union
    python eval/replay_eval.py eval/refs --shifts 1024 --min-agreement 2   # vote

Reports onset / note / note+offset precision, recall and F1, plus how many
notes the gate deleted — under-detection and over-detection stay separable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import mt3_ensemble as E  # noqa: E402
import mt3_post as MP  # noqa: E402
import voices as VO  # noqa: E402


def hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _arrays(triples):
    iv = np.array([[a, b] for a, b, _ in triples], float) if triples else np.zeros((0, 2))
    p = np.array([hz(m) for _, _, m in triples], float) if triples else np.zeros(0)
    return iv, p


def ref_notes(mid: Path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(mid))
    return _arrays(sorted((n.start, n.end, n.pitch)
                          for i in pm.instruments if not i.is_drum for n in i.notes))


def caches_for(mid: Path, shifts: list[str]) -> list[Path]:
    """Base run plus one cache per requested shift tag (e.g. "1024" = 1.024 s)."""
    out = [mid.with_suffix(".mt3.json")]
    out += [mid.with_suffix(f".mt3.s{t}.json") for t in shifts]
    return [p for p in out if p.exists()]


def est_notes(caches: list[Path], sensitivity: float, split_voices: bool,
              min_agreement: int = 1):
    """Replay the post-MT3 pipeline for one clip over one or more runs."""
    runs, report, raw_total = [], None, 0
    for c in caches:
        raw = json.loads(c.read_text())["notes"]
        raw_total += len(raw)
        kept, rep = MP.gate(raw, sensitivity)
        report = rep if report is None else {
            k: (report[k] + rep[k] if isinstance(rep[k], int) else rep[k]) for k in rep}
        runs.append([{"start": float(n["start"]), "end": float(n["end"]),
                      "pitch": int(n["pitch"]), "velocity": int(n["velocity"]),
                      "track": int(n["track"])} for n in kept])

    merged = E.merge(runs)
    accepted, review = E.split(merged, len(runs), min_agreement)

    by_track: dict[int, list[dict]] = {}
    for n in accepted:
        by_track.setdefault(n["track"], []).append(dict(n))

    out = []
    for t, notes in sorted(by_track.items()):
        notes.sort(key=lambda n: (n["start"], n["pitch"]))
        parts = [notes]
        # Mirrors app._mt3_stems: sequence separation only above these floors.
        if split_voices and len(notes) >= 8 and VO.poly_fraction(notes) >= 0.18:
            parts = VO.separate_sequences(notes) or [notes]
        for part in parts:
            out += [(n["start"], n["end"], n["pitch"]) for n in part]
    out.sort()
    return _arrays(out), report, raw_total, review


def score(ref_iv, ref_p, est_iv, est_p):
    import mir_eval
    on = mir_eval.transcription.onset_precision_recall_f1(ref_iv, est_iv, onset_tolerance=0.05)
    nt = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    off = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=0.2)
    return on[:3], nt[:3], off[:3]


def run(refdir: Path, sensitivity: float, split_voices: bool, verbose: bool = True,
        shifts: list[str] | None = None, min_agreement: int = 1):
    rows, dropped, raw_total, n_review = [], 0, 0, 0
    for mid in sorted(refdir.glob("*.mid")):
        caches = caches_for(mid, shifts or [])
        if not caches:
            continue
        ref_iv, ref_p = ref_notes(mid)
        (est_iv, est_p), report, n_raw, review = est_notes(
            caches, sensitivity, split_voices, min_agreement)
        dropped += report["dropped_length"] + report["dropped_velocity"]
        raw_total += n_raw
        n_review += len(review)
        on, nt, off = score(ref_iv, ref_p, est_iv, est_p)
        rows.append((on, nt, off, len(ref_iv), len(est_iv)))
        if verbose:
            print(f"{mid.stem:8s} runs={len(caches)} onsetF1={on[2]:.3f}  noteP={nt[0]:.3f} "
                  f"noteR={nt[1]:.3f} noteF1={nt[2]:.3f}  +offF1={off[2]:.3f}  "
                  f"ref={len(ref_iv):4d} est={len(est_iv):4d} review={len(review):3d}")
    if not rows:
        print(f"no cached *.mt3.json in {refdir} — run eval/mt3_cache.py first", file=sys.stderr)
        return None
    m = np.array([[*r[0], *r[1], *r[2]] for r in rows], float).mean(axis=0)
    return {"onset_f1": m[2], "note_p": m[3], "note_r": m[4], "note_f1": m[5],
            "off_f1": m[8], "clips": len(rows), "raw": raw_total, "dropped": dropped,
            "review": n_review}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--min-note", type=float, help="override MIN_NOTE_BASE (span forced to 0)")
    ap.add_argument("--sweep-min-note", help="comma separated values to compare")
    ap.add_argument("--no-split-voices", action="store_true")
    ap.add_argument("--shifts", default="",
                    help="comma separated shift tags to ensemble in, e.g. 1024")
    ap.add_argument("--min-agreement", type=int, default=1,
                    help="runs a note must appear in to be accepted (1 = union)")
    a = ap.parse_args()
    refdir = Path(a.refdir)
    split = not a.no_split_voices
    shifts = [s.strip() for s in a.shifts.split(",") if s.strip()]

    def apply(v: float | None):
        if v is not None:
            MP.MIN_NOTE_BASE, MP.MIN_NOTE_SPAN = v, 0.0

    if a.sweep_min_note:
        print(f"{'min_note':>9s} {'onsetF1':>8s} {'noteP':>7s} {'noteR':>7s} "
              f"{'noteF1':>7s} {'+offF1':>7s} {'gate drop':>10s}")
        for v in [float(x) for x in a.sweep_min_note.split(",")]:
            apply(v)
            r = run(refdir, a.sensitivity, split, verbose=False,
                    shifts=shifts, min_agreement=a.min_agreement)
            if r:
                print(f"{v:9.3f} {r['onset_f1']:8.3f} {r['note_p']:7.3f} {r['note_r']:7.3f} "
                      f"{r['note_f1']:7.3f} {r['off_f1']:7.3f} "
                      f"{r['dropped']:6d}/{r['raw']:<5d}")
        return 0

    apply(a.min_note)
    amp, ml = MP.gate_params(a.sensitivity)
    print(f"sensitivity={a.sensitivity}  velocity_floor={amp}  min_note={ml:.3f}s  "
          f"split_voices={split}  shifts={shifts or 'none'}  "
          f"min_agreement={a.min_agreement}\n")
    r = run(refdir, a.sensitivity, split, shifts=shifts, min_agreement=a.min_agreement)
    if not r:
        return 2
    print(f"\nMEAN   onset F1 = {r['onset_f1']:.3f}   note P = {r['note_p']:.3f}   "
          f"note R = {r['note_r']:.3f}   note F1 = {r['note_f1']:.3f}   "
          f"note+offset F1 = {r['off_f1']:.3f}   over {r['clips']} clip(s)")
    print(f"       gate dropped {r['dropped']} of {r['raw']} raw MT3 notes; "
          f"{r['review']} note(s) sent to the review queue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
