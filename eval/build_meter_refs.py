#!/usr/bin/env python3
"""
Build a reference set for BEAT / DOWNBEAT / TIME-SIGNATURE detection.

eval/refs cannot measure any of this: eval/build_refs.py writes every excerpt
into a fresh `PrettyMIDI(initial_tempo=120.0)` container, so all six references
claim 120 BPM 4/4 regardless of the source. MAESTRO could not help anyway — it
is performance capture, with no notated meter to be right about.

This set comes from music21's corpus instead, which is score-derived and so has
a real time signature, and renders at a tempo we choose and record.

    python eval/build_meter_refs.py --n 12 --out eval/refs_meter

Each reference is three files:

    mrefNN.wav          rendered audio
    mrefNN.mid          note ground truth (times in the excerpt's timeline)
    mrefNN.meter.json   {tempo, time_sig, beats[], downbeats[], source}

Two deliberate choices keep the benchmark honest:

  * Excerpts start at a RANDOM OFFSET INSIDE A BAR, not on a downbeat. Cutting
    on the barline would put the first downbeat at t=0 and make phase detection
    free, which is not the problem real uploads pose.
  * Onsets get a small Gaussian jitter, so the grid is not machine-perfect. The
    ground-truth beat times stay exact; only the performance moves.

CAVEAT, same as eval/build_refs.py: soundfont renders are cleaner than real
recordings, and a constant tempo is kinder than a live rubato performance. This
is a RELATIVE benchmark — it tells you whether a change helped.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import warnings
from fractions import Fraction
from pathlib import Path

import pretty_midi

warnings.filterwarnings("ignore")

SOUNDFONTS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
]


def pick_soundfont() -> str:
    for s in SOUNDFONTS:
        if Path(s).exists():
            return s
    raise SystemExit("no soundfont found under /usr/share/sounds/sf2/")


def beat_unit(num: int, den: int) -> tuple[Fraction, int]:
    """(quarter-lengths per beat, beats per bar) for a time signature.

    Compound meters are felt in dotted beats: 6/8 is two beats of three eighths,
    not six. Getting this wrong would score a correct detector as wrong.
    """
    if den == 8 and num in (6, 9, 12):
        return Fraction(3, 2), num // 3
    return Fraction(4, den), num


DEFAULT_METERS = "4/4:4,3/4:3,6/8:3,2/4:3,2/2:2,9/8:1"
# Candidates per wanted reference. Most get dropped for length or note
# count, so selecting exactly `quota` scores yields far fewer than that.
POOL_DEPTH = 22


def load_scores(meters: dict[str, int], seed: int) -> list:
    """Scores for each wanted meter, via the corpus metadata index.

    Searching by time signature rather than parsing a random sample matters:
    a shuffle of the whole corpus lands on Palestrina and fiddle reels, which
    gave a first set that was mostly 4/2 and 2/2 — meters nobody uploads.
    """
    from music21 import corpus
    rng = random.Random(seed)
    found: list[tuple[str, int, int, object]] = []
    for sig, quota in meters.items():
        num, den = (int(x) for x in sig.split("/"))
        try:
            hits = list(corpus.search(sig, field="timeSignature"))
        except Exception as e:  # noqa: BLE001
            print(f"  {sig:>5}: search failed ({e})")
            continue
        rng.shuffle(hits)
        got = 0
        for entry in hits:
            if got >= quota * POOL_DEPTH:
                break
            try:
                s = entry.parse()
            except Exception:
                continue
            tss = {t.ratioString
                   for t in s.recurse().getElementsByClass("TimeSignature")}
            if tss != {sig}:          # mixed-meter work: no single ground truth
                continue
            if not s.recurse().notes:
                continue
            name = Path(str(entry.sourcePath)).name
            found.append((sig, name, num, den, s))
            got += 1
        print(f"  {sig:>5}: {got} candidate(s)")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/refs_meter")
    ap.add_argument("--meters", default=DEFAULT_METERS,
                    help='"4/4:3,3/4:3,6/8:2" — signature:count')
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-poly", type=float, default=1.5,
                    help="mean notes per onset event; 1.0 is a bare melody")
    ap.add_argument("--jitter", type=float, default=0.015,
                    help="onset jitter sigma in seconds (0 = metronomic)")
    a = ap.parse_args()

    sf = pick_soundfont()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    print("scanning corpus…")
    meters = {}
    for part in a.meters.split(","):
        sig, _, cnt = part.strip().partition(":")
        meters[sig] = int(cnt or 1)
    scores = load_scores(meters, a.seed)
    if not scores:
        raise SystemExit("no usable scores found")

    made = 0
    produced: dict[str, int] = {}
    for sig, src, num, den, score in scores:
        if produced.get(sig, 0) >= meters.get(sig, 1):
            continue
        ql_per_beat, beats_per_bar = beat_unit(num, den)
        bar_ql = float(ql_per_beat) * beats_per_bar
        score_ql = float(score.duration.quarterLength)
        # A faster tempo needs more written music to fill the same 25 s, so pick
        # the tempo to fit the score instead of discarding every short tune.
        # 60 BPM is dropped on purpose. "Pick the slowest tempo that fits"
        # made 5 of the first 12 references exactly 60 BPM, which is both
        # unrepresentative and the most ambiguous case there is: a 60 BPM
        # quarter carrying running eighths is a grid a human would also tap at
        # 120. Short scores are skipped instead of being pushed down there.
        cands = [72, 84, 96, 108, 120, 132, 144, 160]
        rng.shuffle(cands)
        bpm = next((b for b in cands
                    if score_ql >= (a.secs * b / 60.0) + 2 * bar_ql), None)
        if bpm is None:
            print(f"  skip {src[:30]}: {score_ql:.0f} quarters is too short for {a.secs:.0f}s")
            continue
        qsec = 60.0 / bpm                       # seconds per quarter note
        beat_sec = float(ql_per_beat) * qsec
        bar_sec = beat_sec * beats_per_bar

        tmp = out / "_tmp.mid"
        try:
            from music21 import tempo as m21tempo
            flat = score.flatten()
            for mm in list(flat.getElementsByClass("MetronomeMark")):
                flat.remove(mm)
            flat.insert(0, m21tempo.MetronomeMark(number=bpm))
            flat.write("midi", str(tmp))
            pm = pretty_midi.PrettyMIDI(str(tmp))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {src[:30]}: {e}")
            continue
        finally:
            tmp.unlink(missing_ok=True)

        total = pm.get_end_time()
        if total < a.secs + 2 * bar_sec:
            print(f"  skip {src[:30]}: too short after render")
            continue

        # Start inside a bar, never on the barline. Phase is the thing being
        # measured; handing it over for free would make the benchmark lie.
        n_bars = int((total - a.secs - bar_sec) / bar_sec)
        if n_bars < 2:
            continue
        bar0 = rng.randint(1, max(1, n_bars))
        t0 = bar0 * bar_sec + rng.uniform(0.12, 0.88) * bar_sec
        t1 = t0 + a.secs

        exc = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        n_notes = 0
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            ni = pretty_midi.Instrument(program=inst.program, name=inst.name or "part")
            for n in inst.notes:
                if n.end <= t0 or n.start >= t1:
                    continue
                j = rng.gauss(0.0, a.jitter) if a.jitter else 0.0
                s = max(0.0, n.start - t0 + j)
                e = min(a.secs, n.end - t0 + j)
                if e - s < 0.02:
                    continue
                ni.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch, start=s, end=e))
            if ni.notes:
                exc.instruments.append(ni)
                n_notes += len(ni.notes)
        if n_notes < 30:
            print(f"  skip {src[:30]}: only {n_notes} notes in the window")
            continue
        # Polyphony matters more than note count. A solo melody carries no bass
        # and no chord changes, so it has almost no downbeat cue at all — the
        # first version of this set was 13/16 monophonic folk tunes, which is
        # both the hardest possible case and not what anyone uploads. Real input
        # is accompaniment plus melody.
        starts = sorted(n.start for i in exc.instruments for n in i.notes)
        events = 1 + sum(1 for i in range(1, len(starts))
                         if starts[i] - starts[i - 1] > 0.035)
        poly = n_notes / max(1, events)
        if poly < a.min_poly:
            print(f"  skip {src[:30]}: polyphony {poly:.2f} < {a.min_poly}")
            continue

        # Ground truth on the excerpt timeline. The first beat at or after t0 is
        # `k` beats past the piece's start, and k % beats_per_bar tells us where
        # in the bar it falls.
        k0 = int(-(-t0 // beat_sec))
        beats, downbeats = [], []
        k = k0
        while k * beat_sec < t0 + a.secs:
            bt = k * beat_sec - t0
            if bt >= 0:
                beats.append(round(bt, 6))
                if k % beats_per_bar == 0:
                    downbeats.append(round(bt, 6))
            k += 1

        stem = f"mref{made:02d}"
        mid_p, wav_p, js_p = (out / f"{stem}.mid", out / f"{stem}.wav",
                              out / f"{stem}.meter.json")
        exc.write(str(mid_p))
        r = subprocess.run(
            ["fluidsynth", "-ni", "-g", "0.8", "-F", str(wav_p),
             "-r", str(a.sr), sf, str(mid_p)],
            capture_output=True, text=True)
        if r.returncode != 0 or not wav_p.exists():
            print(f"  render failed {stem}: {r.stderr[-160:]}")
            mid_p.unlink(missing_ok=True)
            continue
        js_p.write_text(json.dumps({
            "tempo": bpm,
            "time_sig": [num, den],
            "beats_per_bar": beats_per_bar,
            "beat_seconds": round(beat_sec, 6),
            "beats": beats,
            "downbeats": downbeats,
            "polyphony": round(poly, 3),
            "jitter_sigma": a.jitter,
            "source": src,
        }, indent=1))
        print(f"  {stem}: {num}/{den} {bpm:3d}bpm  {n_notes:4d} notes  "
              f"poly {poly:.2f}  {len(downbeats):2d} bars  <- {src[:34]}")
        made += 1
        produced[sig] = produced.get(sig, 0) + 1

    print(f"\n{made} meter reference(s) in {out}/   (soundfont: {Path(sf).name})")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
