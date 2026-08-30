#!/usr/bin/env python3
"""
Build a small (audio, ground-truth MIDI) reference set for eval_harness.

Takes source MIDI (MAESTRO = real human piano performances, so the timing is
musical rather than quantised), cuts a fixed-length excerpt starting at a
musically busy point, writes the excerpt MIDI time-shifted to 0, and renders it
to WAV with fluidsynth.

    python eval/build_refs.py --n 6 --secs 25 --out eval/refs

CAVEAT, on purpose: soundfont-rendered audio is CLEANER than a real acoustic
recording (no room, no mic, no pedal blur), so absolute F1 here reads optimistic
vs. real-world audio. It is still a valid RELATIVE benchmark — that is what it
is for: telling whether a change helped or hurt.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import pretty_midi

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


def busiest_start(pm: pretty_midi.PrettyMIDI, secs: float) -> float:
    """Start of the `secs` window holding the most note onsets."""
    ons = sorted(n.start for i in pm.instruments if not i.is_drum for n in i.notes)
    if len(ons) < 20:
        return 0.0
    best, best_t, j = 0, ons[0], 0
    for i, t in enumerate(ons):
        while j < len(ons) and ons[j] < t + secs:
            j += 1
        if j - i > best:
            best, best_t = j - i, t
    return max(0.0, best_t - 0.25)


def excerpt(src: Path, secs: float, program: int | None = None) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(str(src))
    t0 = busiest_start(pm, secs)
    t1 = t0 + secs
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        ni = pretty_midi.Instrument(
            program=inst.program if program is None else program,
            name=inst.name or "part")
        for n in inst.notes:
            if n.end <= t0 or n.start >= t1:
                continue
            ni.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=max(0.0, n.start - t0),
                end=min(secs, n.end - t0)))
        if ni.notes:
            out.instruments.append(ni)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/maestro/maestro-v3.0.0")
    ap.add_argument("--out", default="eval/refs")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    sf = pick_soundfont()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted(Path(a.src).rglob("*.mid*"))
    if not srcs:
        raise SystemExit(f"no MIDI under {a.src}")
    random.Random(a.seed).shuffle(srcs)

    made = 0
    for src in srcs:
        if made >= a.n:
            break
        try:
            pm = excerpt(src, a.secs)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {src.name}: {e}")
            continue
        n_notes = sum(len(i.notes) for i in pm.instruments)
        if n_notes < 40:
            continue
        stem = f"ref{made:02d}"
        mid_p, wav_p = out / f"{stem}.mid", out / f"{stem}.wav"
        pm.write(str(mid_p))
        r = subprocess.run(
            ["fluidsynth", "-ni", "-g", "0.8", "-F", str(wav_p),
             "-r", str(a.sr), sf, str(mid_p)],
            capture_output=True, text=True)
        if r.returncode != 0 or not wav_p.exists():
            print(f"  render failed {stem}: {r.stderr[-200:]}")
            mid_p.unlink(missing_ok=True)
            continue
        print(f"  {stem}: {n_notes:4d} notes  {a.secs:.0f}s  <- {src.name[:52]}")
        made += 1

    print(f"\n{made} reference pair(s) in {out}/   (soundfont: {Path(sf).name})")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
