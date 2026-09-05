#!/usr/bin/env python3
"""
Build (audio, ground-truth MIDI) references for BAND material.

eval/refs is solo piano, rendered from MAESTRO. That set cannot measure the
failure this exists for: on a piano + bass + drums mix, 34% of the piano notes
MT3 emits carry a spurious note an octave above, while on eval/refs only 1.5% of
octave pairs are spurious. A change aimed at that is unmeasurable there.

Differences from build_refs.py, both deliberate:
  * drums are KEPT. They are most of the notes in a band mix, and they are what
    masks the piano's fundamentals.
  * the excerpt is chosen where piano, bass and drums all play, not merely where
    the most notes are.

    python eval/build_band_refs.py --src <dir of GM MIDI> --out eval/refs_band

CAVEAT, the same one build_refs.py carries: soundfont audio is cleaner than a
real recording, so absolute F1 reads optimistic. It is a RELATIVE benchmark —
for telling whether a change helped, which is what it is for.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pretty_midi

SOUNDFONTS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
]
PIANO = range(0, 8)
BASS = range(32, 40)


def pick_soundfont() -> str:
    for s in SOUNDFONTS:
        if Path(s).exists():
            return s
    raise SystemExit("no soundfont under /usr/share/sounds/sf2/")


def _role(inst) -> str:
    if inst.is_drum:
        return "drums"
    if inst.program in PIANO:
        return "piano"
    if inst.program in BASS:
        return "bass"
    return "other"


def busiest_band_start(pm: pretty_midi.PrettyMIDI, secs: float) -> float:
    """Start of the window where piano, bass and drums are all busiest.

    Scored per role and multiplied, so a window with no piano scores zero however
    many drum hits it has: the point of the set is the piano under a band.
    """
    ons = {r: [] for r in ("piano", "bass", "drums")}
    for inst in pm.instruments:
        r = _role(inst)
        if r in ons:
            ons[r].extend(n.start for n in inst.notes)
    for r in ons:
        ons[r].sort()
    starts = sorted(ons["piano"])
    if len(starts) < 20:
        return 0.0

    def count(times, t0, t1):
        import bisect
        return bisect.bisect_left(times, t1) - bisect.bisect_left(times, t0)

    best, best_t = -1.0, starts[0]
    for t in starts:
        score = 1.0
        for r in ("piano", "bass", "drums"):
            score *= count(ons[r], t, t + secs) + 1
        if score > best:
            best, best_t = score, t
    return max(0.0, best_t - 0.25)


def excerpt(src: Path, secs: float) -> pretty_midi.PrettyMIDI | None:
    pm = pretty_midi.PrettyMIDI(str(src))
    t0 = busiest_band_start(pm, secs)
    t1 = t0 + secs
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    roles = set()
    for inst in pm.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum,
                                    name=inst.name or _role(inst))
        for n in inst.notes:
            if n.end <= t0 or n.start >= t1:
                continue
            ni.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=max(0.0, n.start - t0), end=min(secs, n.end - t0)))
        if ni.notes:
            out.instruments.append(ni)
            roles.add(_role(inst))
    if not {"piano", "bass", "drums"} <= roles:
        return None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of GM MIDI files")
    ap.add_argument("--out", default="eval/refs_band")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=44100)
    a = ap.parse_args()

    sf = pick_soundfont()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made, index = 0, []
    for src in sorted(Path(a.src).rglob("*.mid")):
        if made >= a.n:
            break
        try:
            ex = excerpt(src, a.secs)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {src.name}: {e}")
            continue
        if ex is None:
            continue
        stem = out / f"band{made:02d}"
        ex.write(str(stem.with_suffix(".mid")))
        r = subprocess.run(
            ["fluidsynth", "-ni", "-g", "0.8", "-F", str(stem.with_suffix(".wav")),
             "-r", str(a.sr), sf, str(stem.with_suffix(".mid"))],
            capture_output=True)
        if r.returncode != 0 or not stem.with_suffix(".wav").exists():
            print(f"  render failed for {src.name}")
            continue
        counts = {}
        for inst in ex.instruments:
            counts[_role(inst)] = counts.get(_role(inst), 0) + len(inst.notes)
        index.append({"ref": stem.name, "source": src.name, "notes": counts})
        print(f"  {stem.name}  {src.name[:38]:38s} {counts}")
        made += 1
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print(f"{made} references in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
