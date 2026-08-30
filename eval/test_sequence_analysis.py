"""Small deterministic regression tests for chord-vs-sequence analysis.

Run with:
    python eval/test_sequence_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from voices import separate_sequences


def _notes(events):
    return [{"start": t, "end": t + .35, "pitch": p, "velocity": 90}
            for t, pitches in events for p in pitches]


def check(label, events, expected):
    got = sorted(len(part) for part in separate_sequences(_notes(events)))
    assert got == sorted(expected), (label, got, expected)
    print(f"{label}: {got}")


# A moving triad remains one chordal sequence.
check("parallel_triad",
      [(0, (60, 64, 67)), (.5, (62, 65, 69)), (1, (64, 67, 71))], [9])
# A non-parallel upper melody retains an independent sequence.
check("chord_plus_melody",
      [(0, (60, 64, 67, 76)), (.5, (62, 65, 69, 79)),
       (1, (64, 67, 71, 77))], [9, 3])
# A single colour tone is folded back into its chord instead of becoming a part.
check("one_off_extension",
      [(0, (60, 64, 67)), (.5, (62, 65, 69, 72)),
       (1, (64, 67, 71))], [10])
# A low line remains separate from the chord even if attacks coincide.
check("bass_plus_chord",
      [(0, (40, 60, 64, 67)), (.5, (43, 62, 65, 69)),
       (1, (45, 64, 67, 71))], [3, 9])
