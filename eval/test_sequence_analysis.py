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

# A legato melody survives a short MT3 offset overrun as one contour.
check("offset_jitter_melody",
      [(0, (72,)), (.20, (74,)), (.40, (76,)), (.60, (77,))], [4])

# Merging is transitive, so pairwise agreement is not enough. Three lines an
# octave apart pass A-B and B-C but A-C is rejected as too far apart; single
# linkage used to chain them into one 28-semitone "chord".
check("no_chain_merge_across_two_octaves",
      [(0, (48, 60, 72)), (.5, (50, 62, 74)), (1, (52, 64, 76))], [3, 6])


def _held(events, hold):
    """Notes whose offsets overrun the following attacks, as MT3 emits them."""
    out = []
    for t, pitches in events:
        for p in pitches:
            out.append({"start": t, "end": t + hold, "pitch": p, "velocity": 90})
    return out


def check_notes(label, notes, expected):
    got = sorted(len(part) for part in separate_sequences(notes))
    assert got == sorted(expected), (label, got, expected)
    print(f"{label}: {got}")


# MT3 routinely holds a note far past the next attack. Vetoing reuse on that
# overlap locked a line out of its own continuation, so one monophonic melody
# ping-ponged between two contours. Measured on the Canon clip: overruns of
# 0.32-1.32 s against a 0.20 s onset spacing.
check_notes("held_notes_stay_one_line",
            _held([(i * 0.2, (72 + (i % 3),)) for i in range(10)], hold=1.2), [10])

# An octave is the most common large melodic interval. Charging it 12.0 made it
# exactly as expensive as opening a new voice, so every octave leap started one.
check_notes("octave_leap_stays_one_line",
            _held([(0.0, (81,)), (0.2, (69,)), (0.4, (71,)), (0.6, (73,))],
                  hold=0.18), [4])

# ... but a genuinely distant jump still starts a new line.
check_notes("far_jump_starts_new_line",
            _held([(0.0, (81,)), (0.2, (81,)), (0.4, (43,)), (0.6, (45,))],
                  hold=0.18), [2, 2])


def check_span(label, events, limit):
    """No inferred chordal sequence may exceed the octave rule it claims."""
    for part in separate_sequences(_notes(events)):
        by_onset = {}
        for n in part:
            by_onset.setdefault(round(float(n["start"]), 3), []).append(int(n["pitch"]))
        for t, pitches in by_onset.items():
            span = max(pitches) - min(pitches)
            assert span <= limit, (label, f"t={t}", f"span={span}", pitches)
    print(f"{label}: every simultaneous group within {limit} semitones")


# The vertical span of one sequence is the invariant the module documents;
# check it directly rather than only through group sizes.
check_span("chord_span_within_octave",
           [(0, (48, 60, 72)), (.5, (50, 62, 74)), (1, (52, 64, 76))], 12)
