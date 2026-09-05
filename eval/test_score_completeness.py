#!/usr/bin/env python3
"""Every transcribed note has to reach the printed score.

A notated voice is monophonic-with-chords: it cannot hold notes that overlap
with different durations. So when voice separation hands build_score fewer
voices than the music needs, the builder drops whatever does not fit — silently,
because nothing downstream compares the counts.

That is exactly what happened once: _absorb_slivers used a floor of
`max(24, 0.20 * len(track))`, which is 24-100 on the 25-second eval clips but
217 on a two-minute 1086-note piano part. Every secondary voice was folded into
the first and 490 of 1637 notes (30%) never appeared in the score. The existing
layout tests all passed, because their inputs are small.

These use a densely polyphonic part sized like a real song, not a clip.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import voices as VO  # noqa: E402
from score_build import build_score  # noqa: E402


def dense_part(bars: int = 60, bpm: float = 96.0):
    """A four-voice chorale-ish texture: every beat carries four sustained
    pitches, so one notated voice provably cannot hold it."""
    beat = 60.0 / bpm
    notes = []
    for b in range(bars * 4):
        t = b * beat
        for k, p in enumerate((48, 60, 64, 67)):
            # staggered releases: the voices are not a single chord
            notes.append({"start": round(t, 4),
                          "end": round(t + beat * (0.9 + 0.3 * k), 4),
                          "pitch": p + (b % 3), "velocity": 80})
    return sorted(notes, key=lambda n: (n["start"], n["pitch"]))


def printed_notes(doc) -> int:
    """Noteheads that are a real attack (the tail of a tie is not a new note)."""
    n = 0
    for part in asdict(doc)["parts"]:
        for v in part["voices"]:
            for m in v["measures"]:
                for e in m["events"]:
                    for nt in e.get("notes") or []:
                        if not nt.get("tie_stop"):
                            n += 1
    return n


def _score(groups):
    return build_score(
        [{"name": "p", "program": 0, "is_drum": False,
          "voices": groups, "notes": []}],
        tempo=96.0, time_sig=(4, 4))


def test_dense_part_keeps_essentially_every_note():
    src = dense_part()
    groups = VO.separate_sequences(src)
    got = printed_notes(_score(groups))
    assert got >= len(src) * 0.98, (
        f"score printed {got} of {len(src)} notes "
        f"({100 * (len(src) - got) / len(src):.1f}% lost) across {len(groups)} voices")


def test_a_polyphonic_part_is_not_collapsed_to_one_voice():
    groups = VO.separate_sequences(dense_part())
    assert len(groups) > 1, (
        "a part with four sustained simultaneous lines was reduced to one "
        "notation voice; the builder cannot print that without dropping notes")


def test_sliver_floor_does_not_scale_with_track_size():
    """The regression itself: a big track must not raise the absorption floor."""
    small = [[{"start": i * 0.5, "end": i * 0.5 + 0.4, "pitch": 60} for i in range(40)],
             [{"start": i * 0.5, "end": i * 0.5 + 0.4, "pitch": 72} for i in range(40)]]
    big = [[{"start": i * 0.1, "end": i * 0.1 + 0.09, "pitch": 60} for i in range(1000)],
           [{"start": i * 0.1, "end": i * 0.1 + 0.09, "pitch": 72} for i in range(200)]]
    assert len(VO._absorb_slivers([list(g) for g in small], VO.MIN_SEQUENCE_NOTES)) == 2
    assert len(VO._absorb_slivers([list(g) for g in big], VO.MIN_SEQUENCE_NOTES)) == 2, (
        "a 200-note voice was absorbed because the other voice was large")


def test_tiny_stray_sequence_is_still_absorbed():
    parts = [[{"start": i * 0.5, "end": i * 0.5 + 0.4, "pitch": 60} for i in range(80)],
             [{"start": 3.0, "end": 3.2, "pitch": 90}]]
    assert len(VO._absorb_slivers([list(g) for g in parts], VO.MIN_SEQUENCE_NOTES)) == 1, (
        "a one-note stray should not be advertised as its own part")


def wide_line(bars: int = 40, bpm: float = 96.0):
    """One line that genuinely covers two registers: a bass note and a melody
    note alternating, the way an accompaniment figure does."""
    beat = 60.0 / bpm
    out = []
    for b in range(bars * 4):
        t = b * beat
        out.append({"start": round(t, 4), "end": round(t + beat * 0.9, 4),
                    "pitch": 40 + (b % 5), "velocity": 80})
        out.append({"start": round(t + beat * 0.5, 4), "end": round(t + beat * 1.4, 4),
                    "pitch": 79 + (b % 5), "velocity": 80})
    return sorted(out, key=lambda n: (n["start"], n["pitch"]))


def _ledger_load(doc) -> tuple[int, int]:
    """(total ledger lines, noteheads needing four or more)."""
    span = {"treble": (64, 77), "bass": (43, 57), "percussion": (64, 77)}
    total = worst = 0
    for part in asdict(doc)["parts"]:
        clefs = part.get("clefs") or [part.get("clef", "treble")]
        for v in part["voices"]:
            lo, hi = span[clefs[min(v.get("staff", 1) - 1, len(clefs) - 1)]]
            for m in v["measures"]:
                for e in m["events"]:
                    for nt in e.get("notes") or []:
                        k = (lo - nt["midi"]) // 2 if nt["midi"] < lo else (
                            (nt["midi"] - hi) // 2 if nt["midi"] > hi else 0)
                        total += k
                        worst += k >= 4
    return total, worst


def test_a_two_register_line_is_split_across_the_staves():
    src = wide_line()
    doc = _score(VO.separate_sequences(src))
    ranges = []
    for part in asdict(doc)["parts"]:
        for v in part["voices"]:
            ps = [nt["midi"] for m in v["measures"] for e in m["events"]
                  for nt in (e.get("notes") or [])]
            if ps:
                ranges.append(max(ps) - min(ps))
    assert ranges and max(ranges) < 30, (
        f"a printed voice still spans {max(ranges)} semitones; notes should pick "
        "their staff by register instead of inheriting the voice's")


def test_register_split_keeps_ledger_lines_down():
    """Compared against printing the same notes on a single staff, which is what
    happens when a voice inherits one staff for its whole range."""
    src = wide_line()
    total, worst = _ledger_load(_score(VO.separate_sequences(src)))

    def one_staff_cost(clef):
        lo, hi = {"treble": (64, 77), "bass": (43, 57)}[clef]
        return sum((lo - n["pitch"]) // 2 if n["pitch"] < lo else
                   ((n["pitch"] - hi) // 2 if n["pitch"] > hi else 0) for n in src)
    ref = min(one_staff_cost("treble"), one_staff_cost("bass"))

    assert worst == 0, f"{worst} noteheads need four or more ledger lines"
    assert total <= ref * 0.5, (
        f"{total} ledger lines against {ref} for a single staff — the staves are "
        "not being chosen by register")


def test_register_split_loses_no_notes():
    src = wide_line()
    got = printed_notes(_score(VO.separate_sequences(src)))
    assert got >= len(src) * 0.98, f"{got} of {len(src)} notes survived the split"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if fails else 'all score-completeness tests passed'}")
    sys.exit(1 if fails else 0)
