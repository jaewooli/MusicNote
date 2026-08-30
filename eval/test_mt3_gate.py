#!/usr/bin/env python3
"""Gate regression: the post-MT3 filter must not delete ordinary short notes.

63 % of the eval/refs ground truth is shorter than the 80 ms floor the gate
used to apply at default sensitivity, so this is guarded by a test rather than
left as a tunable constant nobody re-checks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import mt3_post as MP


def note(start, dur, velocity=80, pitch=60, track=0):
    return {"start": start, "end": start + dur, "pitch": pitch,
            "velocity": velocity, "program": 0, "is_drum": False, "track": track}


def test_default_keeps_fast_passage_notes():
    # A 16th-note run at 120 bpm is 125 ms; at 200 bpm it is 75 ms. Both are
    # ordinary piano writing, not detector noise.
    raw = [note(i * 0.075, 0.070) for i in range(8)]
    kept, report = MP.gate(raw, 0.5)
    assert len(kept) == 8, f"gate deleted fast-passage notes: {report}"


def test_grace_note_survives():
    kept, _ = MP.gate([note(0.0, 0.040)], 0.5)
    assert len(kept) == 1, "a 40 ms grace note must survive the default gate"


def test_zero_length_is_still_removed():
    kept, report = MP.gate([note(0.0, 0.0), note(1.0, 0.5)], 0.5)
    assert len(kept) == 1 and report["dropped_length"] == 1


def test_velocity_floor_scales_with_sensitivity():
    raw = [note(0.0, 0.5, velocity=5)]
    assert len(MP.gate(raw, 1.0)[0]) == 1, "max sensitivity keeps very quiet notes"
    assert len(MP.gate(raw, 0.0)[0]) == 0, "min sensitivity drops very quiet notes"


def test_report_accounts_for_every_input_note():
    raw = [note(0.0, 0.0), note(1.0, 0.5), note(2.0, 0.5, velocity=1)]
    kept, r = MP.gate(raw, 0.5)
    assert r["input"] == 3
    assert r["kept"] + r["dropped_length"] + r["dropped_velocity"] == r["input"]
    assert len(kept) == r["kept"]


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
    print(f"\n{'FAILED' if fails else 'all gate tests passed'}")
    sys.exit(1 if fails else 0)
