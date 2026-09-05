#!/usr/bin/env python3
"""Merge semantics for the shifted-run MT3 ensemble."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import mt3_ensemble as E


def n(start, pitch, end=None, track=0, is_drum=False):
    return {"start": start, "end": end if end is not None else start + 0.2,
            "pitch": pitch, "track": track, "is_drum": is_drum}


def test_identical_runs_agree_everywhere():
    run = [n(0.0, 60), n(0.5, 62)]
    m = E.merge([run, list(run)])
    assert len(m) == 2 and all(x["agreement"] == 2 for x in m)


def test_vote_withholds_single_run_note_but_keeps_it_for_review():
    m = E.merge([[n(0.0, 60)], [n(0.0, 60), n(1.0, 67)]])
    acc, rev = E.split(m, 2, min_agreement=2)
    assert [x["pitch"] for x in acc] == [60]
    assert [x["pitch"] for x in rev] == [67]
    assert rev[0]["in_score"] is False, "a withheld note must be offered to add"


def test_union_delivers_everything_but_still_flags_disagreement():
    m = E.merge([[n(0.0, 60)], [n(0.0, 60), n(1.0, 67)]])
    acc, rev = E.split(m, 2, min_agreement=1)
    assert sorted(x["pitch"] for x in acc) == [60, 67], "union delivers both"
    assert [x["pitch"] for x in rev] == [67], "union must still flag the shaky note"
    assert rev[0]["in_score"] is True, "flagged note is in the score; verify it"


def test_notes_found_by_every_run_are_never_reviewed():
    run = [n(0.0, 60), n(0.5, 62)]
    _, rev = E.split(E.merge([run, list(run)]), 2, min_agreement=1)
    assert rev == []


def test_repeated_pitch_is_not_collapsed():
    # Two attacks of the same pitch 300 ms apart must stay two notes, and must
    # not both match the same note in the other run.
    run = [n(0.0, 60), n(0.3, 60)]
    m = E.merge([run, list(run)])
    assert len(m) == 2, f"repeated pitch collapsed: {m}"
    assert all(x["agreement"] == 2 for x in m)


def test_small_timing_jitter_still_matches():
    m = E.merge([[n(0.000, 60)], [n(0.030, 60)]])
    assert len(m) == 1 and m[0]["agreement"] == 2


def test_beyond_tolerance_is_two_notes():
    m = E.merge([[n(0.0, 60)], [n(0.5, 60)]])
    assert len(m) == 2 and all(x["agreement"] == 1 for x in m)


def test_same_note_on_different_tracks_still_agrees():
    """MT3's program assignment moves when the segment boundaries move.

    This used to assert the opposite — that a differing track meant a different
    note. Measured on eval/refs_band at agreement 2, keying the match on the
    track cost recall for nothing: R 0.599 / F1 0.671 with the track in the key
    against R 0.633 / F1 0.681 without it. A note every run found was being
    split into several agreement-1 notes and then discarded by the vote.
    """
    m = E.merge([[n(0.0, 60, track=0)], [n(0.0, 60, track=1)]])
    assert len(m) == 1, "the same pitch at the same time is one note"
    assert m[0]["agreement"] == 2, "both runs found it, so both should vote"


def test_a_drum_hit_never_merges_with_a_pitched_note():
    """A drum note's `pitch` is an instrument id, not a pitch: MIDI 38 on
    channel 10 is a snare, and merging it with a real D2 would be merging two
    unrelated events. That distinction stays in the key; the program does not."""
    m = E.merge([[n(0.0, 38, is_drum=True)], [n(0.0, 38, is_drum=False)]])
    assert len(m) == 2, "a snare and a D2 are not the same event"


def test_single_run_accepts_everything():
    m = E.merge([[n(0.0, 60)]])
    acc, rev = E.split(m, 1, min_agreement=2)
    assert len(acc) == 1 and rev == [], "one run cannot vote against itself"


def test_disjoint_runs_deliver_union_and_review_all():
    m = E.merge([[n(0.0, 60)], [n(1.0, 67)]])
    acc, rev = E.split(m, 2, min_agreement=1)
    assert len(acc) == 2 and len(rev) == 2, "nothing was seen twice"


# --- the transposed ensemble member -------------------------------------------
# A run on a resampled copy only helps if the transform inverts EXACTLY; a note
# that comes back a semitone or a few milliseconds off is a new false positive
# rather than a second vote for a real note.

def test_transpose_round_trip_restores_time_and_pitch():
    import mt3_bridge as B
    ratio = 2.0 ** (1 / 12.0)
    got = B._untranspose(
        {"notes": [{"start": round(1.0 / ratio, 4), "end": round(2.0 / ratio, 4),
                    "pitch": 61, "is_drum": False}]}, 1, ratio)
    n = got["notes"][0]
    assert n["pitch"] == 60, f"pitch came back as {n['pitch']}, not 60"
    assert abs(n["start"] - 1.0) < 1e-3 and abs(n["end"] - 2.0) < 1e-3, \
        f"times came back as {n['start']}..{n['end']}"


def test_transpose_leaves_drum_slots_alone():
    import mt3_bridge as B
    # 36 is the GM kick. It is a kit slot, not a pitch: a resampled kick still
    # sounds like a kick and comes back on 36, so subtracting the semitones
    # would rewrite it to a different instrument.
    got = B._untranspose(
        {"notes": [{"start": 0.0, "end": 0.1, "pitch": 36, "is_drum": True}]},
        1, 2.0 ** (1 / 12.0))
    assert got["notes"][0]["pitch"] == 36, "transposed a drum off its own line"


def test_transposed_copy_is_shorter_by_the_ratio():
    import tempfile
    import numpy as np
    import soundfile as sf
    import mt3_bridge as B
    with tempfile.TemporaryDirectory() as td:
        src = f"{td}/src.wav"
        sf.write(src, np.zeros(16000 * 2, dtype=np.float32), 16000)
        out, ratio = B._transposed_copy(src, 1, td)
        y, sr = sf.read(out)
        assert sr == 16000, f"declared rate changed to {sr}"
        assert abs(len(y) / sr - 2.0 / ratio) < 0.05, \
            f"copy is {len(y)/sr:.3f}s, expected {2.0/ratio:.3f}s"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1; print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if fails else 'all ensemble tests passed'}")
    sys.exit(1 if fails else 0)
