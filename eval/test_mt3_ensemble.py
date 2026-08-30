#!/usr/bin/env python3
"""Merge semantics for the shifted-run MT3 ensemble."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import mt3_ensemble as E


def n(start, pitch, end=None, track=0):
    return {"start": start, "end": end if end is not None else start + 0.2,
            "pitch": pitch, "track": track}


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


def test_different_track_is_not_merged():
    m = E.merge([[n(0.0, 60, track=0)], [n(0.0, 60, track=1)]])
    assert len(m) == 2


def test_single_run_accepts_everything():
    m = E.merge([[n(0.0, 60)]])
    acc, rev = E.split(m, 1, min_agreement=2)
    assert len(acc) == 1 and rev == [], "one run cannot vote against itself"


def test_disjoint_runs_deliver_union_and_review_all():
    m = E.merge([[n(0.0, 60)], [n(1.0, 67)]])
    acc, rev = E.split(m, 2, min_agreement=1)
    assert len(acc) == 2 and len(rev) == 2, "nothing was seen twice"


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
