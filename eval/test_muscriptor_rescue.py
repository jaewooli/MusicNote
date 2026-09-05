#!/usr/bin/env python3
"""app._muscriptor_fetch / _muscriptor_timbre_rescue / _muscriptor_corroboration,
tested without a real MuScriptor worker.

Two independent, complementary rescues share one fetch:

* timbre rescue -- gated on an instrument LABEL (MuScriptor's own timbre
  vocabulary), not a pitch cutoff -- see MUSCRIPTOR_TIMBRES in backend/app.py.
* corroboration -- family-agnostic: promotes a review-queue note (agreement 1
  of the shifted-run vote) that MuScriptor independently also found.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import app as A  # noqa: E402


def _note(pitch, start, instrument=None, end=None, agreement=None, is_drum=False):
    d = {"start": start, "end": end if end is not None else start + 0.5,
         "pitch": pitch, "velocity": 100, "track": 0, "program": 33,
         "is_drum": is_drum}
    if instrument is not None:
        d["instrument"] = instrument
    if agreement is not None:
        d["agreement"] = agreement
    return d


# --- _muscriptor_fetch ---------------------------------------------------

def test_fetch_returns_notes_when_the_worker_answers():
    A.MU.available = lambda: True
    A.MU.transcribe = lambda wav_path: {"notes": [_note(40, 1.0, "electric_bass")]}
    try:
        notes = A._muscriptor_fetch("dummy.wav")
    finally:
        del A.MU.available, A.MU.transcribe
    assert notes == [_note(40, 1.0, "electric_bass")]


def test_fetch_returns_none_when_worker_unavailable():
    A.MU.available = lambda: False
    try:
        notes = A._muscriptor_fetch("dummy.wav")
    finally:
        del A.MU.available
    assert notes is None, "a down worker must degrade to 'nothing to rescue with', not an error"


def test_fetch_returns_none_on_worker_error():
    A.MU.available = lambda: True

    def _boom(wav_path):
        raise RuntimeError("worker restarted mid-request")
    A.MU.transcribe = _boom
    try:
        notes = A._muscriptor_fetch("dummy.wav")
    finally:
        del A.MU.available, A.MU.transcribe
    assert notes is None, "a worker exception must not lose the transcription already in hand"


def test_fetch_respects_the_toggle_without_calling_the_worker():
    called = []
    A.MU.available = lambda: called.append(True) or True
    A.MUSCRIPTOR_RESCUE = False
    try:
        notes = A._muscriptor_fetch("dummy.wav")
    finally:
        A.MUSCRIPTOR_RESCUE = True
        del A.MU.available
    assert notes is None and not called, "the toggle should skip the worker call, not just the result"


# --- _muscriptor_timbre_rescue --------------------------------------------

def test_a_flagged_instrument_note_is_added():
    mus_notes = [_note(40, 1.0, "electric_bass")]
    extra = A._muscriptor_timbre_rescue(mus_notes, accepted=[], total_runs=4)
    assert len(extra) == 1, f"expected the flagged-instrument note, got {extra}"
    assert extra[0]["instrument"] == "electric_bass"
    assert extra[0]["agreement"] == 4, "a rescued note must survive the slider"


def test_a_non_flagged_instrument_is_left_alone():
    # acoustic_bass is NOT in MUSCRIPTOR_TIMBRES -- YourMT3 already recalls it
    # well (84.2%), so a second model's opinion there is not trusted alone.
    mus_notes = [_note(40, 1.0, "acoustic_bass")]
    extra = A._muscriptor_timbre_rescue(mus_notes, accepted=[], total_runs=4)
    assert extra == [], f"a family YourMT3 already covers well must not be rescued: {extra}"


def test_timbre_rescue_does_not_duplicate_an_accepted_note():
    mus_notes = [_note(40, 1.02, "electric_bass")]   # within the 50ms match
    accepted = [_note(40, 1.0)]
    extra = A._muscriptor_timbre_rescue(mus_notes, accepted, total_runs=4)
    assert extra == [], f"a note the ensemble already has must not be duplicated: {extra}"


def test_timbre_rescue_with_no_notes_is_a_noop():
    assert A._muscriptor_timbre_rescue(None, accepted=[], total_runs=4) == []


# --- _muscriptor_corroboration ---------------------------------------------

def test_a_corroborated_review_note_is_promoted():
    merged = [_note(52, 2.0, agreement=1)]
    mus_notes = [_note(52, 2.02, "clarinet")]     # independent model, same note
    promoted = A._muscriptor_corroboration(mus_notes, merged, accepted=[], total_runs=4)
    assert promoted == [merged[0]], "promotion mutates the merged entry, not a copy"
    assert merged[0]["agreement"] == 4, "a promoted note must survive the slider like any rescue"
    assert merged[0]["source"] == "muscriptor_corroboration"


def test_an_uncorroborated_review_note_stays_put():
    merged = [_note(52, 2.0, agreement=1)]
    mus_notes = [_note(60, 2.0, "clarinet")]      # different pitch, no corroboration
    promoted = A._muscriptor_corroboration(mus_notes, merged, accepted=[], total_runs=4)
    assert promoted == []
    assert merged[0]["agreement"] == 1, "an uncorroborated note must not be touched"


def test_a_note_already_accepted_another_way_is_not_repromoted():
    # agreement>=2 notes are not the review queue's job to promote -- and a
    # note that already made it into `accepted` some other way gains nothing
    # from a second, near-duplicate entry (mir_eval's one-to-one matching
    # would score the duplicate as a false positive, not a second true one).
    merged = [_note(52, 2.0, agreement=1)]
    accepted = [_note(52, 2.0)]
    mus_notes = [_note(52, 2.0, "clarinet")]
    promoted = A._muscriptor_corroboration(mus_notes, merged, accepted, total_runs=4)
    assert promoted == []
    assert merged[0]["agreement"] == 1


def test_corroboration_ignores_drum_hits():
    merged = [_note(38, 2.0, agreement=1, is_drum=True)]
    mus_notes = [_note(38, 2.0, "drums")]
    promoted = A._muscriptor_corroboration(mus_notes, merged, accepted=[], total_runs=4)
    assert promoted == [], "a drum slot has no independent pitch to corroborate"


def test_corroboration_with_no_notes_is_a_noop():
    merged = [_note(52, 2.0, agreement=1)]
    assert A._muscriptor_corroboration(None, merged, accepted=[], total_runs=4) == []
    assert merged[0]["agreement"] == 1


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok   {name}")
    print("\nall muscriptor-rescue tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
