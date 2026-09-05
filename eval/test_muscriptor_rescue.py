#!/usr/bin/env python3
"""app._muscriptor_timbre_rescue, tested without a real MuScriptor worker.

Mirrors eval/test_mt3_worker.py's bass-rescue tests: same shape (a note not
already accepted gets pulled in with full agreement, a duplicate does not, a
missing/misbehaving worker must not break the job), but gated on an
instrument LABEL (MuScriptor's own timbre vocabulary) instead of a pitch
cutoff -- see MUSCRIPTOR_TIMBRES in backend/app.py for why.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import app as A  # noqa: E402


def _note(pitch, start, instrument, end=None):
    return {"start": start, "end": end if end is not None else start + 0.5,
            "pitch": pitch, "velocity": 100, "track": 0, "program": 33,
            "instrument": instrument}


def test_a_flagged_instrument_note_is_added():
    A.MU.available = lambda: True
    A.MU.transcribe = lambda wav_path: {
        "notes": [_note(40, 1.0, "electric_bass")]}
    try:
        extra = A._muscriptor_timbre_rescue("dummy.wav", accepted=[], total_runs=4)
    finally:
        del A.MU.available, A.MU.transcribe
    assert len(extra) == 1, f"expected the flagged-instrument note, got {extra}"
    assert extra[0]["instrument"] == "electric_bass"
    assert extra[0]["agreement"] == 4, "a rescued note must survive the slider"


def test_a_non_flagged_instrument_is_left_alone():
    # acoustic_bass is NOT in MUSCRIPTOR_TIMBRES -- YourMT3 already recalls it
    # well (84.2%), so a second model's opinion there is not trusted alone.
    A.MU.available = lambda: True
    A.MU.transcribe = lambda wav_path: {
        "notes": [_note(40, 1.0, "acoustic_bass")]}
    try:
        extra = A._muscriptor_timbre_rescue("dummy.wav", accepted=[], total_runs=4)
    finally:
        del A.MU.available, A.MU.transcribe
    assert extra == [], f"a family YourMT3 already covers well must not be rescued: {extra}"


def test_a_note_already_accepted_is_not_duplicated():
    A.MU.available = lambda: True
    A.MU.transcribe = lambda wav_path: {
        "notes": [_note(40, 1.02, "electric_bass")]}   # within the 50ms match
    accepted = [{"start": 1.0, "end": 1.5, "pitch": 40, "is_drum": False}]
    try:
        extra = A._muscriptor_timbre_rescue("dummy.wav", accepted, total_runs=4)
    finally:
        del A.MU.available, A.MU.transcribe
    assert extra == [], f"a note the ensemble already has must not be duplicated: {extra}"


def test_worker_unavailable_returns_empty_without_raising():
    A.MU.available = lambda: False
    try:
        extra = A._muscriptor_timbre_rescue("dummy.wav", accepted=[], total_runs=4)
    finally:
        del A.MU.available
    assert extra == [], "a down worker must degrade to no rescue, not an error"


def test_worker_error_is_swallowed():
    A.MU.available = lambda: True

    def _boom(wav_path):
        raise RuntimeError("worker restarted mid-request")
    A.MU.transcribe = _boom
    try:
        extra = A._muscriptor_timbre_rescue("dummy.wav", accepted=[], total_runs=4)
    finally:
        del A.MU.available, A.MU.transcribe
    assert extra == [], "a worker exception must not lose the transcription already in hand"


def test_toggle_off_skips_the_call_entirely():
    called = []
    A.MU.available = lambda: called.append(True) or True
    A.MUSCRIPTOR_RESCUE = False
    try:
        extra = A._muscriptor_timbre_rescue("dummy.wav", accepted=[], total_runs=4)
    finally:
        A.MUSCRIPTOR_RESCUE = True
        del A.MU.available
    assert extra == [] and not called, "the toggle should skip the worker call, not just the result"


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok   {name}")
    print("\nall muscriptor-rescue tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
