#!/usr/bin/env python3
"""app._tag_notes / _merge_stems carry `family`/`program`/`is_drum` per note.

The frontend synth (Player.FAMILY_VOICE in musicnote-core.js) picks a timbre
from these fields; before this, a note only carried `stem`/`inst` and every
note played through one generic tone regardless of instrument.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import app as A  # noqa: E402


def _note(pitch, start=0.0):
    return {"start": start, "end": start + 0.5, "pitch": pitch, "velocity": 100}


def test_tag_notes_carries_family_program_and_drum_flag():
    out = A._tag_notes([_note(40)], "track33", "베이스", family="bass",
                       program=33, is_drum=False)
    assert out[0]["family"] == "bass"
    assert out[0]["program"] == 33
    assert out[0]["is_drum"] is False
    assert out[0]["stem"] == "track33" and out[0]["inst"] == "베이스"


def test_tag_notes_omits_family_program_when_not_given():
    # A caller with nothing to say about family (the plain CQT engines) must
    # not stamp a misleading family=None onto every note.
    out = A._tag_notes([_note(60)], "lead", "멜로디")
    assert "family" not in out[0] and "program" not in out[0]
    assert out[0]["is_drum"] is False


def test_merge_stems_propagates_family_and_drum_flag_per_stem():
    stems = [
        {"id": "track33", "label": "베이스", "family": "bass", "program": 33,
         "pitched": True, "notes": [_note(40, 0.0)]},
        {"id": "drums", "label": "드럼", "family": "percussion", "program": 0,
         "pitched": False, "notes": [_note(36, 0.5)]},
    ]
    notes, _contour = A._merge_stems(stems)
    by_stem = {n["stem"]: n for n in notes}
    assert by_stem["track33"]["family"] == "bass"
    assert by_stem["track33"]["is_drum"] is False
    assert by_stem["drums"]["family"] == "percussion"
    assert by_stem["drums"]["is_drum"] is True


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok   {name}")
    print("\nall note-tagging tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
