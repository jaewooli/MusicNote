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


def test_tag_notes_carries_measured_brightness():
    out = A._tag_notes([_note(40)], "track33", "베이스", brightness=0.62)
    assert out[0]["brightness"] == 0.62


def test_tag_notes_looks_up_drum_profile_by_pitch():
    # Real key is a string (drum_hit_profile's docstring: json round-trips an
    # int dict key back as a string), so the lookup must tolerate that too.
    profile = {"36": {"centroid_hz": 120.0, "decay_s": 0.2},
              "38": {"centroid_hz": 1900.0, "decay_s": 0.15}}
    out = A._tag_notes([_note(36), _note(41)], "drums", "드럼",
                       is_drum=True, drum_profile=profile)
    assert out[0]["drum_centroid"] == 120.0 and out[0]["drum_decay"] == 0.2
    assert "drum_centroid" not in out[1], "an unmeasured pitch must not get a stray profile"


def test_merge_stems_propagates_brightness_and_drum_profile():
    stems = [
        {"id": "track33", "label": "베이스", "family": "bass", "program": 33,
         "pitched": True, "notes": [_note(40, 0.0)],
         "instrument": {"features": {"brightness": 0.3}}},
        {"id": "drums", "label": "드럼", "family": "percussion", "program": 0,
         "pitched": False, "notes": [_note(36, 0.5)],
         "drum_profile": {"36": {"centroid_hz": 110.0, "decay_s": 0.18}}},
    ]
    notes, _contour = A._merge_stems(stems)
    by_stem = {n["stem"]: n for n in notes}
    assert by_stem["track33"]["brightness"] == 0.3
    assert by_stem["drums"]["drum_centroid"] == 110.0


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok   {name}")
    print("\nall note-tagging tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
