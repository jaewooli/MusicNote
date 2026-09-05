#!/usr/bin/env python3
"""The MT3 worker's two memory guards, tested without loading a model.

Both exist because the kernel OOM-killed the worker mid-request on an 11 GB box
and the app reported it to the user as a dead worker after a 40 minute wait.
Neither guard has any visible effect when it works, so both are easy to break
silently — the batch cap in particular reaches into a third-party adapter and
would simply stop applying if upstream renamed the call.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Fix the environment BEFORE the import: the worker reads these at module level.
os.environ.setdefault("MT3_DEVICE", "cpu")
os.environ.pop("MT3_CHUNK_SEGMENTS", None)
os.environ.pop("MT3_BATCH_SEGMENTS", None)

import numpy as np                                              # noqa: E402
import mt3_worker as W                                          # noqa: E402


class FakeInner:
    """Stands in for the backend's `inference_file`, recording the batch size."""

    def __init__(self):
        self.seen = []

    def __call__(self, bsz=8, audio_segments=None):
        self.seen.append(bsz)
        return "tokens"


class FakeModel:
    def __init__(self):
        self.model = type("M", (), {})()
        self.model.inference_file = FakeInner()


def test_batch_cap_replaces_the_backends_own_batch():
    m = FakeModel()
    inner = m.model.inference_file
    W.MT3_BATCH = 2
    W._cap_batch(m)
    assert W._batch_capped, "the cap did not report success"
    m.model.inference_file(bsz=8, audio_segments=None)
    assert inner.seen == [2], f"backend still batched at {inner.seen}"


def test_cap_is_not_applied_twice():
    m = FakeModel()
    W.MT3_BATCH = 2
    W._cap_batch(m)
    first = m.model.inference_file
    W._cap_batch(m)
    assert m.model.inference_file is first, "wrapped the wrapper"
    assert W._batch_capped


def test_unknown_backend_is_left_alone_and_says_so():
    # A backend that batches somewhere else must not be patched blindly, and
    # must report failure so `_chunks` falls back to bounding memory itself.
    m = type("Other", (), {"model": type("M", (), {})()})()
    W.MT3_BATCH = 2
    W._cap_batch(m)
    assert not W._batch_capped, "claimed to cap a backend it cannot reach"


def test_chunking_is_off_when_the_cap_took():
    W.CHUNK_SEGMENTS = None
    y = np.zeros(int(30 * 16000), dtype=np.float32)
    blocks = list(W._chunks(y, 16000, fallback=False))
    assert len(blocks) == 1, f"whole file split into {len(blocks)} blocks anyway"
    assert blocks[0][1] == 0.0


def test_chunking_takes_over_when_the_cap_failed():
    W.CHUNK_SEGMENTS = None
    y = np.zeros(int(30 * 16000), dtype=np.float32)
    blocks = list(W._chunks(y, 16000, fallback=True))
    assert len(blocks) > 1, "no cap and no chunking: nothing bounds memory"
    # Every block but the last is a whole number of 2.048 s segments, so the
    # model sees exactly the windows it would have seen on the whole file.
    seg = int(round(W.SEGMENT_SECONDS * 16000))
    for block, _ in blocks[:-1]:
        assert len(block) % seg == 0, f"block of {len(block)} splits a segment"
    assert sum(len(b) for b, _ in blocks) == len(y), "chunking dropped audio"


def test_chunk_offsets_reconstruct_the_timeline():
    W.CHUNK_SEGMENTS = 3
    sr = 16000
    y = np.zeros(int(25 * sr), dtype=np.float32)
    at = 0.0
    for block, offset in W._chunks(y, sr):
        assert abs(offset - at) < 1e-9, f"offset {offset} should be {at}"
        at += len(block) / float(sr)
    W.CHUNK_SEGMENTS = None


def test_explicit_zero_disables_chunking_even_as_a_fallback():
    # "0" from the environment means the operator turned it off on purpose;
    # only an UNSET value leaves the fallback available.
    W.CHUNK_SEGMENTS = 0
    y = np.zeros(int(30 * 16000), dtype=np.float32)
    assert len(list(W._chunks(y, 16000, fallback=True))) == 1
    W.CHUNK_SEGMENTS = None


def test_trim_heap_is_safe_to_call():
    W._trim_heap()      # a no-op off glibc, but it must never raise




# --- per-part octave correction ------------------------------------------------
# MT3's biggest single error on band material is octave DISPLACEMENT of a whole
# instrument, and it is invisible to the shifted-run vote because every run makes
# it. The correction is guarded hard (it must beat "leave it alone" by a factor,
# over a part with enough notes) because the plain argmax scores WORSE than doing
# nothing — a harmonic comb an octave down explains everything the right one does
# plus more. These tests pin both directions: that it fires when it should, and
# that it stays out of the way when it should not.

def _tone(midi, seconds, sr=22050):
    """A harmonically rich tone, so the comb NMF has partials to fit."""
    import numpy as np
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3, 4, 5))
    return (0.2 * y).astype("float32")


def _octave_case(sounding_midi, written_midi, n_notes=60):
    """Write a clip that really sounds `sounding_midi`, hand the corrector notes
    written at `written_midi`, and report the shift it chose."""
    import sys as _s, tempfile
    import numpy as np
    import soundfile as sf
    _s.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import app as A
    dur = 0.25
    audio = np.concatenate([_tone(sounding_midi, dur) for _ in range(n_notes)])
    notes = [{"start": i * dur, "end": (i + 1) * dur, "pitch": written_midi,
              "velocity": 100, "track": 0, "program": 0, "is_drum": False}
             for i in range(n_notes)]
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/tone.wav"
        sf.write(path, audio, 22050)
        A._mt3_octaves(notes, path)
    return notes[0]["pitch"] - written_midi


def test_octave_up_whole_part_is_moved_back_down():
    got = _octave_case(sounding_midi=45, written_midi=57)   # really A2, written A3
    assert got == -12, f"part written an octave high was moved {got:+d}, not -12"


def test_correct_part_is_left_alone():
    got = _octave_case(sounding_midi=57, written_midi=57)
    assert got == 0, f"a correctly written part was moved {got:+d}"


def test_short_part_is_never_moved():
    # Below OCTAVE_MIN_NOTES the mean carries too little evidence to act on,
    # whatever the audio says.
    got = _octave_case(sounding_midi=45, written_midi=57, n_notes=5)
    assert got == 0, f"a 5-note part was moved {got:+d} on almost no evidence"


# --- bass rescue --------------------------------------------------------------
# A note the voting ensemble could never reach agreement on (every run is deaf
# below MIDI 36) is rescued from a transposed-up pass instead of being voted on.
# It must (1) actually get added when it's new and below the cutoff, (2) be
# skipped when the accepted set already has it (no duplicate), and (3) be left
# alone when it's at or above the cutoff — that register is not this pass's job.

def _note(pitch, start, end=None, is_drum=False):
    return {"start": start, "end": end if end is not None else start + 0.5,
            "pitch": pitch, "velocity": 100, "track": 0, "program": 0,
            "is_drum": is_drum}


def test_a_new_low_note_is_added():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import app as A
    accepted = [_note(60, 0.0)]
    rescue = [_note(30, 1.0)]
    extra = A._mt3_bass_rescue(rescue, accepted, total_runs=4)
    assert len(extra) == 1, f"expected the new low note to be rescued, got {extra}"
    assert extra[0]["pitch"] == 30
    assert extra[0]["agreement"] == 4, "a rescued note must survive the slider"


def test_a_note_already_accepted_is_not_duplicated():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import app as A
    accepted = [_note(30, 1.0)]
    rescue = [_note(30, 1.02)]        # same pitch, well within the 50ms match
    extra = A._mt3_bass_rescue(rescue, accepted, total_runs=4)
    assert extra == [], f"a note the vote already has must not be duplicated: {extra}"


def test_a_note_at_or_above_cutoff_is_left_alone():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import app as A
    accepted = []
    rescue = [_note(A.BASS_RESCUE_CUTOFF, 1.0), _note(A.BASS_RESCUE_CUTOFF + 12, 1.0)]
    extra = A._mt3_bass_rescue(rescue, accepted, total_runs=4)
    assert extra == [], "the cutoff register is the base ensemble's job, not rescue's"


def test_drum_hits_are_never_rescued():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    import app as A
    accepted = []
    rescue = [_note(36, 1.0, is_drum=True)]     # MIDI 36 is a kick, not a bass note
    extra = A._mt3_bass_rescue(rescue, accepted, total_runs=4)
    assert extra == [], "a drum slot has no register to rescue"


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok   {name}")
    print("\nall mt3-worker tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
