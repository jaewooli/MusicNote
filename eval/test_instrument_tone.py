#!/usr/bin/env python3
"""transcribe.instrument_brightness / drum_hit_profile: real per-song
timbre measurements, tested on synthesised audio with a known answer.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import transcribe as T  # noqa: E402

SR = 22050


def _tone(midi, seconds, n_harmonics=1, sr=SR):
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, n_harmonics + 1))
    return (0.2 * y / max(1, n_harmonics)).astype("float32")


def _notes(midi, n=8, dur=0.3, gap=0.05):
    return [{"start": i * (dur + gap), "end": i * (dur + gap) + dur, "pitch": midi}
            for i in range(n)]


def test_a_richer_harmonic_series_measures_brighter():
    notes = _notes(57, n=6)
    dull = np.concatenate([_tone(57, 0.35) for _ in notes])
    bright = np.concatenate([_tone(57, 0.35, n_harmonics=5) for _ in notes])
    sal_dull = T._salience_cqt(dull, SR)
    sal_bright = T._salience_cqt(bright, SR)
    b_dull = T.instrument_brightness(notes, sal_dull)
    b_bright = T.instrument_brightness(notes, sal_bright)
    assert b_dull is not None and b_bright is not None
    assert b_bright > b_dull, (b_dull, b_bright)


def test_brightness_is_none_without_a_salience_map():
    assert T.instrument_brightness(_notes(57), {"C": None, "t": None}) is None


def test_brightness_is_none_for_empty_notes():
    y = _tone(57, 1.0)
    sal = T._salience_cqt(y, SR)
    assert T.instrument_brightness([], sal) is None


def _hit_train(times, sr=SR, dur=1.2, freq=4000.0, tau=0.006):
    """A short burst at each time, with a controllable decay time constant --
    stands in for a drum hit at some brightness/decay."""
    y = np.zeros(int(sr * dur), dtype="float32")
    t_axis = np.arange(int(sr * 0.14)) / sr
    click = (0.3 * np.sin(2 * np.pi * freq * t_axis) * np.exp(-t_axis / tau)).astype("float32")
    for t0 in times:
        i0 = int(t0 * sr)
        end = min(len(y), i0 + len(click))
        y[i0:end] += click[:end - i0]
    return y


def test_drum_hit_profile_reads_a_bright_hit_as_bright():
    times = [0.1, 0.4, 0.7, 1.0]
    y = _hit_train(times, freq=5000.0)
    notes = [{"start": t, "pitch": 42} for t in times]     # 42 = closed hihat
    profile = T.drum_hit_profile(y, SR, notes)
    assert "42" in profile, f"expected a string-keyed entry, got {list(profile)}"
    assert profile["42"]["centroid_hz"] > 1500, profile     # a real click, not the noise floor


def test_drum_hit_profile_ranks_a_faster_decay_shorter():
    times = [0.1, 0.4, 0.7, 1.0]
    fast = T.drum_hit_profile(_hit_train(times, tau=0.004), SR,
                              [{"start": t, "pitch": 42} for t in times])
    slow = T.drum_hit_profile(_hit_train(times, tau=0.05), SR,
                              [{"start": t, "pitch": 46} for t in times])
    assert fast["42"]["decay_s"] < slow["46"]["decay_s"], (fast, slow)


def test_drum_hit_profile_is_empty_on_silence():
    y = np.zeros(SR, dtype="float32")
    notes = [{"start": 0.1, "pitch": 36}]
    assert T.drum_hit_profile(y, SR, notes) == {}


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok   {name}")
    print("\nall instrument-tone tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
