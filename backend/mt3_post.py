"""Post-processing applied to raw MT3 worker output before notation.

Extracted from ``app._mt3_stems`` so that the gate can be measured and tuned
offline against ``eval/refs`` without paying for MT3 inference every time.

The gate is the last place a correctly transcribed note can still be deleted,
so its parameters are explicit and overridable rather than inline constants.
"""
from __future__ import annotations

import os

# Velocity floor at sensitivity 0. Scaled by (1 - sensitivity).
#
# NOTE: YourMT3 emits a constant velocity of 100 for every note, so this gate
# never fires on that model. Do not add a length-based substitute — see
# MIN_NOTE_SPAN. The sensitivity control the model could not provide now comes
# from cross-run agreement instead; see required_agreement() at the bottom.
VELOCITY_SPAN = float(os.environ.get("MUSICNOTE_MT3_VELOCITY_SPAN", "18"))

# Minimum note length in seconds: MIN_NOTE_BASE + MIN_NOTE_SPAN * (1 - s).
#
# This is a sanity floor for degenerate zero-length events, NOT a filter.
# Measured on eval/refs with cached MT3 output (eval/replay_eval.py), the old
# values (0.05 + 0.06 -> 80 ms at the default sensitivity) deleted 542 of 610
# MT3 notes on ref00 and pinned note F1 at 0.175; with no length gate the same
# MT3 output scores 0.865. MT3's median note is 30 ms because it emits discrete
# note events, so a length threshold sized for frame-peak detectors removes
# almost the entire transcription.
#
# Any value at or below 8 ms scored identically, so the floor is deliberately
# far from the region where it starts costing recall.
MIN_NOTE_BASE = float(os.environ.get("MUSICNOTE_MT3_MIN_NOTE_BASE", "0.005"))
# Sensitivity must not shorten-gate notes: that is what made the slider act as
# a "delete fast passages" control. Length is a validity check; sensitivity is
# applied through the velocity floor instead.
MIN_NOTE_SPAN = float(os.environ.get("MUSICNOTE_MT3_MIN_NOTE_SPAN", "0.0"))


def gate_params(sensitivity: float) -> tuple[int, float]:
    """Return (velocity floor, minimum note seconds) for a sensitivity."""
    s = max(0.0, min(1.0, sensitivity))
    return int(round(VELOCITY_SPAN * (1.0 - s))), MIN_NOTE_BASE + MIN_NOTE_SPAN * (1.0 - s)


def gate(raw: list[dict], sensitivity: float) -> tuple[list[dict], dict]:
    """Drop notes below the velocity / length floor.

    Returns the surviving notes and a report of what was removed, so that
    over-filtering is visible in logs and in the eval harness instead of
    silently reducing recall.
    """
    amp_thr, min_len = gate_params(sensitivity)
    kept, by_velocity, by_length = [], 0, 0
    for n in raw:
        if int(n["velocity"]) < amp_thr:
            by_velocity += 1
            continue
        if (float(n["end"]) - float(n["start"])) < min_len:
            by_length += 1
            continue
        kept.append(n)
    return kept, {"input": len(raw), "kept": len(kept),
                  "dropped_velocity": by_velocity, "dropped_length": by_length,
                  "velocity_floor": amp_thr, "min_note_seconds": round(min_len, 4)}


# --- per-note confidence from cross-run agreement ----------------------------
#
# YourMT3 reports a constant velocity of 100, so for a long time the MT3 path
# had no per-note confidence at all: `conf` was 0.55 + 0.4*(100/127) = 0.865 for
# every single note, and the "uncertain note" highlight in the UI meant nothing.
#
# Running four passes at shifted segment boundaries makes agreement a real
# signal. Measured against the reference MIDI (onset within 50 ms, same pitch)
# over 10433 merged notes on both eval sets:
#
#   합의  밴드: 맞을 확률 / 옥타브유령    솔로: 맞을 확률 / 옥타브유령
#     1        27.6%  /  13.6%              24.2%  /  22.3%
#     2        41.8%  /  10.8%              42.1%  /  26.2%
#     3        51.1%  /   6.3%              89.4%  /   5.2%
#     4        88.5%  /   2.8%              99.6%  /   0.1%
#
# AUC for "is this note correct" is 0.755 on band material and 0.964 on solo
# piano. For comparison, the spectral octave features tried earlier scored 0.5,
# i.e. nothing. Octave ghosts concentrate at low agreement (13.6% vs 2.8%), so
# this is also the first thing that reliably MARKS them — though deleting them
# still does not pay, see AGREEMENT_OCTAVE_NOTE below.
#
# The table is the pooled rate across both sets. It is only calibrated for a
# four-run ensemble; other run counts fall back to the plain fraction.
_CONF_BY_AGREEMENT_OF_4 = {1: 0.27, 2: 0.42, 3: 0.61, 4: 0.96}

# The shipping ensemble is three runs (0, 1.024, 1.536), so this is the table
# that is actually used. Measured the same way after the merge key stopped
# keying on the track, over 8539 merged notes:
#
#   합의   밴드 맞을 확률 / 옥타브유령    솔로 맞을 확률 / 옥타브유령
#     1        24.2%  /  14.7%              30.4%  /  22.9%
#     2        57.1%  /  10.3%              76.8%  /   9.1%
#     3        86.8%  /   2.2%              99.4%  /   0.2%
#
# AUC 0.811 on band and 0.946 on solo — a better separator than the four-run
# version scored (0.755 / 0.964), because dropping the track from the key stops
# splitting one real note across several agreement-1 entries.
#
# At the shipping threshold (agreement >= 2) every delivered note is therefore
# 0.61 or 0.94. Nothing lands under 0.5, so the score prints clean; the piano
# roll, which outlines anything under 0.7, still shows which notes are the
# shaky ones. That division is deliberate: the score is for reading, the roll
# is for checking.
_CONF_BY_AGREEMENT_OF_3 = {1: 0.25, 2: 0.61, 3: 0.94}

# Dropping a low-agreement note that sits an octave above a simultaneous
# higher-agreement note was measured and does NOT help: on band material note F1
# went 0.671 -> 0.666/0.669/0.671 for agreement gaps of 1/2/3, and on solo piano
# 0.951 -> 0.948/0.953/0.951. It buys precision and loses more recall. This is
# the second independent method to fail at octave removal (the first was
# spectral, F1 change +0.000), so octaves are marked, never deleted.
AGREEMENT_OCTAVE_NOTE = "measured twice; removal does not improve F1"


def confidence(agreement: int, runs: int) -> float:
    """0..1 chance this note is real, from how many runs found it."""
    a = max(1, int(agreement))
    r = max(1, int(runs))
    if r == 1:
        return 0.75          # single run: no evidence either way
    if r == 3:
        return _CONF_BY_AGREEMENT_OF_3.get(min(a, 3), 0.94)
    if r == 4:
        return _CONF_BY_AGREEMENT_OF_4.get(min(a, 4), 0.96)
    # Uncalibrated run count: a plain fraction, deliberately not dressed up as
    # a measured probability.
    return round(min(1.0, 0.20 + 0.78 * (a / r)), 2)


def required_agreement(sensitivity: float, runs: int) -> int:
    """Map the sensitivity slider onto the ensemble vote.

    This is what finally gives the slider an effect on the MT3 path. It used to
    threshold `velocity`, which YourMT3 pins at 100, so the control did nothing
    at all (see the note on VELOCITY_SPAN above). Agreement is a real axis:
    sensitivity 1 keeps everything the model ever proposed, 0 keeps only what
    every run agreed on. The default 0.5 with four runs gives 2, which is the
    configuration measured in ACCURACY.md.
    """
    r = max(1, int(runs))
    s = max(0.0, min(1.0, float(sensitivity)))
    return int(max(1, min(r, round(r - s * (r - 1)))))
