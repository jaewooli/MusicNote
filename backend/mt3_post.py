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
# NOTE: YourMT3 emits a constant velocity of 100 for every note, so on that
# model this gate never fires and the sensitivity slider has no effect on the
# MT3 path at all. Do not add a length-based substitute — see MIN_NOTE_SPAN.
# Giving MT3 a real sensitivity control needs a per-note confidence the model
# does not currently expose.
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
