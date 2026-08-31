"""Regression tests for the notation layer: staves, printed voices, key.

These cover the shape of the printed page rather than which notes were heard.
A transcript can be perfectly accurate and still be unreadable, which is what
these guard against.

Run with:
    python eval/test_score_layout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from score_build import MAX_VOICES_PER_STAFF, _beat_grid, build_score
from score_model import DIVISIONS


def line(pitches, t0=0.0, step=0.5, dur=0.45):
    return [{"start": t0 + i * step, "end": t0 + i * step + dur,
             "pitch": p, "velocity": 90} for i, p in enumerate(pitches)]


def check(label, cond, detail=""):
    assert cond, f"{label}: {detail}"
    print(f"{label}: ok")


# --- a part handed pre-separated `voices` must still get a key and a length ---
# The key estimate and the total duration both read the part's notes. They used
# to look only at `notes`, so every MT3 job (which supplies `voices`) came out
# in C major with a beat grid one second long, dropping most of the piece.
d_major = line([74, 78, 81, 78, 74, 69, 66, 62], step=0.5)
doc = build_score([{"name": "P", "voices": [d_major]}], tempo=120, time_sig=(4, 4))
check("key_read_from_voices", doc.key_fifths == 2, f"fifths={doc.key_fifths}")
n_meas = max(len(v.measures) for p in doc.parts for v in p.voices)
check("length_read_from_voices", n_meas >= 2, f"{n_meas} measures for a 4 s clip")
check("no_notes_lost", doc.note_count() >= len(d_major),
      f"{doc.note_count()} of {len(d_major)}")


# --- a keyboard-range part is split onto a grand staff ----------------------
right = line([72, 76, 79, 76, 72, 76, 79, 84], step=0.5)
left = line([36, 43, 40, 47, 36, 43, 40, 47], step=0.5)
doc = build_score([{"name": "Piano", "voices": [right, left]}],
                  tempo=120, time_sig=(4, 4))
part = doc.parts[0]
check("grand_staff", part.staves == 2, f"staves={part.staves}")
check("grand_staff_clefs", part.clefs == ["treble", "bass"], str(part.clefs))
check("voices_land_on_both_staves",
      {v.staff for v in part.voices} == {1, 2},
      str([(v.number, v.staff) for v in part.voices]))


# --- one narrow register stays on one staff ---------------------------------
doc = build_score([{"name": "Flute",
                    "voices": [line([72, 74, 76, 77]), line([79, 81, 83, 84])]}],
                  tempo=120, time_sig=(4, 4))
check("narrow_range_stays_one_staff", doc.parts[0].staves == 1,
      f"staves={doc.parts[0].staves}")


# --- no staff ever prints more than two voices ------------------------------
# Seven notation voices on one staff was the single biggest source of clutter:
# each one draws a full layer of rests on top of the others.
many = [line([84, 83, 81], step=0.7), line([79, 77, 76], step=0.55),
        line([74, 72, 71], step=0.45), line([69, 67, 66], step=0.65),
        line([64, 62, 61], step=0.5), line([59, 57, 55], step=0.6),
        line([52, 50, 48], step=0.75)]
doc = build_score([{"name": "Piano", "voices": many}], tempo=120, time_sig=(4, 4))
part = doc.parts[0]
per_staff: dict[int, int] = {}
for v in part.voices:
    per_staff[v.staff] = per_staff.get(v.staff, 0) + 1
check("voices_per_staff_capped",
      all(c <= MAX_VOICES_PER_STAFF for c in per_staff.values()), str(per_staff))
check("merging_keeps_every_pitch",
      {n.midi for v in part.voices for m in v.measures
       for e in m.events for n in e.notes}
      >= {int(n["pitch"]) for v in many for n in v},
      "a pitch disappeared while folding voices together")

# --- a printed voice never overlaps itself ----------------------------------
# The measure walker assumes one event at a time; an overlap there silently
# drops whichever note started second.
for v in part.voices:
    for m in v.measures:
        total = sum(e.dur for e in m.events)
        check_total = total <= DIVISIONS * 4 + 1
        assert check_total, (v.number, m.number, total)
print("voice_fits_its_measure: ok")

# --- the beat grid must be uniform, including where it is extrapolated ------
# Clamping the leading extrapolated beat to t=0 made a short first beat. Onsets
# inside it then read as odd fractions, `_pick_subdiv` reached for a 32nd grid
# to explain them, and a run of plain eighths came out as a dotted-16th rest
# plus off-beat eighths that no longer beamed by beat.
per = 0.448
detected = [0.406 + i * per for i in range(12)]
grid = _beat_grid(detected, 60.0 / per, 5.0)
gaps = [round(grid[i + 1] - grid[i], 6) for i in range(len(grid) - 1)]
check("beat_grid_uniform", len(set(gaps)) == 1, f"periods={sorted(set(gaps))}")
check("beat_grid_covers_zero", grid[0] <= 0.0 < grid[1], f"starts at {grid[0]:.3f}")
check("beat_grid_keeps_detected_phase",
      any(abs(g - 0.406) < 1e-6 for g in grid), "detected beat times were moved")

# The same run of eighths must come out as eighths, not as a 32nd-grid mess.
eighth = per / 2
notes = [{"start": 0.159 + i * eighth, "end": 0.159 + (i + 1) * eighth,
          "pitch": 60 + i % 5, "velocity": 90} for i in range(16)]
doc = build_score([{"name": "P", "voices": [notes]}], beats=detected,
                  tempo=60.0 / per, time_sig=(4, 4))
kinds = {e.type for v in doc.parts[0].voices for m in v.measures
         for e in m.events if e.notes}
check("even_eighths_stay_eighths", kinds == {"eighth"}, f"got {sorted(kinds)}")

# --- a repeated note must stay two noteheads --------------------------------
# Two attacks of one pitch rounding onto the same grid tick used to be
# deduplicated into a single notehead. That is a deleted note, not a rounded
# rhythm, and it cost 56 notes across eval/refs.
per = 0.5
beats = [i * per for i in range(1, 17)]
rep = []
for i in range(8):                       # pairs of fast repeated notes
    t = i * per
    rep += [{"start": t, "end": t + 0.11, "pitch": 67, "velocity": 90},
            {"start": t + 0.12, "end": t + 0.23, "pitch": 67, "velocity": 90}]
doc = build_score([{"name": "P", "voices": [rep]}], beats=beats, tempo=120,
                  time_sig=(4, 4))
heads = sum(1 for p_ in doc.parts for v in p_.voices for m in v.measures
            for e in m.events for n in e.notes if not n.tie_stop)
check("repeated_note_not_deduplicated", heads >= len(rep),
      f"{heads} noteheads for {len(rep)} attacks")


# --- every printed measure fits inside its own bar --------------------------
# VexFlow lays notes out by accumulated ticks, so a measure printing more than a
# measure pushes its own notes past the barline and everything after it off the
# beat. Half of all measures used to come out the wrong length.
def measure_ticks(doc):
    num, den = doc.time_sig
    tpb = DIVISIONS * 4 // den
    return tpb * (num * (4 // den) if den <= 4 else num * 4 // den)


busy = []
for i in range(32):                      # 16ths with irregular releases
    t = i * 0.125
    busy.append({"start": t, "end": t + 0.06 + 0.05 * (i % 3),
                 "pitch": 60 + (i * 5) % 13, "velocity": 90})
doc = build_score([{"name": "P", "voices": [busy]}], beats=beats, tempo=120,
                  time_sig=(4, 4))
tpm = measure_ticks(doc)
over = [(v.number, m.number, sum(e.dur for e in m.events))
        for p_ in doc.parts for v in p_.voices for m in v.measures
        if sum(e.dur for e in m.events) > tpm]
check("no_measure_overflows_its_bar", not over, f"{over} against tpm={tpm}")

# and the notes all survived that
heads = sum(1 for p_ in doc.parts for v in p_.voices for m in v.measures
            for e in m.events for n in e.notes if not n.tie_stop)
check("busy_line_keeps_its_notes", heads >= len(busy),
      f"{heads} noteheads for {len(busy)} notes")

# --- barlines follow the detected downbeat ----------------------------------
# meter.detect locates downbeats from accent evidence, and the score used to
# ignore them and count bars from wherever the extrapolated grid began. On 5 of
# 7 measured clips that put every barline in the wrong place, one of them by 3
# beats, so a bar could open on a rest that the music does not have.
per = 0.5
beats = [i * per for i in range(24)]
downbeats = [beats[i] for i in range(2, 24, 4)]      # bar starts on beat index 2
tune = [{"start": beats[i], "end": beats[i] + 0.45,
         "pitch": 72 if i in (2, 6, 10, 14, 18) else 67, "velocity": 90}
        for i in range(2, 22)]
doc = build_score([{"name": "P", "voices": [tune]}], beats=beats, tempo=120,
                  time_sig=(4, 4), downbeats=downbeats)
bars = [m for p_ in doc.parts for v in p_.voices for m in v.measures]
span = bars[0].end - bars[0].start
worst = max(min(abs(d - (bars[0].start + k * span)) for k in range(len(bars) + 2))
            for d in downbeats)
check("barlines_land_on_downbeats", worst < 1e-6, f"worst offset {worst:.3f}s")
check("no_empty_leading_bar",
      any(not e.is_rest for e in bars[0].events),
      "the score opens on a bar of rests that stands for no audio")

# ignoring the downbeats must put them somewhere else, or the test proves nothing
doc0 = build_score([{"name": "P", "voices": [tune]}], beats=beats, tempo=120,
                   time_sig=(4, 4))
b0 = [m for p_ in doc0.parts for v in p_.voices for m in v.measures]
sp0 = b0[0].end - b0[0].start
worst0 = max(min(abs(d - (b0[0].start + k * sp0)) for k in range(len(b0) + 2))
             for d in downbeats)
check("downbeats_actually_change_the_bars", worst0 > 1e-6,
      "the fixture cannot detect a regression")

print("\nall score-layout checks passed")
