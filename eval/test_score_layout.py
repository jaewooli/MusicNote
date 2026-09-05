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
from score_build import (MAX_VOICES_PER_STAFF, VOICE_OVERFLOW_COST,
                         VOICE_OVERFLOW_LIMIT, MIN_TICKS, _beat_grid,
                         _reduce_voices, build_score)
from score_model import (DIVISIONS, _DRUM_MAP as DRUM_MAP, drum_spell,
                         krumhansl_fifths, spell)


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


# --- a minor key gets its OWN signature, not its parallel major's -----------
# Scoring only the major profile put a G minor piece in G major and printed an
# accidental on every B-flat and E-flat in it.
g_minor = [55, 58, 62, 63, 62, 58, 55, 65, 62, 58, 70, 67, 63, 58, 55, 62, 58, 55]
check("minor_key_uses_the_relative_signature",
      krumhansl_fifths(g_minor) == -2, f"fifths={krumhansl_fifths(g_minor)}")

# --- signatures past four flats exist ---------------------------------------
d_flat = [61, 63, 65, 66, 68, 70, 72, 61, 68, 66, 61, 63, 65, 61, 70, 68, 66, 61]
check("five_flats_is_reachable", krumhansl_fifths(d_flat) == -5,
      f"fifths={krumhansl_fifths(d_flat)}")


# --- two kit pieces that play together need two lines -----------------------
# `_emit` prints one notehead per drawn line, so a kit piece sharing a line with
# another is deleted whenever both are struck at once. Sharing is deliberate for
# a pair that IS one drawn note (two crash cymbals); it was an accident for the
# bass drum and the low floor tom, and cost 69 of the 71 collisions measured
# over eval/refs_band.
KIT_PAIRS_THAT_MAY_SHARE = {
    (35, 36), (38, 40), (37, 39), (49, 57), (51, 59), (52, 55), (54, 56),
}
lines = {}
for piece in sorted(DRUM_MAP):          # named pieces only — the fallback line
    lines.setdefault(drum_spell(piece), []).append(piece)   # is shared on purpose
clashes = [tuple(v) for v in lines.values() if len(v) > 1
           and any((a, b) not in KIT_PAIRS_THAT_MAY_SHARE
                   for i, a in enumerate(v) for b in v[i + 1:])]
check("kit_pieces_do_not_share_a_line", not clashes, f"{clashes} draw the same")
check("kick_and_floor_tom_differ", drum_spell(36) != drum_spell(41),
      f"both draw as {drum_spell(36)}")


# --- notes are spelled for the signature, not for the sign of `fifths` ------
# The two fixed tables this replaced chose sharps or flats by whether `fifths`
# was positive, which cannot spell a key past four accidentals and had no way to
# say "natural" at all.
def spelled(midi, fifths):
    st, al, oc = spell(midi, fifths)
    return st + ("#" * al if al > 0 else "b" * -al)

check("f_sharp_major_spells_its_own_seventh", spelled(65, 6) == "E#",
      f"got {spelled(65, 6)}")
check("g_flat_major_spells_its_own_fourth", spelled(71, -6) == "Cb",
      f"got {spelled(71, -6)}")
check("chromatic_white_key_stays_natural", spelled(65, 2) == "F",
      f"a chromatic F in D major came out as {spelled(65, 2)}")
check("no_double_accidentals", all(abs(spell(m, f)[1]) <= 1
                                   for f in range(-6, 7) for m in range(21, 109)),
      "a double sharp or flat reached the page")
check("cb_is_written_an_octave_up", spell(71, -6)[2] == 5,
      f"octave {spell(71, -6)[2]} for the C-flat that sounds as B4")


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


# --- a staff never prints more than three voices ----------------------------
# Seven notation voices on one staff was the single biggest source of clutter:
# each one draws a full layer of rests on top of the others. The ceiling is two,
# with a third allowed only where folding to two would cut held notes short
# (see score_build.VOICE_OVERFLOW_LIMIT) — this input is exactly that case.
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
      all(c <= VOICE_OVERFLOW_LIMIT for c in per_staff.values()), str(per_staff))
# --- the third voice has to be earned ---------------------------------------
# Two lines that never overlap fold into one voice for free; merging costs
# nothing, so nothing is bought by keeping them apart.
tidy = [line([72, 74, 76, 77], t0=0.0, step=0.5, dur=0.4),
        line([60, 62, 64, 65], t0=0.25, step=0.5, dur=0.2)]
doc = build_score([{"name": "Piano", "voices": tidy}], tempo=120, time_sig=(4, 4))
counts: dict[int, int] = {}
for v in doc.parts[0].voices:
    counts[v.staff] = counts.get(v.staff, 0) + 1
check("plain_staff_still_folds_to_two",
      all(c <= MAX_VOICES_PER_STAFF for c in counts.values()), str(counts))

# --- a held note keeps its length instead of being cut to the next attack ----
# `_merge_two` shortens whatever is still sounding when the other line attacks,
# so folding a whole-bar C5 under running quarters would print it as a quarter.
# That is what the third voice is for; a fold that costs nothing still happens.
q = DIVISIONS
sustained = {0: {"end": 4 * q, "notes": [(72, 90)]}}
above = {i * q: {"end": i * q + q, "notes": [(79 + i, 90)]} for i in range(4)}
below = {i * q + q // 2: {"end": i * q + q, "notes": [(60 + i, 90)]} for i in range(4)}
kept = _reduce_voices([above, sustained, below], MAX_VOICES_PER_STAFF,
                      VOICE_OVERFLOW_LIMIT, VOICE_OVERFLOW_COST)
ends = [e["end"] for v in kept for e in v.values()
        if any(pt == 72 for pt, _ in e["notes"])]
check("held_note_keeps_its_length", ends == [4 * q],
      f"{len(kept)} voice(s), the whole-bar C5 printed as {ends} of {4 * q}")

# Folding to two would cut it to a single beat, which is what the charge buys:
folded = _reduce_voices([above, sustained, below], MAX_VOICES_PER_STAFF)
check("without_the_charge_it_is_cut",
      len(folded) == 2 and [e["end"] for v in folded for e in v.values()
                            if any(pt == 72 for pt, _ in e["notes"])] != [4 * q],
      "the two-voice fold no longer clips, so this test proves nothing")

# A fold that costs nothing still happens — two lines attacking together are a
# chord voiced across two lines, not two voices worth printing separately.
chord = {i * q: {"end": i * q + q, "notes": [(76 + i, 90)]} for i in range(4)}
free = _reduce_voices([chord, above], 1, VOICE_OVERFLOW_LIMIT, VOICE_OVERFLOW_COST)
check("free_fold_still_happens", len(free) == 1, f"kept {len(free)} voices")

# --- a printed bar holds a bar of ticks, to within one unprintable scrap -----
# VexFlow places notes by accumulated ticks, so a bar that stops short of its
# barline spaces differently in every renderer. A triplet grid interleaved with
# the plain one can leave a remainder with no note value at all; that scrap is
# smaller than the shortest value we print, and it is the only slack allowed.
busy = build_score([{"name": "Piano",
                     "voices": [line([72, 74, 76, 77, 79, 77, 76, 74],
                                     step=1.0 / 3, dur=0.30),
                                line([48, 50, 52, 53], step=0.5, dur=0.45)]}],
                   tempo=120, time_sig=(4, 4))
bar_ticks = DIVISIONS * 4
lengths = [sum(e.dur for e in m.events)
           for p_ in busy.parts for v in p_.voices for m in v.measures]
check("bars_hold_a_bar_of_ticks",
      all(0 <= bar_ticks - t < MIN_TICKS for t in lengths[:-1]),
      f"bar totals {lengths} against {bar_ticks}")

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

# --- a note's END is not stuck on the grid its neighbours' ATTACKS chose ------
# One attack per beat, each note stopping halfway through its beat. `_pick_subdiv`
# sees a single onset per beat and picks a one-slot grid, which used to round
# every note end out to the whole beat. Measured against the reference MIDI that
# was most of the page's offset loss (+offset F1 0.453 vs 0.563 for its input).
beats = [i * 0.5 for i in range(26)]
# 0.3 s of a 0.5 s beat: on a one-slot grid that rounds UP to the whole beat.
# (Exactly half a beat would round to zero and hit the `b <= a` fallback
# instead, which tests a different branch.)
half = [{"start": beats[i], "end": beats[i] + 0.3, "pitch": 60, "velocity": 90}
        for i in range(2, 22)]
doc = build_score([{"name": "P", "voices": [half]}], beats=beats, tempo=120,
                  time_sig=(4, 4))
sounding = [e.dur for p_ in doc.parts for v in p_.voices for m in v.measures
            for e in m.events if not e.is_rest]
beat_ticks = DIVISIONS
check("part_beat_note_is_not_printed_as_a_whole_beat",
      sounding and max(sounding) < beat_ticks,
      f"longest sounding value {max(sounding, default=0)} of {beat_ticks} ticks/beat")

# control: with ends pinned to the onset grid the same input DOES round up, so
# the fixture can actually see the regression it is guarding.
import score_build as _SB
_floor = _SB.END_SUBDIV_FLOOR
_SB.END_SUBDIV_FLOOR = 1
doc0 = build_score([{"name": "P", "voices": [half]}], beats=beats, tempo=120,
                   time_sig=(4, 4))
_SB.END_SUBDIV_FLOOR = _floor
s0 = [e.dur for p_ in doc0.parts for v in p_.voices for m in v.measures
      for e in m.events if not e.is_rest]
check("without_the_end_grid_it_rounds_up", s0 and max(s0) >= beat_ticks,
      f"longest sounding value {max(s0, default=0)}; fixture proves nothing")

# --- a rest that was really there survives ------------------------------------
# `_absorb_short_rests` holds a note to the next attack only when the gap is
# under a 16th. It used to swallow any gap shorter than the note itself, which
# after a half note meant eating a two-beat rest.
gapped = [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 90},
          {"start": 2.0, "end": 3.0, "pitch": 60, "velocity": 90}]
doc = build_score([{"name": "P", "voices": [gapped]}],
                  beats=[i * 0.5 for i in range(12)], tempo=120, time_sig=(4, 4))
first = [m for p_ in doc.parts for v in p_.voices for m in v.measures][0]
check("a_full_beat_rest_is_not_swallowed",
      any(e.is_rest for e in first.events),
      "a two-beat gap after a two-beat note printed as no rest at all")

print("\nall score-layout checks passed")
