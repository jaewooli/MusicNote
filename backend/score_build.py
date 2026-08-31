"""
notes (seconds) + beat grid  →  ScoreDoc

The metrical stage that was missing. Instead of snapping everything to a fixed
16th grid and keeping only the top note, this:

  1. builds a continuous beat grid (from detected beats, or from the tempo),
  2. groups beats into measures by the time signature,
  3. picks each beat's subdivision independently — binary (…/4, /8, /16) or
     ternary (triplets) — by whichever explains that beat's onsets with least
     error, so triplets and swing survive,
  4. quantises onsets/offsets to the chosen subdivision,
  5. lays the resulting lines out as *notation*: a keyboard-range part is split
     across a grand staff, and each staff keeps at most two printed voices,
  6. emits chords (real polyphony), rests, ties across barlines, and tuplet
     markup.

`build_score()` is the single entry point every consumer should use.
"""
from __future__ import annotations

from fractions import Fraction

from score_model import (DIVISIONS, Chord, Measure, Note, Part, ScoreDoc, Voice,
                         krumhansl_fifths, spell, split_duration)

# candidate subdivisions per beat: (parts_per_beat, tuplet_or_None)
_SUBDIVS = [
    (1, None), (2, None), (4, None), (8, None),          # binary
    (3, (3, 2)), (6, (6, 4)),                            # triplets
]


def _beat_grid(beats: list[float], tempo: float, total: float) -> list[float]:
    """A beat time for every beat covering [0, total], extrapolating as needed.

    Every extrapolated beat is a FULL period. Clamping the leading one to t=0
    used to manufacture a short first beat: on a 133 bpm clip it was 0.406 s
    against a real 0.448 s, so the notes inside it read as 0.39 and 0.91 of a
    beat, `_pick_subdiv` reached for a 32nd grid to explain them, and a run of
    plain eighths came out as a dotted-16th rest followed by off-beat eighths
    that no longer beamed. The grid may therefore start slightly before zero —
    an anacrusis, which is what that music actually is.
    """
    bs = sorted(float(b) for b in (beats or []) if b >= 0)
    if len(bs) >= 4:
        # median period, extended both ways so the grid always covers the piece
        per = sorted(bs[i + 1] - bs[i] for i in range(len(bs) - 1))[len(bs) // 2] or 0.5
        per = max(0.05, per)          # a degenerate period must not loop forever
        while bs[0] > 1e-6:
            bs.insert(0, bs[0] - per)
        while bs[-1] < total + per:
            bs.append(bs[-1] + per)
        return bs
    per = 60.0 / (tempo or 120.0)
    n = int(total / per) + 3
    return [i * per for i in range(n)]


# Attacks closer together than this are one chord, not two rhythmic events.
CHORD_GAP = 0.035
# Cost of a subdivision that lands two separate attacks on the same slot. It has
# to outweigh any accuracy a coarse grid can win, because the consequence is not
# an inaccurate rhythm but a *deleted note*: the two collapse into one notehead.
COLLISION_COST = 0.09


def _pick_subdiv(onsets: list[float], t0: float, t1: float):
    """Choose the subdivision that explains this beat's onsets best."""
    if not onsets:
        return 4, None
    span = max(1e-6, t1 - t0)
    # Chord tones share one attack; only genuinely separate attacks can collide.
    distinct: list[float] = []
    for o in sorted(onsets):
        if not distinct or o - distinct[-1] > CHORD_GAP:
            distinct.append(o)
    best = None
    for parts, tup in _SUBDIVS:
        err = 0.0
        slots = set()
        for o in distinct:
            x = (o - t0) / span * parts
            err += abs(x - round(x)) / parts
            slots.add(round(x))
        err /= len(distinct)
        # Prefer simpler grids; triplets must earn their place. The penalty
        # grows faster than the grid does, because a 32nd grid will always fit
        # noisy onsets slightly better than a 16th one while reading far worse.
        penalty = 0.004 * parts * (1 + parts / 8) + (0.010 if tup else 0.0)
        penalty += COLLISION_COST * (len(distinct) - len(slots))
        sc = err + penalty
        if best is None or sc < best[0]:
            best = (sc, parts, tup)
    return best[1], best[2]


def _q(t: float, t0: float, t1: float, parts: int) -> Fraction:
    """Quantise a time inside one beat → Fraction of that beat."""
    span = max(1e-6, t1 - t0)
    return Fraction(max(0, min(parts, round((t - t0) / span * parts))), parts)


def _phase_bars(grid: list[float], downbeats, beats_per_measure: int) -> list[float]:
    """Shift the bar phase so a detected downbeat lands on a barline.

    `meter.detect` locates downbeats from accent evidence (how many notes attack
    there, how low the bass is, how long the longest note is) and reports how
    confident it is. The score used to ignore that and simply count bars from
    wherever the extrapolated beat grid began — which put every barline in the
    wrong place on 5 of 7 measured clips, one of them 3 beats out. Beats are
    prepended, never dropped, so whatever comes before the first downbeat
    becomes an anacrusis instead of disappearing.
    """
    if not downbeats or beats_per_measure < 2 or len(grid) < 2:
        return grid
    per = grid[1] - grid[0]
    if per <= 0:
        return grid
    votes: dict[int, int] = {}
    for d in downbeats:
        k = round((float(d) - grid[0]) / per)
        votes[k % beats_per_measure] = votes.get(k % beats_per_measure, 0) + 1
    if not votes:
        return grid
    off = max(votes, key=lambda k: (votes[k], -k))
    pad = (beats_per_measure - off) % beats_per_measure
    return [grid[0] - (pad - i) * per for i in range(pad)] + grid


def build_score(parts_in, *, beats=None, tempo=120.0, time_sig=(4, 4),
                title="MusicNote", key_fifths=None, max_measures=400,
                downbeats=None) -> ScoreDoc:
    """parts_in: [{name, notes:[{start,end,pitch,velocity}], program, is_drum,
    voices?}]. A part may carry pre-separated `voices` (list of note lists);
    otherwise it becomes one voice."""
    num, den = int(time_sig[0]), int(time_sig[1])
    num = max(1, min(32, num))
    den = den if den in (2, 4, 8, 16) else 4
    beats_per_measure = num * (4 // den) if den <= 4 else Fraction(num * 4, den)
    beats_per_measure = int(beats_per_measure) or num

    # A part may arrive as a flat note list or as pre-separated `voices`; both
    # have to reach the key estimate and the total duration, or a score built
    # from voices gets key C and a beat grid one second long.
    all_notes = [n for p in parts_in
                 for n in (list(p.get("notes") or [])
                           + [x for v in (p.get("voices") or []) for x in v])]
    if key_fifths is None:
        key_fifths = krumhansl_fifths([int(n["pitch"]) for n in all_notes])
    total = max((float(n["end"]) for n in all_notes), default=1.0)
    grid = _phase_bars(_beat_grid(beats or [], tempo, total), downbeats,
                       beats_per_measure)
    # Phasing prepends up to a bar of beats. Drop any WHOLE bar that ends before
    # the first note — the phase is a multiple of the bar, so this keeps the
    # barlines where they were and just stops the score opening on an empty bar.
    first = min((float(n["start"]) for n in all_notes), default=0.0)
    while (len(grid) > beats_per_measure + 1
           and grid[beats_per_measure] <= first + 1e-9):
        grid = grid[beats_per_measure:]

    def subdivisions(onsets: list[float]):
        """Per-beat subdivision for one part — a piano playing straight while the
        bass plays triplets must not inherit the bass's tuplets."""
        oo = sorted(onsets)
        res: list[tuple[int, tuple[int, int] | None]] = []
        i0 = 0
        for b in range(len(grid) - 1):
            t0, t1 = grid[b], grid[b + 1]
            while i0 < len(oo) and oo[i0] < t0:
                i0 += 1
            k, loc = i0, []
            while k < len(oo) and oo[k] < t1:
                loc.append(oo[k])
                k += 1
            res.append(_pick_subdiv(loc, t0, t1))
        return res

    n_beats = min(len(grid) - 1, beats_per_measure * max_measures)
    n_meas = max(1, -(-n_beats // beats_per_measure))
    ticks_per_beat = DIVISIONS * 4 // den if den <= 4 else DIVISIONS * 4 // den
    ticks_per_measure = ticks_per_beat * beats_per_measure

    def make_mappers(sub):
        def to_ticks(t: float) -> int | None:
            lo, hi = 0, len(grid) - 2
            if t < grid[0]:
                return 0
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if grid[mid] <= t:
                    lo = mid
                else:
                    hi = mid - 1
            if lo >= len(sub):
                return None
            frac = _q(t, grid[lo], grid[lo + 1], sub[lo][0])
            return lo * ticks_per_beat + int(frac * ticks_per_beat)

        def tup_at(tick: int):
            b = tick // ticks_per_beat
            return sub[b][1] if 0 <= b < len(sub) else None

        def step_at(tick: int) -> int:
            """One grid position at this point in the piece, in ticks."""
            b = tick // ticks_per_beat
            parts = sub[b][0] if 0 <= b < len(sub) else 4
            return max(1, ticks_per_beat // max(1, parts))
        return to_ticks, tup_at, step_at

    doc = ScoreDoc(title=title, parts=[], key_fifths=key_fifths,
                   time_sig=(num, den), tempo=round(float(tempo or 120.0), 1))

    for pi, pin in enumerate(parts_in):
        vlists = pin.get("voices")
        if not vlists:
            source_notes = pin.get("notes", [])
            # Never feed overlapping intervals into the single-voice event
            # walker: it would have to skip later starts.  Keep every detected
            # pitch in independent notation voices instead.
            if source_notes:
                from voices import poly_fraction, separate_voices
                vlists = (separate_voices(source_notes, max_voices=None)
                          if poly_fraction(source_notes) > 0.0
                          else [source_notes])
            else:
                vlists = []
        vlists = [v for v in vlists if v]
        if not vlists:
            continue
        to_ticks, tup_at, step_at = make_mappers(
            subdivisions([float(n["start"]) for v in vlists for n in v]))
        part = Part(id=f"P{pi + 1}", name=pin.get("name") or f"Part {pi + 1}",
                    program=int(pin.get("program", 0)),
                    is_drum=bool(pin.get("is_drum", False)))

        # 1. every source line becomes a quantised event map, high register first
        lines = []
        for vnotes in vlists:
            evs = _quantise(vnotes, to_ticks, ticks_per_beat, step_at)
            if evs:
                lines.append(evs)
        if not lines:
            continue
        lines.sort(key=lambda e: -_median_pitch(e))

        # 2. spread them over one or two staves, then hold each staff to two
        #    printed voices — three or more on a staff is unreadable, and every
        #    extra one adds a full layer of rests.
        staff_of, part.clefs = _plan_staves([_pitches(e) for e in lines],
                                            part.is_drum)
        part.staves = len(part.clefs)
        part.clef = part.clefs[0]
        vnum = 0
        for st in range(1, part.staves + 1):
            group = [e for e, s in zip(lines, staff_of) if s == st]
            for evs in _reduce_voices(group, MAX_VOICES_PER_STAFF):
                _absorb_short_rests(evs, ticks_per_beat)
                vnum += 1
                voice = Voice(number=vnum, staff=st)
                _fill_measures(voice, evs, n_meas, ticks_per_measure, key_fifths,
                               tup_at, ticks_per_beat)
                # Preserve the source-audio interval of every printed measure.
                # The notation is quantised, but playback remains on the original
                # seconds timeline and must not assume a fixed tempo.
                for meas in voice.measures:
                    mi = meas.number - 1
                    b0 = min(mi * beats_per_measure, len(grid) - 1)
                    b1 = min((mi + 1) * beats_per_measure, len(grid) - 1)
                    meas.start, meas.end = float(grid[b0]), float(grid[b1])
                if voice.measures:
                    part.voices.append(voice)

        if part.voices:
            doc.parts.append(part)

    _trim_leading_empty(doc)
    for part in doc.parts:
        # header info on the first measure of the first voice
        if part.voices and part.voices[0].measures:
            m0 = part.voices[0].measures[0]
            m0.key_fifths, m0.time_sig = key_fifths, (num, den)
            m0.tempo, m0.clef = doc.tempo, part.clef
    return doc


def _trim_leading_empty(doc: ScoreDoc) -> None:
    """Drop the bars before the first note.

    Phasing the barlines onto the detected downbeat prepends up to a bar of
    beats, and quantisation then often pushes the opening attack onto that
    downbeat — leaving the score to open on a bar of rests that stands for no
    audio at all. Whole bars are removed from every voice together, so the
    voices stay aligned and the barlines keep their phase.
    """
    first = None
    for p in doc.parts:
        for v in p.voices:
            for m in v.measures:
                if any(not e.is_rest for e in m.events):
                    first = m.number if first is None else min(first, m.number)
                    break
    if not first or first < 2:
        return
    drop = first - 1
    for p in doc.parts:
        for v in p.voices:
            v.measures = [m for m in v.measures if m.number > drop]
            for m in v.measures:
                m.number -= drop


# --------------------------------------------------------------------------- #
# notation layout: staves and printed voices
# --------------------------------------------------------------------------- #
MAX_VOICES_PER_STAFF = 2


def _quantise(vnotes, to_ticks, tpb: int, step_at=None) -> dict[int, dict]:
    """One source line -> {onset_tick: {end, notes:[(pitch, velocity)]}}.

    Notes landing on the same tick are one chord, one notehead per pitch. The
    exception is a *repeated* pitch: two attacks of the same note rounding onto
    one tick used to be deduplicated away, which is a deleted note rather than
    an inaccurate rhythm. Measured on eval/refs that silently lost 56 notes.
    `_pick_subdiv` now charges for such a collision and usually avoids it; when
    the grid still cannot separate them the re-attack is moved to the next grid
    position instead of being dropped.
    """
    evs: dict[int, dict] = {}
    for n in sorted(vnotes, key=lambda x: (float(x["start"]), int(x["pitch"]))):
        a = to_ticks(float(n["start"]))
        b = to_ticks(float(n["end"]))
        if a is None:
            continue
        pitch = int(n["pitch"])
        step = max(1, (step_at(a) if step_at else tpb // 4))
        guard = 0
        while (a in evs and any(p == pitch for p, _ in evs[a]["notes"])
               and guard < 4):
            a += step
            guard += 1
        if b is None or b <= a:
            b = a + tpb // 4
        e = evs.setdefault(a, {"end": b, "notes": []})
        e["end"] = max(e["end"], b)
        e["notes"].append((pitch, int(n.get("velocity", 90))))
    for e in evs.values():
        e["notes"] = _uniq_notes(e["notes"])
    return evs


def _uniq_notes(pairs) -> list[tuple[int, int]]:
    seen: dict[int, int] = {}
    for p, vel in pairs:
        seen[p] = max(seen.get(p, 0), vel)
    return sorted(seen.items())


def _pitches(evs: dict[int, dict]) -> list[int]:
    return [p for e in evs.values() for p, _ in e["notes"]]


def _median_pitch(evs: dict[int, dict]) -> float:
    ps = sorted(p for e in evs.values() for p, _ in e["notes"])
    return ps[len(ps) // 2] if ps else 67.0


# Middle line of each staff, in MIDI, and how far from it a note may sit before
# it starts costing ledger lines a reader has to count.
_CLEF_CENTRE = {"treble": 71, "bass": 50}
LEDGER_FREE = 11
# Splitting onto two staves has to be worth it; this is the toll, in the same
# units as the ledger cost (average semitones outside the staff, per note).
SPLIT_BIAS = 0.20
# Weight of putting more voices on a staff than it can print clearly.
CROWD_WEIGHT = 1.0


def _ledger_cost(pitches, clef: str) -> float:
    c = _CLEF_CENTRE[clef]
    return sum(max(0.0, abs(p - c) - LEDGER_FREE) for p in pitches)


def _plan_staves(line_pitches: list[list[int]], is_drum: bool):
    """Spread a part's lines over one or two staves -> (staff per line, clefs).

    Chosen by how far the notes end up from the staff they are printed on, not
    by balancing voice counts: an inner line whose median sits near middle C can
    still reach four ledger lines below a treble staff, and counting ledger
    lines is most of what makes a transcription unreadable. Every register
    boundary is tried, each staff gets whichever clef writes its own notes with
    the fewest ledger lines, and one staff wins unless splitting actually pays.
    """
    n = len(line_pitches)
    if is_drum or n == 0:
        return [1] * n, ["treble"]
    total = sum(len(p) for p in line_pitches) or 1

    def staff(group) -> tuple[str, float]:
        ps = [p for line in group for p in line]
        if not ps:
            return "treble", 0.0
        clef = min(("treble", "bass"), key=lambda c: _ledger_cost(ps, c))
        return clef, _ledger_cost(ps, clef)

    def crowd(k: int) -> float:
        return CROWD_WEIGHT * max(0, k - MAX_VOICES_PER_STAFF) ** 2

    clef, cost = staff(line_pitches)
    best = (cost / total + crowd(n), [1] * n, [clef])
    for k in range(1, n):
        ct, top = staff(line_pitches[:k])
        cb, bot = staff(line_pitches[k:])
        sc = (top + bot) / total + crowd(k) + crowd(n - k) + SPLIT_BIAS
        if sc < best[0]:
            best = (sc, [1] * k + [2] * (n - k), [ct, cb])
    return best[1], best[2]


def _merge_two(a: dict[int, dict], b: dict[int, dict]) -> dict[int, dict]:
    """Print two lines as one voice. Events attacking on the same tick become a
    chord; where they do not, the sounding one is cut short at the next attack,
    because a printed voice cannot overlap itself. No pitch is dropped."""
    out = {t: {"end": e["end"], "notes": list(e["notes"])} for t, e in a.items()}
    for t, e in b.items():
        if t in out:
            out[t]["end"] = max(out[t]["end"], e["end"])
            out[t]["notes"] = _uniq_notes(out[t]["notes"] + e["notes"])
        else:
            out[t] = {"end": e["end"], "notes": list(e["notes"])}
    ts = sorted(out)
    for i in range(len(ts) - 1):
        if out[ts[i]]["end"] > ts[i + 1]:
            out[ts[i]]["end"] = ts[i + 1]
    return out


def _merge_cost(a: dict[int, dict], b: dict[int, dict]) -> float:
    """What merging these two lines would cost. Lines that attack together (a
    chord voiced across two lines) cost nothing to combine.

    Two prices are charged. Shortening a note that is still sounding when the
    other line attacks is the cheap one — the pitch survives, only its printed
    length changes. A *unison* is the expensive one: both lines hold the same
    pitch on the same tick, so the two collapse into a single notehead and one
    of them stops being visible. That is chosen last, never by accident.
    """
    ts = sorted(set(a) | set(b))
    if not ts:
        return 0.0
    clipped = unison = 0
    for i, t in enumerate(ts):
        if t in a and t in b:
            pa = {p for p, _ in a[t]["notes"]}
            unison += len(pa & {p for p, _ in b[t]["notes"]})
        if i + 1 < len(ts):
            end = max(m[t]["end"] for m in (a, b) if t in m)
            if end > ts[i + 1]:
                clipped += 1
    return (clipped + 4.0 * unison) / len(ts)


def _reduce_voices(lines: list[dict], limit: int) -> list[dict]:
    """Fold a staff's lines down to `limit` printed voices, always merging the
    neighbouring pair (in register) that loses the least."""
    lines = list(lines)
    while len(lines) > limit:
        i = min(range(len(lines) - 1),
                key=lambda k: (_merge_cost(lines[k], lines[k + 1]), k))
        lines[i:i + 2] = [_merge_two(lines[i], lines[i + 1])]
    return lines


def _absorb_short_rests(evs: dict[int, dict], tpb: int) -> None:
    """A gap the model did not reliably measure is not a rest.

    MT3's onsets are excellent and its note lengths are not. Measured on
    eval/refs over 2977 matched notes, the estimated/reference duration ratio
    has median 0.90 with p10 0.42 and p90 1.23, and 17.4% of notes are off by
    more than half. Printed literally, one repeated figure comes out as a jumble
    of different note values scattered with rests that were never played —
    exactly the "same note, different length every time" complaint.

    So the printed rhythm is built from the onsets, which are trustworthy: a gap
    becomes a rest only when it is at least as long as the note it follows (and
    at least a 16th). Anything shorter is release noise, and the note is held to
    the next attack. A gap of a beat or more is always kept, however long the
    preceding note was.
    """
    tiny = tpb // 4
    starts = sorted(evs)
    for i, a in enumerate(starts[:-1]):
        nxt = starts[i + 1]
        gap = nxt - evs[a]["end"]
        if gap <= 0:
            continue
        sounded = evs[a]["end"] - a
        if gap < max(tiny, min(sounded, tpb)):
            evs[a]["end"] = nxt


def _fill_measures(voice: Voice, evs: dict[int, dict], n_meas: int,
                   tpm: int, fifths: int, tup_at, tpb: int) -> None:
    """Walk the timeline emitting chords + rests, splitting at barlines with ties.

    `cur` advances by what was actually PRINTED, not by the span that was asked
    for. The two differ whenever a span has no single readable note value, and
    following the requested span instead left half of all measures not adding up
    to a full measure — VexFlow places notes by accumulated ticks, so every note
    after the discrepancy sat off its beat, and the bar either overflowed its
    barline or stopped short of it. Advancing by the printed value keeps the
    printed bar exact; a note whose value had to be rounded simply gets a short
    rest after it, which is where the missing time really is.
    """
    starts = sorted(evs)
    k = 0
    carry: dict | None = None      # note(s) sounding across the barline
    for mi in range(n_meas):
        m0, m1 = mi * tpm, (mi + 1) * tpm
        meas = Measure(number=mi + 1)
        cur = m0
        if carry:                  # finish the tied-over note first
            end = min(carry["end"], m1)
            cur += _emit(meas, cur, end, carry["notes"], fifths, tup_at, tpb,
                         tie_start=carry["end"] > m1, tie_stop=True,
                         budget=m1 - cur)
            carry = None if end >= carry["end"] else {**carry}
        while k < len(starts) and starts[k] < cur:
            k += 1
        while cur < m1:
            nxt = starts[k] if k < len(starts) and starts[k] < m1 else None
            if nxt is None:
                _emit(meas, cur, m1, [], fifths, tup_at, tpb, budget=m1 - cur)
                cur = m1
                break
            if nxt > cur:
                cur += _emit(meas, cur, nxt, [], fifths, tup_at, tpb,
                             budget=m1 - cur)
                if cur >= m1:
                    break
            e = evs[nxt]
            end = min(max(e["end"], cur + MIN_TICKS), m1)
            printed = _emit(meas, cur, end, e["notes"], fifths, tup_at, tpb,
                            tie_start=e["end"] > m1, budget=m1 - cur)
            if not printed:
                break
            if e["end"] > m1:
                carry = {"end": e["end"], "notes": e["notes"]}
            cur += printed
            k += 1
        if meas.events:
            voice.measures.append(meas)
    while voice.measures and all(c.is_rest for c in voice.measures[-1].events):
        voice.measures.pop()
    _mark_tuplets(voice)


def _components(dur: int, tup, budget: int):
    """(list of (type, dots), ticks printed) for a span.

    Exact first: `split_duration` writes any span as tied values, and an exact
    bar is what keeps the notes on their beats. It falls back to the nearest
    single readable value when the span cannot be written exactly (a binary span
    inside a triplet beat, say) or when being exact would cost a chain of more
    than three tied noteheads, which reads worse than a slightly wrong length.
    """
    dur = min(dur, budget)
    if dur < MIN_TICKS:
        return [], 0
    comps = split_duration(dur, tup)
    got = sum(_ticks_of(ty, dots) for ty, dots in comps)
    if tup:
        got = got * tup[1] // tup[0]
    if got == dur and len(comps) <= MAX_TIED_COMPONENTS:
        return comps, got
    snapped = _snap_dur(dur, tup, dur)
    if snapped < MIN_TICKS or snapped > budget:
        return [], 0
    comps = split_duration(snapped, tup)
    got = sum(_ticks_of(ty, dots) for ty, dots in comps)
    if tup:
        got = got * tup[1] // tup[0]
    return comps, got


def _emit(meas: Measure, a: int, b: int, pitches, fifths: int, tup_at, tpb: int,
          tie_start: bool = False, tie_stop: bool = False,
          budget: int | None = None) -> int:
    """Emit [a,b) as one or more tied values, and report the ticks printed.

    A span lying inside a single tuplet beat is named in that tuplet's terms (so
    1/3 beat = a triplet eighth, not a 16th tied to a 64th)."""
    dur = b - a
    if budget is not None:
        dur = min(dur, budget)
    if dur <= 0:
        return 0
    tup = tup_at(a)
    if tup and (b - 1) // tpb != a // tpb:
        tup = None                       # spans beats: fall back to plain values
    if tup and _is_plain(dur):
        tup = None                       # a full quarter inside a triplet beat
    comps, printed = _components(dur, tup, dur)
    if not comps:
        return 0
    for ci, (ty, dots) in enumerate(comps):
        pdur = _ticks_of(ty, dots)
        if tup:
            pdur = pdur * tup[1] // tup[0]
        notes = []
        for p, vel in pitches:
            st, al, oc = spell(int(p), fifths)
            notes.append(Note(step=st, alter=al, octave=oc, midi=int(p),
                              velocity=int(vel),
                              tie_start=(ci < len(comps) - 1) or tie_start,
                              tie_stop=(ci > 0) or (tie_stop and ci == 0)))
        meas.events.append(Chord(notes=notes, dur=pdur, type=ty, dots=dots,
                                 tuplet=tup))
    return printed


def _mark_tuplets(voice: Voice) -> None:
    """Bracket consecutive same-tuplet events (start on the first, stop on the last)."""
    for m in voice.measures:
        i = 0
        while i < len(m.events):
            if not m.events[i].tuplet:
                i += 1
                continue
            j = i
            while j + 1 < len(m.events) and m.events[j + 1].tuplet == m.events[i].tuplet:
                j += 1
            m.events[i].tuplet_start = True
            m.events[j].tuplet_stop = True
            i = j + 1


def _is_plain(ticks: int) -> bool:
    """Exactly one un-tupleted note value (with dots)?"""
    from score_model import _plain
    return _plain(Fraction(ticks, DIVISIONS)) is not None


MIN_TICKS = DIVISIONS // 8      # a 32nd — nothing shorter belongs in a transcript
# Beyond this a tied chain reads worse than a slightly rounded note value.
MAX_TIED_COMPONENTS = 3


def _writable_ticks(tup, limit: int) -> list[int]:
    """Durations a reader can actually parse: plain values (≤1 dot) and, in a
    tuplet beat, that tuplet's values. Ends are noisy (measured note+offset F1
    is very low), so a readable near value beats an exact unreadable one."""
    from score_model import MAX_DOTS, _TYPES
    out = set()
    for base, _ in _TYPES:
        for dots in range(MAX_DOTS + 1):
            v = base * (Fraction(2) - Fraction(1, 2 ** dots))
            for t in (int(v * DIVISIONS),
                      int(v * DIVISIONS * tup[1] / tup[0]) if tup else 0):
                if MIN_TICKS <= t <= limit:
                    out.add(t)
    return sorted(out)


def _snap_dur(dur: int, tup, limit: int) -> int:
    """Nearest writable duration; never below a 32nd."""
    cands = _writable_ticks(tup, max(limit, MIN_TICKS))
    if not cands:
        return max(dur, MIN_TICKS)
    return min(cands, key=lambda c: (abs(c - dur), c))


def _ticks_of(ty: str, dots: int) -> int:
    from score_model import _TYPES
    for base, name in _TYPES:
        if name == ty:
            mult = Fraction(2) - Fraction(1, 2 ** dots)
            return int(base * mult * DIVISIONS)
    return DIVISIONS
