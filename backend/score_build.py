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
  5. emits chords (real polyphony), rests, ties across barlines, and tuplet
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
    """A beat time for every beat covering [0, total], extrapolating as needed."""
    bs = sorted(float(b) for b in (beats or []) if b >= 0)
    if len(bs) >= 4:
        # median period, extended both ways so the grid always covers the piece
        per = sorted(bs[i + 1] - bs[i] for i in range(len(bs) - 1))[len(bs) // 2] or 0.5
        while bs[0] - per > 0.02:
            bs.insert(0, bs[0] - per)
        if bs[0] > 0.02:
            bs.insert(0, max(0.0, bs[0] - per))
        while bs[-1] < total + per:
            bs.append(bs[-1] + per)
        return bs
    per = 60.0 / (tempo or 120.0)
    n = int(total / per) + 3
    return [i * per for i in range(n)]


def _pick_subdiv(onsets: list[float], t0: float, t1: float):
    """Choose the subdivision that explains this beat's onsets best."""
    if not onsets:
        return 4, None
    span = max(1e-6, t1 - t0)
    best = None
    for parts, tup in _SUBDIVS:
        err = 0.0
        for o in onsets:
            x = (o - t0) / span * parts
            err += abs(x - round(x)) / parts
        err /= len(onsets)
        # prefer simpler grids; triplets must earn their place
        penalty = 0.004 * parts + (0.010 if tup else 0.0)
        sc = err + penalty
        if best is None or sc < best[0]:
            best = (sc, parts, tup)
    return best[1], best[2]


def _q(t: float, t0: float, t1: float, parts: int) -> Fraction:
    """Quantise a time inside one beat → Fraction of that beat."""
    span = max(1e-6, t1 - t0)
    return Fraction(max(0, min(parts, round((t - t0) / span * parts))), parts)


def build_score(parts_in, *, beats=None, tempo=120.0, time_sig=(4, 4),
                title="MusicNote", key_fifths=None, max_measures=400) -> ScoreDoc:
    """parts_in: [{name, notes:[{start,end,pitch,velocity}], program, is_drum,
    voices?}]. A part may carry pre-separated `voices` (list of note lists);
    otherwise it becomes one voice."""
    num, den = int(time_sig[0]), int(time_sig[1])
    num = max(1, min(32, num))
    den = den if den in (2, 4, 8, 16) else 4
    beats_per_measure = num * (4 // den) if den <= 4 else Fraction(num * 4, den)
    beats_per_measure = int(beats_per_measure) or num

    all_notes = [n for p in parts_in for n in p.get("notes", [])]
    if key_fifths is None:
        key_fifths = krumhansl_fifths([int(n["pitch"]) for n in all_notes])
    total = max((float(n["end"]) for n in all_notes), default=1.0)
    grid = _beat_grid(beats or [], tempo, total)

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
        return to_ticks, tup_at

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
        to_ticks, tup_at = make_mappers(
            subdivisions([float(n["start"]) for v in vlists for n in v]))
        pitches = [int(n["pitch"]) for v in vlists for n in v]
        med = sorted(pitches)[len(pitches) // 2] if pitches else 67
        part = Part(id=f"P{pi + 1}", name=pin.get("name") or f"Part {pi + 1}",
                    program=int(pin.get("program", 0)),
                    is_drum=bool(pin.get("is_drum", False)),
                    clef="bass" if med < 57 else "treble")

        for vi, vnotes in enumerate(vlists):
            # quantised events, merged into chords by identical onset
            evs: dict[int, dict] = {}
            for n in vnotes:
                a = to_ticks(float(n["start"]))
                b = to_ticks(float(n["end"]))
                if a is None:
                    continue
                if b is None or b <= a:
                    b = a + ticks_per_beat // 4
                e = evs.setdefault(a, {"end": b, "notes": []})
                e["end"] = max(e["end"], b)
                e["notes"].append((int(n["pitch"]), int(n.get("velocity", 90))))
            for e in evs.values():           # one notehead per pitch in a chord
                seen: dict[int, int] = {}
                for p, vel in e["notes"]:
                    seen[p] = max(seen.get(p, 0), vel)
                e["notes"] = sorted(seen.items())
            _absorb_short_rests(evs, ticks_per_beat)
            voice = Voice(number=vi + 1)
            _fill_measures(voice, evs, n_meas, ticks_per_measure, key_fifths,
                           tup_at, ticks_per_beat)
            # Preserve the source-audio interval of every printed measure. The
            # notation is quantised, but playback remains on the original
            # seconds timeline and must not assume a perfectly fixed tempo.
            for mi, meas in enumerate(voice.measures):
                b0 = min(mi * beats_per_measure, len(grid) - 1)
                b1 = min((mi + 1) * beats_per_measure, len(grid) - 1)
                meas.start, meas.end = float(grid[b0]), float(grid[b1])
            if voice.measures:
                part.voices.append(voice)

        if part.voices:
            # header info on the first measure of the first voice
            m0 = part.voices[0].measures[0]
            m0.key_fifths, m0.time_sig = key_fifths, (num, den)
            m0.tempo, m0.clef = doc.tempo, part.clef
            doc.parts.append(part)
    return doc


def _absorb_short_rests(evs: dict[int, dict], tpb: int) -> None:
    """A gap shorter than a 16th is a detection artefact, not a rest — extend the
    previous event over it so the score does not fill up with 32nd rests."""
    tiny = tpb // 4
    starts = sorted(evs)
    for i, a in enumerate(starts[:-1]):
        gap = starts[i + 1] - evs[a]["end"]
        if 0 < gap < tiny:
            evs[a]["end"] = starts[i + 1]


def _fill_measures(voice: Voice, evs: dict[int, dict], n_meas: int,
                   tpm: int, fifths: int, tup_at, tpb: int) -> None:
    """Walk the timeline emitting chords + rests, splitting at barlines with ties."""
    starts = sorted(evs)
    k = 0
    carry: dict | None = None      # note(s) sounding across the barline
    for mi in range(n_meas):
        m0, m1 = mi * tpm, (mi + 1) * tpm
        meas = Measure(number=mi + 1)
        cur = m0
        if carry:                  # finish the tied-over note first
            end = min(carry["end"], m1)
            _emit(meas, cur, end, carry["notes"], fifths, tup_at, tpb,
                  tie_start=carry["end"] > m1, tie_stop=True)
            cur = end
            carry = None if end >= carry["end"] else {**carry}
        while k < len(starts) and starts[k] < cur:
            k += 1
        while cur < m1:
            nxt = starts[k] if k < len(starts) and starts[k] < m1 else None
            if nxt is None:
                _emit(meas, cur, m1, [], fifths, tup_at, tpb)
                cur = m1
                break
            if nxt > cur:
                _emit(meas, cur, nxt, [], fifths, tup_at, tpb)
                cur = nxt
            e = evs[nxt]
            end = max(min(e["end"], m1), cur + 1)
            _emit(meas, cur, end, e["notes"], fifths, tup_at, tpb,
                  tie_start=e["end"] > m1)
            if e["end"] > m1:
                carry = {"end": e["end"], "notes": e["notes"]}
            cur = end
            k += 1
        if meas.events:
            voice.measures.append(meas)
    while voice.measures and all(c.is_rest for c in voice.measures[-1].events):
        voice.measures.pop()
    _mark_tuplets(voice)


def _emit(meas: Measure, a: int, b: int, pitches, fifths: int, tup_at, tpb: int,
          tie_start: bool = False, tie_stop: bool = False) -> None:
    """Emit [a,b) as one or more tied values. A span lying inside a single
    tuplet beat is named in that tuplet's terms (so 1/3 beat = a triplet
    eighth, not a 16th tied to a 64th)."""
    dur = b - a
    if dur <= 0:
        return
    tup = tup_at(a)
    if tup and (b - 1) // tpb != a // tpb:
        tup = None                       # spans beats: fall back to plain values
    if tup and _is_plain(dur):
        tup = None                       # a full quarter inside a triplet beat
    if dur < MIN_TICKS or (not _is_plain(dur)
                           and not (tup and _is_plain(dur * tup[0] // tup[1]))):
        dur = _snap_dur(dur, tup, max(dur, MIN_TICKS))     # keep it readable
        if dur <= 0:
            return
    comps = split_duration(dur, tup)      #   is still just a quarter
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
