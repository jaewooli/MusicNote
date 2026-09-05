"""Strict, continuity-oriented voice allocation.

This module deliberately does *not* claim to identify performers. It turns a
polyphonic note stream into readable, non-overlapping lines for notation.
"""
from __future__ import annotations

from statistics import median

# How much of the shorter line's attacks must land on the other line's before
# the pair counts as rhythmically chordal. 0.72 demanded near-perfect agreement,
# which no arpeggio or sustained harmony ever reaches.
ATTACK_RATIO = 0.62
# Two lines with no attack agreement are still one chord if they sound together
# for this share of the shorter one's sounding time.
SUSTAIN_FRACTION = 0.55
# ... and hold a roughly fixed interval while doing so (semitones, weighted SD).
# Above this they are moving independently, which is counterpoint, not a chord.
SUSTAIN_SPREAD = 3.5
# Beyond an octave apart, two lines are separate registers (bass vs melody)
# however well their rhythm agrees.
MAX_CHORD_SPAN = 12


def poly_fraction(notes: list[dict]) -> float:
    """Fraction of notes that overlap in time with at least one other note."""
    if len(notes) < 2:
        return 0.0
    ns = sorted(notes, key=lambda x: x["start"])
    return sum(any(other["start"] < note["end"] - 1e-3 for other in ns[i + 1:])
               for i, note in enumerate(ns)) / len(ns)


def _new_voice() -> dict:
    return {"last_pitch": None, "free_at": 0.0, "notes": [], "pitches": []}


def _continuity_cost(voice: dict, note: dict) -> float:
    """Cost of continuing ``voice`` with ``note``.

    The recent pitch is the strongest signal, while the median pitch keeps two
    interleaving registers from swapping identity every few notes. A long
    silence weakens continuity without making a new voice mandatory.
    """
    last = voice["last_pitch"]
    if last is None:
        return 0.0
    gap = max(0.0, float(note["start"]) - float(voice["free_at"]))
    tessitura = median(voice["pitches"][-12:]) if voice["pitches"] else last
    return (abs(note["pitch"] - last)
            + abs(note["pitch"] - tessitura) * 0.35
            + min(gap, 2.0) * 3.0)


def _finish_at(voice: dict, when: float, min_dur: float) -> None:
    """End the last event before ``when`` instead of allowing an overlap.

    AMT offsets are less reliable than onsets, so a clean re-attack is better
    than two simultaneous notes in one notated line.
    """
    if not voice["notes"]:
        return
    prev = voice["notes"][-1]
    if prev["end"] <= when:
        return
    if when - prev["start"] >= min_dur:
        prev["end"] = when
    else:
        voice["notes"].pop()
    voice["free_at"] = when



def _leap_cost(semitones: int) -> float:
    """Melodic distance between two consecutive notes of one line.

    Plain ``abs`` charges an octave leap 12.0, which equals the cost of opening
    a brand new contour, so any octave jump silently started a second voice —
    the Canon melody was cut at exactly its two octave leaps. An octave keeps
    the pitch class, so it is musically a near neighbour, not a new performer;
    displaced intervals are charged their distance from the octave plus a fixed
    penalty for the displacement itself.
    """
    d = abs(int(semitones))
    return min(float(d), abs(d - 12) + 6.0)


def _onset_groups(notes: list[dict], chord_gap: float) -> list[list[dict]]:
    """Cluster nearly simultaneous attacks into one vertical event."""
    groups: list[list[dict]] = []
    for note in notes:
        if not groups or float(note["start"]) - float(groups[-1][0]["start"]) > chord_gap:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def _provisional_contours(notes: list[dict], chord_gap: float,
                          continuity_gap: float) -> list[list[dict]]:
    """Join note attacks into contours while tolerating uncertain MT3 offsets.

    MT3 often leaves a prior note 40--150 ms too long. Requiring its literal
    offset to finish before the next onset fractures one melody into alternating
    tracks. A small overlap is allowed when pitch/register continuity is
    convincing; all notes at one onset are assigned *jointly* so chord voices
    cannot cross merely because the nearest note was claimed first.
    """
    tracks: list[dict] = []
    for group in _onset_groups(notes, chord_gap):
        group = sorted(group, key=lambda n: n["pitch"])
        options: list[list[tuple[float, int | None]]] = []
        for note in group:
            start, pitch = float(note["start"]), int(note["pitch"])
            choices = [(12.0, None)]  # cost of opening an independent contour
            for ti, track in enumerate(tracks):
                gap = start - track["last_start"]
                overlap = track["last_end"] - start
                if gap <= chord_gap or gap > continuity_gap:
                    continue
                # Overlap is a preference, never a veto. MT3's offsets are far
                # less reliable than the old 0.18 s allowance assumed: on the
                # Canon clip it held melody notes 0.32-1.32 s past the next
                # attack. Vetoing on that locked a line out of its own
                # continuation, so one monophonic melody ping-ponged between
                # two contours (a1 -> a2 -> a1 -> a2) — exactly the fracture
                # this function exists to prevent.
                cost = (_leap_cost(pitch - track["last_pitch"])
                        + min(max(0.0, overlap), 0.5) * 2.0
                        + min(gap, 2.0) * 0.7)
                if cost <= 12.0:
                    choices.append((cost, ti))
            options.append(choices)

        # A chord normally has only 2--6 notes. Exhaustive one-to-one matching
        # is tiny here and avoids a local nearest-pitch decision swapping lines.
        best: tuple[float, list[int | None]] | None = None
        def assign(i: int, used: set[int], cost: float, out: list[int | None]) -> None:
            nonlocal best
            if best is not None and cost >= best[0]:
                return
            if i == len(group):
                best = (cost, out[:])
                return
            for item_cost, ti in options[i]:
                if ti is not None and ti in used:
                    continue
                if ti is not None:
                    used.add(ti)
                out.append(ti)
                assign(i + 1, used, cost + item_cost, out)
                out.pop()
                if ti is not None:
                    used.remove(ti)
        assign(0, set(), 0.0, [])
        assert best is not None
        for note, ti in zip(group, best[1]):
            if ti is None:
                tracks.append({"notes": []})
                ti = len(tracks) - 1
            track = tracks[ti]
            # A notated line cannot overlap itself. When a line continues, the
            # previous note ends where the new one starts; this also repairs
            # MT3's over-held offsets instead of carrying them into the score.
            if track["notes"]:
                prev = track["notes"][-1]
                if float(prev["end"]) > float(note["start"]):
                    prev["end"] = float(note["start"])
            track["notes"].append(note)
            track["last_pitch"] = int(note["pitch"])
            track["last_start"] = float(note["start"])
            track["last_end"] = float(note["end"])
    return [t["notes"] for t in tracks if t["notes"]]


def _aligned_pairs(a: list[dict], b: list[dict], tolerance: float) -> list[tuple[dict, dict]]:
    """One-to-one onset alignment of two provisional melodic lines."""
    pairs, j = [], 0
    for left in a:
        while j < len(b) and float(b[j]["start"]) < float(left["start"]) - tolerance:
            j += 1
        candidates = [k for k in (j - 1, j) if 0 <= k < len(b)
                      and abs(float(b[k]["start"]) - float(left["start"])) <= tolerance]
        if candidates:
            k = min(candidates, key=lambda x: abs(float(b[x]["start"]) - float(left["start"])))
            pairs.append((left, b[k]))
            j = k + 1
    return pairs


def _sustain_overlap(a: list[dict], b: list[dict]) -> tuple[float, float, float]:
    """How much two lines simply *ring together* -> (fraction, mean, spread).

    The onset test below can only see lines that attack together. A held chord,
    or a broken one whose tones enter one after another and then sustain, never
    passes it — which is why a sustained harmony kept being advertised as one
    new sequence per chord tone. Sounding together for most of their length is
    the other kind of evidence that two lines are one chord.

    `fraction` is shared sounding time over the shorter line's own sounding
    time; `mean`/`spread` describe the vertical interval, weighted by how long
    each pairing actually sounds.
    """
    dur_a = sum(float(n["end"]) - float(n["start"]) for n in a)
    dur_b = sum(float(n["end"]) - float(n["start"]) for n in b)
    if dur_a <= 0 or dur_b <= 0:
        return 0.0, 0.0, 0.0
    shared, wsum, samples = 0.0, 0.0, []
    j = 0
    for x in a:
        xs, xe = float(x["start"]), float(x["end"])
        while j < len(b) and float(b[j]["end"]) <= xs:
            j += 1
        k = j
        while k < len(b) and float(b[k]["start"]) < xe:
            ov = min(xe, float(b[k]["end"])) - max(xs, float(b[k]["start"]))
            if ov > 0:
                shared += ov
                wsum += ov
                samples.append((ov, int(b[k]["pitch"]) - int(x["pitch"])))
            k += 1
    if not samples or wsum <= 0:
        return 0.0, 0.0, 0.0
    mean = sum(w * d for w, d in samples) / wsum
    spread = (sum(w * (d - mean) ** 2 for w, d in samples) / wsum) ** 0.5
    return shared / min(dur_a, dur_b), mean, spread


def _chord_verdict(a: list[dict], b: list[dict],
                   tolerance: float) -> tuple[float | None, bool]:
    """Judge whether two lines behave as one chordal pattern.

    Returns ``(score, conflict)``. These are three outcomes, not two:

    * ``(score, False)``  the pair is chordal; lower score is a better merge.
    * ``(None, True)``    enough shared attacks to judge, and they failed on
      register or interval — the pair must never share a sequence.
    * ``(None, False)``   too few shared attacks to have an opinion. This is an
      abstention, not a veto: two contours covering different stretches of the
      clip legitimately have little overlap to compare.

    Collapsing the last two into one "no" is what makes a one-off colour tone
    look like a rejected line.

    This is deliberately a *whole-sequence* test.  A vertical sonority alone
    never passes: the lines need repeated shared attacks and a stable interval.
    """
    pairs = _aligned_pairs(a, b, tolerance)
    ratio = len(pairs) / min(len(a), len(b))
    if len(pairs) >= 2 and ratio >= ATTACK_RATIO:
        intervals = [int(right["pitch"]) - int(left["pitch"]) for left, right in pairs]
        centre = sum(intervals) / len(intervals)
        spread = sum((x - centre) ** 2 for x in intervals) / len(intervals)
        # Notes more than an octave apart are normally separate registers (bass
        # / melody), even when their rhythm happens to coincide.
        if abs(centre) > MAX_CHORD_SPAN or spread > 1.25 ** 2:
            return None, True
        # Prefer the most stable interval and the most complete alignment.
        return spread + (1.0 - ratio) * 2.0, False

    # No usable attack evidence. Two lines that sound together for most of their
    # length are still one chord — a pad, or a broken chord left ringing.
    frac, mean, spread = _sustain_overlap(a, b)
    if frac < SUSTAIN_FRACTION:
        return None, False
    if abs(mean) > MAX_CHORD_SPAN:
        return None, True
    if spread > SUSTAIN_SPREAD:
        return None, False        # they move against each other: independent
    # Ranked below any attack-aligned merge, so real chords still win first.
    return 1.5 + spread * 0.5 + (1.0 - frac) * 2.0, False


def _parallel_chord_score(a: list[dict], b: list[dict], tolerance: float) -> float | None:
    """Merge score for two lines, or None when they are not one chordal pattern."""
    return _chord_verdict(a, b, tolerance)[0]


# Two groups more than this far apart in register are different lines even when
# their times do not clash — stitching a bass figure onto a melody that happens
# to have paused reads worse than leaving them apart.
LINK_SPAN = 18
# A group may start this soon after the previous one ends and still be its
# continuation; AMT offsets routinely run a little long.
LINK_GAP = 0.12


def _intervals(group: list[dict]) -> list[tuple[float, float]]:
    """The times a group actually occupies, merged into disjoint spans."""
    out: list[list[float]] = []
    for n in sorted(group, key=lambda x: float(x["start"])):
        a, b = float(n["start"]), float(n["end"])
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _clashes(a: list[tuple[float, float]], b: list[tuple[float, float]],
             tol: float) -> bool:
    """Do two occupied-time lists overlap anywhere?  Both are sorted."""
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][1] - tol <= b[j][0]:
            i += 1
        elif b[j][1] - tol <= a[i][0]:
            j += 1
        else:
            return True
    return False


def _merge_intervals(a: list[tuple[float, float]],
                     b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for x, y in sorted(a + b):
        if out and x <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], y)
        else:
            out.append([x, y])
    return [(x, y) for x, y in out]


def _link_sequential(groups: list[list[dict]]) -> list[list[dict]]:
    """Join groups that never sound at the same time into one line.

    The chord merge above can only ever join things that overlap: it asks
    whether two lines attack together or ring together. So a line that stops and
    starts again — a melody resting for a bar, a phrase leaping more than an
    octave, anything that made `_provisional_contours` open a new contour — is
    split forever, however obviously it continues.

    Joining two groups whose times do not clash costs nothing: no note moves and
    no duration is shortened, because a printed voice only ever needed its notes
    not to overlap. Measured on a 144 s piano track this is the difference
    between 30 advertised "sequences" and 9, against 10 notes ever sounding at
    once, with no note lost.

    Occupied time is compared span by span rather than first-note-to-last, so a
    short figure can sit inside a long line's rest instead of forcing a line of
    its own. Register distance decides between candidates, and caps how far a
    line may be stitched.
    """
    if len(groups) < 2:
        return groups
    ordered = sorted(groups, key=lambda g: min(float(n["start"]) for n in g))
    streams: list[dict] = []
    for group in ordered:
        iv = _intervals(group)
        pitch = median(int(n["pitch"]) for n in group)
        free = [st for st in streams
                if abs(st["pitch"] - pitch) <= LINK_SPAN
                and not _clashes(st["iv"], iv, LINK_GAP)]
        if free:
            st = min(free, key=lambda st: abs(st["pitch"] - pitch))
            st["notes"].extend(group)
            st["iv"] = _merge_intervals(st["iv"], iv)
            st["pitch"] = median(int(n["pitch"]) for n in st["notes"])
        else:
            streams.append({"notes": list(group), "iv": iv, "pitch": pitch})
    return [st["notes"] for st in streams]


# A sequence has to be big enough to be worth showing as its own line: at least
# this many notes AND this share of the track. Below that it is a handful of
# stray events, and advertising it as a part just lengthens the stem list.
#
# Measured on eval/refs_band against the reference MIDI's real instrument
# assignment, scoring "are these two notes in the same part" pairwise: folding
# slivers back took 64 lines to 57 for a pairwise F1 of 0.612 vs 0.615, i.e.
# 11% fewer lines at no real cost.
# How polyphonic a track has to be before it is split at all.
#
# This was 0.18, which is barely more than "two notes ever overlap", so nearly
# every track got split. Measured on eval/refs_band against the reference MIDI's
# real instrument assignment: the split at 0.18 produced 73 lines for a pairwise
# F1 of 0.602, against 49 lines and 0.595 for not splitting at all — 24 extra
# lines bought 0.007. Raising the gate to 0.50 gives 64 lines and 0.615, better
# on both counts, and the curve is flat either side of it (0.35 -> 0.608,
# 0.65 -> 0.605) so it is not a tuned-to-the-set knife edge.
SPLIT_POLY_GATE = 0.50

# A sequence has to be big enough to be worth showing as its own line. The
# threshold is ABSOLUTE on purpose. It was briefly
#     floor = max(MIN_SEQUENCE_NOTES, 0.20 * len(track))
# which looks harmless on the 25-second eval clips (tracks of 100-500 notes give
# a floor of 24-100) and is catastrophic on a real song: a two-minute piano part
# of 1086 notes gets a floor of 217, so every genuine secondary voice is folded
# into the first. That part is 91% polyphonic with up to seven simultaneous
# notes, which one notated voice cannot hold, and build_score then drops
# whatever does not fit — 490 of 1637 notes missing from the printed score.
#
#   흡수 규칙              밴드 라인  밴드 F1  곡 성부  악보 음   손실
#   흡수 안 함                  64    0.615     11    1637     0%
#   max(절대, 비율)  (버그)      57    0.611      5    1316   19.6%
#   절대값만          (현재)      59    0.614     11    1637     0%
#
# The share term bought two fewer lines and cost a fifth of the score.
MIN_SEQUENCE_NOTES = 24

# Each sequence becomes its own staff downstream (app._mt3_stems), so more
# than a couple is not "more voices found" to a reader, it is a single
# instrument's chord voicings fragmenting into a wall of near-identical
# staves — a piano playing melody-plus-harmony coming back as five equal-
# looking "성부" lines instead of a melody and an accompaniment. Real
# notation for a polyphonic instrument tops out at two staves (a grand
# staff) for exactly this reason.
MAX_SEQUENCE_PARTS = 2


def _absorb_slivers(parts: list[list[dict]], min_notes: int) -> list[list[dict]]:
    """Fold sequences too small to stand on their own into the largest one."""
    if len(parts) < 2:
        return parts
    keep = [p for p in parts if len(p) >= min_notes]
    if not keep or len(keep) == len(parts):
        return parts
    host = max(keep, key=len)
    for p in parts:
        if len(p) < min_notes:
            host.extend(p)
    return keep


def separate_sequences(notes: list[dict], chord_gap: float = 0.035,
                       continuity_gap: float = 2.4,
                       min_notes: int = MIN_SEQUENCE_NOTES) -> list[list[dict]]:
    """Infer musical sequences using global chord-pattern evidence.

    1. Build provisional monophonic contours (every note is retained).
    2. Compare complete contours, not individual onset groups.
    3. Merge contours only if they repeatedly attack together, stay in the
       same register, and keep a near-constant vertical interval.  Such a
       component is a chordal sequence.  Other contours remain independent.

    A one-off extra tone has no sequence evidence, so it is absorbed into a
    nearby chord rather than being advertised as a new part.
    """
    ns = sorted(({**n, "start": float(n["start"]), "end": float(n["end"])}
                 for n in notes if float(n["end"]) > float(n["start"])),
                key=lambda n: (n["start"], n["pitch"]))
    if len(ns) < 2:
        return [ns] if ns else []

    # This stage only supplies atomic contours.  The decision to call something
    # a chord happens below with look-ahead across the complete clip.
    atoms = _provisional_contours(ns, chord_gap, continuity_gap)
    parent = list(range(len(atoms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    proposals = []
    conflicts: set[tuple[int, int]] = set()
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            score, conflict = _chord_verdict(atoms[i], atoms[j], chord_gap + 0.02)
            if score is not None:
                proposals.append((score, i, j))
            if conflict:
                conflicts.add((i, j))

    # Merging is transitive, so a pairwise test alone is not enough: single
    # linkage merged A-B and B-C into one group even when A-C had been rejected,
    # letting a "chord" exceed the octave limit _chord_verdict enforces on each
    # pair — three lines an octave apart collapsed into one 28-semitone
    # sequence. A join is now blocked when any cross pair actively conflicts.
    # Pairs that merely lack shared attacks do not block it.
    members: dict[int, list[int]] = {i: [i] for i in range(len(atoms))}
    for _score, i, j in sorted(proposals):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if any((min(a, b), max(a, b)) in conflicts
               for a in members[ri] for b in members[rj]):
            continue
        join(i, j)
        merged = members.pop(ri) + members.pop(rj)
        members[find(i)] = merged

    groups: dict[int, list[dict]] = {}
    for i, atom in enumerate(atoms):
        groups.setdefault(find(i), []).extend(atom)

    # An isolated, simultaneous colour note is a chord extension unless it has
    # enough repeated events to establish its own rhythmic sequence.
    stable = {root for root, group in groups.items()
              if len({round(float(n["start"]), 3) for n in group}) >= 2}
    for root in list(groups):
        if root in stable:
            continue
        transient = groups[root]
        candidates = []
        for other in stable:
            for n in transient:
                for m in groups[other]:
                    if (abs(float(n["start"]) - float(m["start"])) <= chord_gap + 0.02
                            and abs(int(n["pitch"]) - int(m["pitch"])) <= 12):
                        candidates.append((abs(int(n["pitch"]) - int(m["pitch"])), other))
        if candidates:
            target = min(candidates)[1]
            groups[target].extend(transient)
            del groups[root]

    parts = _link_sequential([g for g in groups.values() if g])
    parts = _absorb_slivers(parts, min_notes)
    parts = [sorted(group, key=lambda n: (n["start"], n["pitch"])) for group in parts]
    if len(parts) > MAX_SEQUENCE_PARTS:
        parts = _melody_plus_accompaniment(parts)
    else:
        parts.sort(key=lambda part: -median(n["pitch"] for n in part))
    return parts


def _melody_plus_accompaniment(parts: list[list[dict]]) -> list[list[dict]]:
    """Collapse 3+ inferred sequences to (the melody, everything else).

    "The melody" is picked by note count first, register second: a moving
    tune visits far more distinct pitches than a held chord or pad does even
    when the pad happens to sit higher, so register alone (the previous
    sort key) could and did put a static high pad in front of the actual
    tune. This is a heuristic, not a melody detector — it will occasionally
    pick a busy bassline over a sparse vocal line — but it is a better
    default than "whichever is highest," and it caps what the score shows
    as separate staves at two regardless of how many sequences the pairwise
    chord/contour analysis above found.
    """
    lead = max(parts, key=lambda part: (len(part), median(n["pitch"] for n in part)))
    rest = [n for part in parts if part is not lead for n in part]
    out = [lead]
    if rest:
        out.append(sorted(rest, key=lambda n: (n["start"], n["pitch"])))
    return out


def separate_voices(notes: list[dict], max_voices: int | None = None,
                    chord_gap: float = 0.03, min_dur: float = 0.03) -> list[list[dict]]:
    """Infer pitch-continuous, non-overlapping notation lines.

    The normal mode has *no fixed voice count*. At every onset it assigns a
    note to an available line, preferring the most continuous pitch/register
    history. The output count grows only when simultaneous notes require it;
    every detected event is retained. ``max_voices`` is legacy explicit-cap
    support only.
    """
    ns = sorted(({**n, "start": float(n["start"]), "end": float(n["end"])}
                 for n in notes if float(n["end"]) > float(n["start"])),
                key=lambda n: (n["start"], -n["pitch"]))
    if len(ns) < 2:
        return [ns] if ns else []

    cap = None if max_voices is None else max(1, int(max_voices))
    voices: list[dict] = []
    i = 0
    while i < len(ns):
        t0, group = ns[i]["start"], []
        while i < len(ns) and ns[i]["start"] - t0 <= chord_gap:
            group.append(ns[i])
            i += 1
        group.sort(key=lambda n: -n["pitch"])
        used: set[int] = set()
        for n in group:
            free = [vi for vi, v in enumerate(voices)
                    if vi not in used and v["free_at"] <= n["start"] + 1e-6]
            if free:
                vi = min(free, key=lambda k: _continuity_cost(voices[k], n))
            elif cap is None or len(voices) < cap:
                voices.append(_new_voice())
                vi = len(voices) - 1
            else:
                choices = [k for k in range(len(voices)) if k not in used]
                vi = min(choices, key=lambda k: (
                    abs((voices[k]["last_pitch"] if voices[k]["last_pitch"] is not None else n["pitch"])
                        - n["pitch"])
                    + 18.0 * max(0.0, voices[k]["free_at"] - n["start"])))
                _finish_at(voices[vi], n["start"], min_dur)
            voice = voices[vi]
            voice["notes"].append(n)
            voice["last_pitch"] = int(n["pitch"])
            voice["free_at"] = n["end"]
            voice["pitches"].append(int(n["pitch"]))
            used.add(vi)

    parts = [v["notes"] for v in voices if v["notes"]]
    for part in parts:
        part.sort(key=lambda n: (n["start"], n["pitch"]))
    parts.sort(key=lambda p: -median(n["pitch"] for n in p))
    return parts
