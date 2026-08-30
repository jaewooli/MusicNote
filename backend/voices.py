"""Strict, continuity-oriented voice allocation.

This module deliberately does *not* claim to identify performers. It turns a
polyphonic note stream into readable, non-overlapping lines for notation.
"""
from __future__ import annotations

from statistics import median


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
    if len(pairs) < 2 or len(pairs) / min(len(a), len(b)) < 0.72:
        return None, False
    intervals = [int(right["pitch"]) - int(left["pitch"]) for left, right in pairs]
    centre = sum(intervals) / len(intervals)
    spread = sum((x - centre) ** 2 for x in intervals) / len(intervals)
    # Notes more than an octave apart are normally separate registers (bass /
    # melody), even when their rhythm happens to coincide.
    if abs(centre) > 12 or spread > 1.25 ** 2:
        return None, True
    # Prefer the most stable interval and the most complete rhythmic alignment.
    return spread + (1.0 - len(pairs) / min(len(a), len(b))) * 2.0, False


def _parallel_chord_score(a: list[dict], b: list[dict], tolerance: float) -> float | None:
    """Merge score for two lines, or None when they are not one chordal pattern."""
    return _chord_verdict(a, b, tolerance)[0]


def separate_sequences(notes: list[dict], chord_gap: float = 0.035,
                       continuity_gap: float = 2.4) -> list[list[dict]]:
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

    parts = [sorted(group, key=lambda n: (n["start"], n["pitch"]))
             for group in groups.values() if group]
    parts.sort(key=lambda part: -median(n["pitch"] for n in part))
    return parts


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
