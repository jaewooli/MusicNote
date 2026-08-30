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



def _onset_groups(notes: list[dict], chord_gap: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for note in notes:
        if not groups or float(note["start"]) - float(groups[-1][0]["start"]) > chord_gap:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def _shape_cost(previous: list[int], current: list[int], gap: float) -> float:
    """Chord/contour transition score; lower means one continuing sequence."""
    a, b = sorted(previous), sorted(current)
    if len(a) != len(b):
        return float("inf")
    ac, bc = median(a), median(b)
    # The interval shape identifies a chordal accompaniment even when it moves
    # in parallel; the centre identifies its register.
    shape = sum(abs((x - ac) - (y - bc)) for x, y in zip(a, b)) / len(a)
    centre = abs(ac - bc)
    return shape * 1.5 + centre * 0.42 + min(gap, 2.0) * 0.7


def separate_sequences(notes: list[dict], chord_gap: float = 0.035,
                       continuity_gap: float = 2.4) -> list[list[dict]]:
    """Infer independent musical sequences, allowing chords inside a sequence.

    This deliberately differs from :func:`separate_voices`: an onset cluster
    starts as one vertical event (a possible chord), not one line per pitch.
    At later clusters we match subsets to prior chord shapes and registers. A
    separate sequence is introduced only when a subset keeps recurring as an
    independent contour; otherwise simultaneous notes remain a chord in the
    same sequence. This is an interpretation for readable parts, not source
    separation or performer identification.
    """
    ns = sorted(({**n, "start": float(n["start"]), "end": float(n["end"])}
                 for n in notes if float(n["end"]) > float(n["start"])),
                key=lambda n: (n["start"], n["pitch"]))
    if not ns:
        return []

    lanes: list[dict] = []
    for group in _onset_groups(ns, chord_gap):
        group = sorted(group, key=lambda n: n["pitch"])
        now = float(group[0]["start"])
        remaining = set(range(len(group)))
        proposals = []
        for li, lane in enumerate(lanes):
            gap = now - lane["last_start"]
            if gap < -chord_gap or gap > continuity_gap:
                continue
            count = len(lane["last_pitches"])
            if count > len(group) or count > 4:
                continue
            # Large chord subsets become ambiguous and expensive; their full
            # shape is still considered when the onset cluster has that size.
            from itertools import combinations
            for idxs in combinations(range(len(group)), count):
                pitches = [int(group[k]["pitch"]) for k in idxs]
                cost = _shape_cost(lane["last_pitches"], pitches, gap)
                # Chord shape and register must both be plausible. A singleton
                # gets a slightly tighter bound so it cannot steal a chord tone.
                limit = 4.8 + 0.75 * count + 0.55 * min(gap, 2.0)
                if cost <= limit:
                    proposals.append((cost, -count, li, idxs, pitches))
        # Resolve the strongest continuations first. Disjoint subsets let a
        # melody and an accompaniment chord continue at the same onset.
        used_lanes = set()
        for _cost, _size, li, idxs, pitches in sorted(proposals):
            if li in used_lanes or not set(idxs) <= remaining:
                continue
            lane = lanes[li]
            chosen = [group[k] for k in idxs]
            lane["notes"].extend(chosen)
            lane["last_pitches"] = pitches
            lane["last_start"] = now
            lane["centres"].append(float(median(pitches)))
            remaining.difference_update(idxs)
            used_lanes.add(li)
        if remaining:
            # No prior contour explains these notes. Keep the vertical event
            # intact as one tentative chord; later events may establish an
            # independent line alongside it.
            chosen = [group[k] for k in sorted(remaining)]
            pitches = [int(n["pitch"]) for n in chosen]
            lanes.append({"notes": chosen, "last_pitches": pitches,
                          "last_start": now, "centres": [float(median(pitches))]})

    parts = [lane["notes"] for lane in lanes if lane["notes"]]
    for part in parts:
        part.sort(key=lambda n: (n["start"], n["pitch"]))
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
