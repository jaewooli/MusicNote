"""Combine several MT3 runs of the same audio into one note list.

YourMT3 consumes non-overlapping 2.048 s segments, so an attack that lands on a
segment boundary can be dropped entirely. Running the model again with a
half-segment lead-in moves every boundary to a segment centre, and the two runs
disagree exactly where the model is unreliable.

That disagreement is a far better omission detector than a spectral heuristic:
it is produced by the same model on the same audio, so it carries no separate
false-positive mode of its own. `agreement` counts how many runs found a note,
which gives the UI a real per-note confidence — MT3 itself reports a constant
velocity and so provides none.
"""
from __future__ import annotations

ONSET_TOLERANCE = 0.05


def _key(note: dict) -> tuple:
    """Grouping key for matching the same note across runs.

    Deliberately NOT keyed on the track. MT3 assigns a program per note event
    and that assignment is not stable when the segment boundaries move: a note
    every run found can come back as track 0 in one and track 3 in another.
    Keyed on the track, merge then saw four different notes, each with
    agreement 1, and the vote threw all four away — a recall loss with no
    precision to show for it.

    Measured on eval/refs_band at agreement 2 (solo piano is one track, so the
    key never differed there and the numbers are identical):

        (pitch, track)     P 0.762  R 0.599  F1 0.671
        (pitch, is_drum)   P 0.737  R 0.633  F1 0.681
        (pitch)            P 0.734  R 0.633  F1 0.680

    is_drum stays in the key because a drum note's "pitch" is an instrument id,
    not a pitch: MIDI 38 on channel 10 is a snare, and merging it with a real
    D2 would be merging two unrelated events. The program is a guess; the
    pitched/percussive split is not.

    The surviving note keeps the track of whichever run found it first, which is
    also the run that supplies the canonical timing.
    """
    return int(note["pitch"]), bool(note.get("is_drum", False))


def merge(runs: list[list[dict]], onset_tolerance: float = ONSET_TOLERANCE) -> list[dict]:
    """Merge runs, tagging each note with how many runs contain it.

    Notes are matched one-to-one within a pitch/track group by nearest onset, so
    a repeated note is not collapsed into its neighbour. The first run supplies
    the canonical timing; later runs only vote and contribute unmatched notes.
    """
    runs = [r for r in runs if r]
    if not runs:
        return []

    merged: list[dict] = [{**n, "agreement": 1, "runs": [0],
                           "_ends": [float(n["end"])]} for n in runs[0]]
    for ri, run in enumerate(runs[1:], start=1):
        buckets: dict[tuple, list[dict]] = {}
        for m in merged:
            buckets.setdefault(_key(m), []).append(m)
        for note in run:
            pool = [m for m in buckets.get(_key(note), [])
                    if ri not in m["runs"]
                    and abs(float(m["start"]) - float(note["start"])) <= onset_tolerance]
            if pool:
                best = min(pool, key=lambda m: abs(float(m["start"]) - float(note["start"])))
                best["agreement"] += 1
                best["runs"].append(ri)
                best["_ends"].append(float(note["end"]))
            else:
                entry = {**note, "agreement": 1, "runs": [ri],
                         "_ends": [float(note["end"])]}
                merged.append(entry)
                buckets.setdefault(_key(note), []).append(entry)

    # Every agreeing run produced its own estimate of where the note stopped,
    # and the ensemble was throwing all but run 0's away. The median of them is
    # free accuracy — no extra inference, and it fixes nothing about which notes
    # exist, so onset F1 is untouched:
    #
    #                       밴드 onF1 / offF1     솔로 onF1 / offF1
    #   run 0 그대로          0.681 / 0.562        0.952 / 0.868
    #   end 중앙값            0.681 / 0.574        0.952 / 0.880
    #   end 최댓값            0.681 / 0.583        0.952 / 0.875
    #
    # The maximum scores higher on band material but lower on solo piano, and it
    # lengthens every note systematically, which gives the notation more overlaps
    # to resolve. The median is the robust estimate and gains the same +0.012 on
    # both sets. Medianing the START as well measured identically (band 0.573),
    # so it is left alone rather than added on a theory.
    for n in merged:
        ends = n.pop("_ends", None)
        if ends and len(ends) > 1:
            ends.sort()
            mid = ends[len(ends) // 2] if len(ends) % 2 else (
                (ends[len(ends) // 2 - 1] + ends[len(ends) // 2]) / 2.0)
            n["end"] = max(float(n["start"]) + 0.03, mid)

    merged.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))
    return merged


def split(merged: list[dict], total_runs: int, min_agreement: int = 1
          ) -> tuple[list[dict], list[dict]]:
    """Split merged notes into (delivered, review queue).

    These are two independent decisions and must not be conflated:

    * delivery — a note is delivered when it reached ``min_agreement`` runs.
      1 is union (highest recall); 2 is a vote (higher precision).
    * review   — a note is uncertain when *not every* run found it, whether or
      not it was delivered. Under union this flags notes that are in the score
      but shaky; under a vote it also lists the notes the vote withheld.

    Each review entry carries ``in_score`` so the UI can say "included, verify"
    versus "possible omission, approve to add". Nothing is ever auto-inserted.
    """
    if total_runs < 2:
        return list(merged), []
    delivered = [n for n in merged if n["agreement"] >= min_agreement]
    review = [{**n, "in_score": n["agreement"] >= min_agreement}
              for n in merged if n["agreement"] < total_runs]
    return delivered, review
