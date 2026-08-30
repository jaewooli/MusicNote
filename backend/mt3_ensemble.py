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
    return int(note["pitch"]), int(note.get("track", 0))


def merge(runs: list[list[dict]], onset_tolerance: float = ONSET_TOLERANCE) -> list[dict]:
    """Merge runs, tagging each note with how many runs contain it.

    Notes are matched one-to-one within a pitch/track group by nearest onset, so
    a repeated note is not collapsed into its neighbour. The first run supplies
    the canonical timing; later runs only vote and contribute unmatched notes.
    """
    runs = [r for r in runs if r]
    if not runs:
        return []

    merged: list[dict] = [{**n, "agreement": 1, "runs": [0]} for n in runs[0]]
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
            else:
                entry = {**note, "agreement": 1, "runs": [ri]}
                merged.append(entry)
                buckets.setdefault(_key(note), []).append(entry)

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
