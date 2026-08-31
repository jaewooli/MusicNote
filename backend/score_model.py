"""
ScoreDoc — the notation document model.

The missing abstraction. Until now every consumer (VexFlow on screen,
musicxml.py on export, the editor) re-derived notation from a flat
``{start,end,pitch}`` list, each with its own code and its own 16th-note grid.
That is why the score could not express triplets, chords, or real durations, and
why display and export drifted apart.

Now: notes + a beat grid are converted ONCE into a ScoreDoc

    Score → Part → Voice → Measure → [Chord | Rest]
                                        └ Note(step, alter, octave, tie)

with explicit note values (type + dots + tuplet), and every consumer renders
that. Times inside the doc are in *ticks* (DIVISIONS per quarter note), never
seconds, so the metrical structure is the source of truth.

Standalone: no numpy, no librosa — just the stdlib, so it is cheap to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

DIVISIONS = 480          # ticks per quarter note (MusicXML `divisions`)

# note value (in quarters) -> MusicXML type name
_TYPES: list[tuple[Fraction, str]] = [
    (Fraction(8), "breve"), (Fraction(4), "whole"), (Fraction(2), "half"),
    (Fraction(1), "quarter"), (Fraction(1, 2), "eighth"), (Fraction(1, 4), "16th"),
    (Fraction(1, 8), "32nd"), (Fraction(1, 16), "64th"),
]
_STEPS = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
_ALTER_SHARP = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
_STEPS_FLAT = ["C", "D", "D", "E", "E", "F", "G", "G", "A", "A", "B", "B"]
_ALTER_FLAT = [0, -1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0]

# major-key fifths -> set of pitch classes in the key
_FIFTHS_PCS = {
    0: {0, 2, 4, 5, 7, 9, 11}, 1: {7, 9, 11, 0, 2, 4, 6}, 2: {2, 4, 6, 7, 9, 11, 1},
    3: {9, 11, 1, 2, 4, 6, 8}, 4: {4, 6, 8, 9, 11, 1, 3}, 5: {11, 1, 3, 4, 6, 8, 10},
    -1: {5, 7, 9, 10, 0, 2, 4}, -2: {10, 0, 2, 3, 5, 7, 9}, -3: {3, 5, 7, 8, 10, 0, 2},
    -4: {8, 10, 0, 1, 3, 5, 7},
}


@dataclass
class Note:
    step: str
    alter: int
    octave: int
    midi: int
    tie_start: bool = False
    tie_stop: bool = False
    velocity: int = 90


@dataclass
class Chord:
    """One rhythmic event: 1+ simultaneous notes sharing a duration."""
    notes: list[Note]
    dur: int                       # ticks
    type: str = "quarter"
    dots: int = 0
    tuplet: tuple[int, int] | None = None    # (actual, normal) e.g. (3, 2)
    tuplet_start: bool = False
    tuplet_stop: bool = False

    @property
    def is_rest(self) -> bool:
        return not self.notes


@dataclass
class Measure:
    number: int
    # Original-audio interval represented by this notated measure. These keep
    # the score playhead aligned when a detected beat grid is not perfectly
    # constant-tempo.
    start: float | None = None
    end: float | None = None
    events: list[Chord] = field(default_factory=list)
    # set only when they change
    key_fifths: int | None = None
    time_sig: tuple[int, int] | None = None
    tempo: float | None = None
    clef: str | None = None


@dataclass
class Voice:
    number: int
    measures: list[Measure] = field(default_factory=list)
    # Which staff of the part this voice is printed on (1 = top). A keyboard
    # part spanning both hands prints on two staves; everything else on one.
    staff: int = 1


@dataclass
class Part:
    id: str
    name: str
    program: int = 0
    is_drum: bool = False
    clef: str = "treble"
    voices: list[Voice] = field(default_factory=list)
    # One clef per staff, top to bottom. `clef` stays as the top staff's clef
    # so older consumers keep working.
    staves: int = 1
    clefs: list[str] = field(default_factory=lambda: ["treble"])


@dataclass
class ScoreDoc:
    title: str = "MusicNote"
    parts: list[Part] = field(default_factory=list)
    divisions: int = DIVISIONS
    key_fifths: int = 0
    time_sig: tuple[int, int] = (4, 4)
    tempo: float = 120.0

    def note_count(self) -> int:
        return sum(len(c.notes)
                   for p in self.parts for v in p.voices
                   for m in v.measures for c in m.events)


# --------------------------------------------------------------------------- #
# duration → notated value(s)
# --------------------------------------------------------------------------- #
MAX_DOTS = 1     # double dots are rare in real engraving; a tie reads better


def _plain(q: Fraction) -> tuple[str, int] | None:
    """A single note value (with up to MAX_DOTS dots) worth exactly `q` quarters."""
    for base, name in _TYPES:
        for dots in range(MAX_DOTS + 1):
            mult = Fraction(2) - Fraction(1, 2 ** dots)   # 1, 3/2
            if base * mult == q:
                return name, dots
    return None


def split_duration(ticks: int, tup: tuple[int, int] | None = None) -> list[tuple[str, int]]:
    """Ticks → list of (type, dots) to be tied together. Inside a tuplet the
    value is scaled by normal/actual before being named."""
    out: list[tuple[str, int]] = []
    rem = Fraction(ticks, DIVISIONS)
    if tup:
        rem = rem * tup[0] / tup[1]        # notated value is longer than sounded
    guard = 0
    while rem > 0 and guard < 16:
        guard += 1
        hit = _plain(rem)
        if hit:
            out.append(hit)
            break
        # largest plain value that fits, then continue with the remainder
        best = None
        for base, name in _TYPES:
            for dots in range(MAX_DOTS, -1, -1):
                mult = Fraction(2) - Fraction(1, 2 ** dots)
                v = base * mult
                if v <= rem and (best is None or v > best[0]):
                    best = (v, name, dots)
        if not best:
            break
        out.append((best[1], best[2]))
        rem -= best[0]
    return out or [("16th", 0)]


def spell(midi: int, fifths: int) -> tuple[str, int, int]:
    """MIDI → (step, alter, octave), spelled for the key signature."""
    pc = midi % 12
    octave = midi // 12 - 1
    flat = fifths < 0
    step = (_STEPS_FLAT if flat else _STEPS)[pc]
    alter = (_ALTER_FLAT if flat else _ALTER_SHARP)[pc]
    if flat and alter == -1:
        # Cb/Fb never arise from this table; octave is unaffected
        pass
    return step, alter, octave


def krumhansl_fifths(pitches: list[int]) -> int:
    """Cheap key estimate → MusicXML `fifths`, over a pitch-class histogram."""
    if not pitches:
        return 0
    hist = [0] * 12
    for p in pitches:
        hist[p % 12] += 1
    total = sum(hist) or 1
    prof = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    best, best_f = None, 0
    for fifths, pcs in _FIFTHS_PCS.items():
        tonic = (fifths * 7) % 12
        sc = sum(hist[(tonic + i) % 12] / total * prof[i] for i in range(12))
        # small penalty for notes outside the key
        sc -= 0.6 * sum(hist[p] / total for p in range(12) if p not in pcs)
        if best is None or sc > best:
            best, best_f = sc, fifths
    return best_f
