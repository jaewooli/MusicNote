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
# MusicXML `fifths` -> the pitch classes its key signature spells without an
# accidental. Every signature an engraver actually writes is here: the old table
# stopped at four flats, so a piece in D-flat had to borrow a neighbour's
# signature and pay an accidental on every G-flat in it.
_FIFTHS_PCS = {f: frozenset(((f * 7) % 12 + i) % 12
                            for i in (0, 2, 4, 5, 7, 9, 11))
               for f in range(-6, 7)}


@dataclass
class Note:
    step: str
    alter: int
    octave: int
    midi: int
    tie_start: bool = False
    tie_stop: bool = False
    velocity: int = 90
    # A drum note has no pitch. `step`/`octave` are then the LINE it is drawn
    # on, not a sound, and `midi` is the GM kit piece. Consumers must print it
    # with no accidental and no key signature.
    unpitched: bool = False
    notehead: str = "normal"        # "normal" | "x" | "circle-x" | "diamond"


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


# General MIDI percussion -> (staff line, notehead). The lines follow the usual
# drum-kit convention read against a treble staff: kick in the bottom space,
# snare in the third space, hi-hat above the top line, cymbals crossed.
#
# This is a DISPLAY position, not a pitch. Note 36 is a kick drum, not C2, and
# treating those numbers as pitches is what let 1723 drum notes decide the key
# signature of a piece that has none of them in it.
_DRUM_MAP: dict[int, tuple[str, int, str]] = {
    35: ("F", 4, "normal"), 36: ("F", 4, "normal"),        # bass drum
    37: ("C", 5, "x"),                                      # side stick
    38: ("C", 5, "normal"), 40: ("C", 5, "normal"),         # snare
    39: ("C", 5, "x"),                                      # hand clap
    # The low floor tom sat in the bass drum's own space, so 69 of the 71 kit
    # collisions measured over eval/refs_band were a kick and a floor tom
    # hitting together and printing as one notehead. It goes on the bottom line.
    41: ("E", 4, "normal"), 43: ("G", 4, "normal"),         # floor toms
    45: ("A", 4, "normal"), 47: ("B", 4, "normal"),         # low / low-mid tom
    48: ("D", 5, "normal"), 50: ("E", 5, "normal"),         # hi-mid / high tom
    42: ("G", 5, "x"), 44: ("D", 4, "x"), 46: ("G", 5, "circle-x"),   # hi-hat
    49: ("A", 5, "x"), 57: ("A", 5, "x"),                   # crash
    51: ("F", 5, "x"), 59: ("F", 5, "x"), 53: ("F", 5, "diamond"),    # ride
    52: ("B", 5, "x"), 55: ("B", 5, "x"),                   # china / splash
    54: ("E", 5, "x"), 56: ("E", 5, "x"),                   # tambourine / cowbell
}
# Anything the table does not name still has to go somewhere legible.
_DRUM_FALLBACK = ("C", 5, "x")


def drum_spell(midi: int) -> tuple[str, int, str]:
    """GM percussion note -> (step, octave, notehead) to draw it on."""
    return _DRUM_MAP.get(int(midi), _DRUM_FALLBACK)


# Line of fifths. Position 0 is C, +1 a fifth up (G), -1 a fifth down (F); the
# seven naturals occupy -1..5 and every sharp/flat is seven steps away. A key
# signature is exactly a window on this line: `fifths` f spells [f-1, f+5]
# without an accidental.
_LOF_STEPS = ["F", "C", "G", "D", "A", "E", "B"]
# How far sharp of the key centre a chromatic note is spelled.
CHROMATIC_LEAN = 4


def _lof(position: int) -> tuple[str, int]:
    """Line-of-fifths position → (step, alter). 6 is F#, -8 is F-flat."""
    return _LOF_STEPS[(position + 1) % 7], (position + 1) // 7


def spell(midi: int, fifths: int) -> tuple[str, int, int]:
    """MIDI → (step, alter, octave), spelled for the key signature.

    The two fixed sharp/flat tables this replaced could not spell a key: they
    chose by the SIGN of `fifths` alone, so F-sharp major printed a natural F
    against its own signature and C major spelled a borrowed flat third as D#.
    Choosing the spelling nearest the key's centre on the line of fifths gets
    both right, and reproduces the usual choices everywhere in between.
    """
    pc = midi % 12
    # 7 * position ≡ pc (mod 12), and 7 is its own inverse mod 12.
    base = (7 * pc) % 12
    cands = [base + 12 * k for k in range(-2, 3)]
    if pc in _FIFTHS_PCS[fifths]:
        # In the key. The signature names exactly one spelling — the position
        # inside its window — and taking any other would print an accidental on
        # a note that needs none. F-sharp major's seventh is E-sharp, not F.
        best = next(p for p in cands if fifths - 1 <= p <= fifths + 5)
    else:
        # Chromatic. Double accidentals are never worth it here, and a natural
        # beats an alteration at the same distance: a chromatic F in D major is
        # an F with a natural sign, not an E-sharp. Among what is left, lean
        # sharp of the signature by `CHROMATIC_LEAN` — measured over 15 scores
        # by counting the accidentals each choice prints.
        centre = fifths + CHROMATIC_LEAN
        plain = [p for p in cands if abs(_lof(p)[1]) <= 1] or cands
        best = min(plain, key=lambda p: (abs(_lof(p)[1]), abs(p - centre), p))
    step, alter = _lof(best)
    # Cb sits in the octave above the B it sounds as, B# in the one below.
    return step, alter, (midi - alter) // 12 - 1


# Krumhansl-Kessler key profiles. BOTH are needed: scoring only the major one
# maps a minor piece onto its parallel major, whose signature is three sharps
# away from the right one. Measured on eval/refs_band, band01 came out in G
# major and printed 70 accidentals where its own signature (two flats, G minor)
# needs 6.
_PROFILE_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                  5.19, 2.39, 3.66, 2.29, 2.88]
_PROFILE_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                  4.75, 3.98, 2.69, 3.34, 3.17]


def krumhansl_fifths(pitches: list[int]) -> int:
    """Cheap key estimate → MusicXML `fifths`, over a pitch-class histogram.

    A minor key is reported as its relative major's signature, which is what a
    key signature actually is — A minor and C major are both `fifths` 0.
    """
    if not pitches:
        return 0
    hist = [0] * 12
    for p in pitches:
        hist[p % 12] += 1
    total = sum(hist) or 1
    best, best_f = None, 0
    for fifths, pcs in _FIFTHS_PCS.items():
        tonic = (fifths * 7) % 12
        # small penalty for notes the signature does not spell
        outside = 0.6 * sum(hist[p] / total for p in range(12) if p not in pcs)
        for prof, root in ((_PROFILE_MAJOR, tonic),
                           (_PROFILE_MINOR, (tonic + 9) % 12)):
            sc = sum(hist[(root + i) % 12] / total * prof[i]
                     for i in range(12)) - outside
            # Ties go to the simpler signature: six sharps and six flats spell
            # the same notes, and nobody wants to read the six.
            if best is None or (sc, -abs(fifths)) > (best, -abs(best_f)):
                best, best_f = sc, fifths
    return best_f
