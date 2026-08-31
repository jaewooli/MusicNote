"""
Beat, downbeat and time-signature detection from transcribed notes.

Why not librosa: `librosa.beat.beat_track` reads a spectral-flux onset envelope
off the raw audio, which is weak on music without percussion and prone to
half/double tempo errors. Measured on eval/refs_meter it got the tempo right on
2 of 12 clips (8 doubles), beat F1 0.636, and it produces no downbeat at all.
But by the time we need a beat grid we already HAVE the onsets: MT3 transcribes
them at onset F1 0.958. This module works from those instead.

    detect(notes) -> {tempo, beats, downbeats, time_sig, beats_per_bar,
                      compound, confidence}

`notes` is the pipeline's own note dicts: {start, end, pitch, velocity}.
"""
from __future__ import annotations

import math

import numpy as np

# The TATUM (fastest regular pulse) is what onset timing actually shows, so it
# is what gets searched. 0.10 .. 0.80 s covers a 32nd at a slow tempo through a
# quarter at a fast one.
MIN_TATUM = 0.10
MAX_TATUM = 0.80
# Searched in log space; 900 steps over the tatum range puts neighbours about
# 0.23 % apart, well under anything audible.
N_PERIODS = 900
# Onsets this close are one musical event (a chord, or a rolled one).
CHORD_TOL = 0.035
# How near a beat an onset must fall to count as occupying that slot.
SLOT_TOL = 0.15
# Bar lengths in tatums, restricted to those that actually decompose into a
# notatable (tatums-per-beat, beats-per-bar). Leaving this open produced bars of
# 7 and 14 tatums, i.e. 7/4 and 14/4.
BAR_TATUMS = (2, 3, 4, 6, 8, 9, 12, 16)
# Bar-length prior: log-normal around 2.4 s, so 1.2 .. 4.8 s sits inside one
# sigma. This is a far tighter and better-founded constraint than a tempo
# prior — bar lengths vary much less across music than tempi do.
BAR_CENTRE = 2.4
BAR_WIDTH = 0.75
# Tatums per beat. 3 means compound time (6/8, 9/8, 12/8).
BEAT_TATUMS = (1, 2, 3, 4)
# Beats per bar worth notating.
METERS = (2, 3, 4, 6)
# Beat-tempo prior, used only to choose among the beat levels a bar allows.
PRIOR_CENTRE = 100.0
PRIOR_WIDTH = 1.1
# Keeps a reading with no accent contrast scoring above zero, so the priors can
# still decide between two structurally equal readings.
CONTRAST_FLOOR = 0.05
# How much the intermediate-beat accent counts next to the downbeat accent.
BEAT_WEIGHT = 0.4


def _events(notes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Note starts collapsed into musical events.

    -> (times, weights, min_pitch, max_duration). Weight is how many notes
    attack together: a four-note chord is stronger evidence of a beat than a
    passing sixteenth, and it is what makes downbeats stand out later.
    """
    ns = sorted((n for n in notes if n.get("start") is not None),
                key=lambda n: float(n["start"]))
    t: list[float] = []
    w: list[float] = []
    lo: list[int] = []
    dur: list[float] = []
    for n in ns:
        s = float(n["start"])
        p = int(n.get("pitch", 60))
        d = max(0.0, float(n.get("end", s)) - s)
        if t and s - t[-1] <= CHORD_TOL:
            w[-1] += 1.0
            lo[-1] = min(lo[-1], p)
            dur[-1] = max(dur[-1], d)
        else:
            t.append(s)
            w.append(1.0)
            lo.append(p)
            dur.append(d)
    return (np.asarray(t), np.asarray(w), np.asarray(lo, dtype=float),
            np.asarray(dur))


def _align(t: np.ndarray, w: np.ndarray, tau: float,
           span: float) -> tuple[float, float, float]:
    """(strength, phase, occupancy) for one candidate beat period.

    The best-phase alignment of onsets to a grid of spacing tau is just the
    magnitude of Z = sum w_i exp(2 pi i o_i / tau); its argument is that phase.
    Occupancy is the fraction of grid slots that actually contain an onset —
    on its own it rules out periods far too short to be beats.
    """
    z = np.sum(w * np.exp(2j * np.pi * t / tau))
    strength = float(abs(z) / w.sum())
    phase = float((np.angle(z) / (2 * np.pi)) * tau) % tau
    k = np.round((t - phase) / tau)
    hit = np.abs((t - phase) - k * tau) < SLOT_TOL * tau
    n_slots = max(1, int(round(span / tau)) + 1)
    occupancy = min(1.0, len(np.unique(k[hit])) / n_slots)
    return strength, phase, occupancy


def _grid(t0: float, tau: float, phase: float, end: float) -> list[float]:
    k0 = math.ceil((t0 - phase) / tau)
    out = []
    k = k0
    while phase + k * tau <= end + 1e-9:
        v = phase + k * tau
        if v >= -1e-9:
            out.append(round(float(v), 6))
        k += 1
    return out


def _tatum(t: np.ndarray, w: np.ndarray, span: float) -> tuple[float, float, float]:
    """Fastest regular pulse the onsets actually lie on.

    This is the one thing onset TIMING determines reliably. It is not the beat:
    a tune in running eighths has its onsets on the eighth grid whatever the
    notated beat is, which is why estimating the beat directly from |Z| gave 10
    double-tempo errors out of 16. The beat is recovered later, from accent.
    """
    taus = np.geomspace(MIN_TATUM, MAX_TATUM, N_PERIODS)
    best = (0.0, 0.0, 0.0)
    for tau in taus:
        strength, phase, occupancy = _align(t, w, tau, span)
        sc = strength * occupancy
        if sc > best[0]:
            best = (sc, tau, phase)
    return best[1], best[2], best[0]


def _accents(t: np.ndarray, w: np.ndarray, lo: np.ndarray, dur: np.ndarray,
             tau: float, phase: float, end: float) -> tuple[np.ndarray, list[float]]:
    """Accent strength on each tatum slot, and the slot times.

    Three cues, each scaled by its own maximum and summed: how many notes attack
    there, how low the lowest of them is, and how long the longest is. A bar
    starts where the bass lands and where a long note begins. Empty slots score
    zero, which is itself the strongest possible statement that nothing is
    accented there.
    """
    k0 = max(0, math.ceil(-phase / tau))
    slots = [phase + k * tau for k in range(k0, int((end - phase) / tau) + 1)]
    n = len(slots)
    acc, bass, leng = np.zeros(n), np.zeros(n), np.zeros(n)
    win = SLOT_TOL * tau * 2.0
    for i, b in enumerate(slots):
        m = np.abs(t - b) <= win
        if not m.any():
            continue
        acc[i] = float(w[m].sum())
        bass[i] = max(0.0, 108.0 - float(lo[m].min()))
        leng[i] = float(dur[m].max())

    def norm(x: np.ndarray) -> np.ndarray:
        top = float(x.max())
        return x / top if top > 1e-9 else x

    return norm(acc) + norm(bass) + norm(leng), slots


def _compound(t: np.ndarray, tau: float) -> bool:
    """Is the beat three tatums long (6/8, 9/8, 12/8) rather than two or four?

    Decided from inter-onset intervals, not from accent. Accent is the natural
    cue but it is far too noisy here — on the reference set it picked compound
    time about as often on simple-time clips as on compound ones. Timing is
    solid: compound music is full of three-tatum steps and simple music is full
    of two- and four-tatum ones.
    """
    if len(t) < 12:
        return False
    steps = np.round(np.diff(t) / tau)
    three = float(np.sum((steps == 3) | (steps == 6)))
    duple = float(np.sum((steps == 2) | (steps == 4)))
    return three > duple * 1.25


def _beat_span(tau: float, compound: bool) -> int:
    """Tatums per beat. In compound time this is fixed at three; otherwise the
    beat is whichever level lands nearest an ordinary tempo."""
    if compound:
        return 3
    best = (0.0, 1)
    for s_beat in (1, 2, 4):
        bpm = 60.0 / (s_beat * tau)
        p = math.exp(-0.5 * (math.log2(bpm / PRIOR_CENTRE) / PRIOR_WIDTH) ** 2)
        if p > best[0]:
            best = (p, s_beat)
    return best[1]


def _beat_phase(a: np.ndarray, s_beat: int) -> int:
    """Which tatum in each group of s_beat carries the beat.

    Decided before the bar, and on its own evidence. Reading it off the bar
    phase instead makes it as unreliable as the bar phase is, and a beat grid
    off by one tatum scores zero however good the tempo was.  There are n/s
    samples behind this choice against n/B behind the bar's, so it is the far
    steadier of the two.
    """
    if s_beat <= 1:
        return 0
    idx = np.arange(len(a))
    best = (0, -1e9)
    for q in range(s_beat):
        on = idx % s_beat == q
        if not on.any() or on.all():
            continue
        sc = float(a[on].mean() - a[~on].mean())
        if sc > best[1]:
            best = (q, sc)
    return best[0]


def _structure(a: np.ndarray, tau: float, s_beat: int,
               q: int) -> tuple[int, int, float]:
    """(beats per bar, bar phase in tatums, contrast) once the beat grid is set.

    Only the bar is left to accent, which is the one thing accent is actually
    good for: the downbeat is where the bass lands and where long notes start.
    """
    n = len(a)
    idx = np.arange(n)
    sd = float(a.std()) or 1.0
    on_beat = idx % s_beat == q
    best = (4, q, -1e9)
    for m in METERS:
        B = s_beat * m
        if n < B * 2:
            continue
        bar_prior = math.exp(
            -0.5 * (math.log2(B * tau / BAR_CENTRE) / BAR_WIDTH) ** 2)
        for j in range(m):                     # a downbeat is always a beat
            r = q + j * s_beat
            on_down = idx % B == r
            others = on_beat & ~on_down
            if not on_down.any() or not others.any():
                continue
            sc = float(a[on_down].mean() - a[others].mean()) / sd * bar_prior
            if sc > best[2]:
                best = (m, r, sc)
    return best


def detect(notes: list[dict]) -> dict:
    """Beat grid, downbeats and time signature for a transcription."""
    t, w, lo, dur = _events(notes or [])
    empty = {"tempo": 0.0, "beats": [], "downbeats": [], "time_sig": (4, 4),
             "beats_per_bar": 4, "compound": False, "confidence": 0.0}
    if len(t) < 8:
        return empty
    span = float(t[-1] - t[0])
    if span <= 0:
        return empty

    tau, phase, conf = _tatum(t, w, span)
    if tau <= 0:
        return empty
    a, slots = _accents(t, w, lo, dur, tau, phase, float(t[-1]) + tau)
    if len(slots) < 8:
        return empty

    compound = _compound(t, tau)
    s_per_beat = _beat_span(tau, compound)
    q = _beat_phase(a, s_per_beat)
    m, r, contrast = _structure(a, tau, s_per_beat, q)
    B = s_per_beat * m

    beats = [b for i, b in enumerate(slots) if i % s_per_beat == q]
    downbeats = [b for i, b in enumerate(slots) if i % B == r]

    return {
        "tempo": round(60.0 / (s_per_beat * tau), 1),
        "beats": [round(float(b), 6) for b in beats],
        "downbeats": [round(float(b), 6) for b in downbeats],
        "time_sig": (m * 3, 8) if compound else (m, 4),
        "beats_per_bar": m,
        "compound": compound,
        "bar_tatums": B,
        "tatum": round(tau, 5),
        "confidence": round(float(conf), 4),
        "downbeat_margin": round(float(contrast), 4),
    }
