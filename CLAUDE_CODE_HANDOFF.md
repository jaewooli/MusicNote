# MusicNote Claude Code handoff

## Goal

Turn short audio / YouTube clips into a piano roll, readable score, MIDI and
MusicXML. The user values musical correctness over a fixed number of voices:
a chord may be one sequence, while independently continuing patterns should be
separate.

## Current architecture

- `backend/app.py`: FastAPI routes and MT3 result assembly.
- `backend/mt3_worker.py`: separate PM2 process using `yourmt3`; accepts one WAV
  path and returns MIDI-like notes. It now pads the right edge by 0.75 s before
  inference and clamps emitted events to original duration.
- `backend/voices.py`: two-stage sequence interpretation:
  1. `_provisional_contours` joins notes with pitch/register continuity and
     permits <=180 ms MT3 offset overrun. Simultaneous notes are assigned jointly
     with minimum cost, preventing crossing caused by greedy matching.
  2. `separate_sequences` merges contours into a chord only with whole-clip
     evidence, and there are two admissible kinds: repeated aligned attacks at a
     near-constant interval, or sustained ringing together at a steady interval
     (`_sustain_overlap`). Both stay within an octave. It is not
     performer/source separation.
- `backend/score_build.py`: beat-grid quantisation, chords/rests/ties -> ScoreDoc,
  `_beat_grid` extrapolates in FULL periods (a clamped short first beat used to
  push the opening onsets onto a 32nd grid, so plain eighths printed as a
  dotted-16th rest plus off-beat eighths),
  then the *notation layout*: `_plan_staves` splits a keyboard-range part over a
  grand staff, and `_reduce_voices` folds each staff down to two printed voices
  (`MAX_VOICES_PER_STAFF`). Merging is lossless for pitch: simultaneous events
  become chords, and only sounding *lengths* are shortened.
- `frontend/musicnote-core.js`: renders ScoreDoc with VexFlow. It measures every
  bar with `preCalculateMinTotalWidth` and packs bars into systems by the width
  they need, so dense bars stop spilling notes past their barline. The playhead
  reads the drawn note positions (`Score._layout[].marks`), not an even spread
  over the bar; `scoreTimeAt` is its exact inverse, so clicking the score seeks
  to the instant the playhead was showing. Beams are built before the notes are
  drawn (that is what removes their flags) and are handed the WHOLE voice
  including rests, because `generateBeams` finds beat boundaries by adding up
  the durations it is given.

## Important distinction

`MT3 track` = model-predicted instrument track.

`sequence` = musical pattern inferred inside that track.

`notation voice` = simultaneous written layer. Do not label sequences as
separate performers. `_score_parts` now groups same MT3 source track into one
score Part with `voices`, instead of drawing every sequence as a separate
instrument staff.

The three are separated by *stage*, and each has its own criterion:

1. MT3 track — the model's own instrument prediction, never re-judged.
2. Sequence (`voices.separate_sequences`) — contours joined by melodic
   continuity (`_leap_cost`, new line costs a fixed 12.0, no join across a
   2.4 s gap), then merged into a chord on either **attack** evidence
   (`ATTACK_RATIO` of shared onsets at a near-constant interval) or **sustain**
   evidence (`_sustain_overlap`: ringing together for `SUSTAIN_FRACTION` of the
   shorter line at a steady interval). The sustain route exists because a held
   pad or a broken chord never attacks together, and used to be advertised as
   one new sequence per chord tone.
3. Notation voice (`score_build`) — how those sequences are printed: at most
   `MAX_VOICES_PER_STAFF` per staff, across at most two staves.

## Known validated issue: Canon Shorts

Test URL: `https://www.youtube.com/shorts/1tY7Jg5Ap7A`.

- 8.208 s has a strong CQT attack near E4 but MT3 emits no event there. This is
  an MT3 omission before sequence splitting or score construction.
- Source has an additional end attack around 12.56 s; prior MT3 output ended at
  G5 near 12.54 s. Right-padding is now applied to improve end-boundary recall.
- The score builder did not drop either note; if a note is absent from MT3 raw,
  it cannot appear in the score.

## Validation now

`backend/quality.py`:

1. finds missing CQT onset candidates;
2. runs independent high-recall keyboard transcription (`piano-tx`, with
   Basic Pitch/CQT fallback) once per MT3 job;
3. exposes only matching candidates as `validation.confirmed_missing_notes`.

Never auto-insert these notes. Next UI work: draw confirmed candidates on the
piano roll and add an explicit user approval action that adds a note.

## Tests / commands

```bash
python3 eval/test_sequence_analysis.py
python3 eval/test_score_layout.py
python3 -m py_compile backend/voices.py backend/app.py backend/quality.py backend/mt3_worker.py
node --check frontend/musicnote-core.js
pm2 restart musicnote
pm2 restart mt3-worker
```

The sequence test covers: parallel chord, chord + independent melody, one-off
chord extension, bass + chord, and short MT3 offset-overrun melody.

`test_score_layout.py` covers the printed page rather than the notes: key and
length read from a part's `voices`, the grand-staff split, the two-voice cap,
and that folding voices together loses no pitch.

## Measured on eval/refs (6 clips, 3262 notes through the full chain)

Numbers to re-check after touching `score_build.py`; the scripts that produced
them are one-off, but each claim below is reproducible from `eval/refs`.

| | before | after |
|---|---|---|
| notes lost inside the score builder | 82 (2.5%) | **16 (0.49%)** |
| measures printing exactly one measure | 50.6% | **86.0%** |
| measures printing MORE than a measure | 6 | **0** |

Where the lost notes went: 56 were two attacks of one pitch rounding onto the
same grid tick and being deduplicated (fixed by charging `_pick_subdiv` for a
collision, plus a nudge in `_quantise`); 26 were unisons created by folding
voices onto one staff (fixed by charging unisons in `_merge_cost`). The
remaining 16 are genuine unisons, where one notehead is correct.

**Barlines follow the detected downbeat.** `meter.detect` locates downbeats
from accent evidence and `build_score` now takes them (`downbeats=`) and phases
the bars onto them, prepending beats so anything earlier becomes an anacrusis
and then dropping whole empty leading bars. Before this the score counted bars
from wherever the extrapolated grid began, and the barline was wrong on 5 of 7
measured clips — one of them by 3 beats. All 7 now land on the downbeat exactly.
Downbeats are only valid alongside the beats they came from, so an overridden
tempo drops both.

**Nothing in the pipeline measures loudness.** YourMT3 emits velocity 100 for
every note (verified: 610/610, 745/745, 414/414 on eval/refs). So `<dynamics>`
in the MusicXML is derived from a constant and means nothing, and the `conf`
field in the API is likewise constant — real per-note confidence is the
ensemble's `agreement`. What `meter._accents` calls an accent is not amplitude
either: it is how many notes attack on a slot, how low the lowest is, and how
long the longest is. Adding real dynamics needs an amplitude measurement the
pipeline does not currently make.

Two model-level facts worth not re-deriving:

- **Onsets are trustworthy, lengths are not.** est/ref duration ratio over 2977
  matched notes: median 0.90, p10 0.42, p90 1.23, and 17.4% off by more than
  half. This is why `_absorb_short_rests` builds the printed rhythm from onsets.
- **The first note of a clip needs a lead-in.** The very first onset is found
  6/13 times by a plain run and 12/13 with the 1.024 s silent lead-in the
  ensemble's second pass already uses; the union recovers it, and it survives to
  the printed score 12/13. The end is already covered by
  `END_PADDING_SECONDS` (last onset 6/7 either way). The 1.024 s run is also
  slightly better overall (F1 0.951 vs 0.947), so a permanent lead-in on the
  primary run is worth measuring — but it must NOT be a multiple of the 2.048 s
  segment, or the two ensemble runs land on the same boundaries and the second
  pass stops finding anything.

## Remaining priority work

1. Run the Canon clip after padding and compare MT3 raw tail / confirmed
   candidates. Tune candidate timing and pitch tolerance from this evidence.
2. Render `confirmed_missing_notes` as piano-roll markers; user approval only
   should update notes / MIDI / MusicXML.
3. Replace the legacy editor's top-note-only renderer with the ScoreDoc renderer.
   It is the last consumer that re-derives notation; it also records no tick
   marks, so the playhead falls back to an even spread inside each bar there.
4. `_plan_staves` assigns whole lines to a staff, and on dense material that is
   not enough: `separate_sequences` returns lines spanning three to five octaves
   (measured on eval/refs: `160n 40-84`, `90n 31-94`), which no staff can hold.
   47.6% of printed notes there need ledger lines whichever split is chosen —
   the planner is not the bottleneck, the granularity is. Real engraving assigns
   hands per note, not per line. Short sparse clips are already fine (4-5%).
5. Implement remote GPU protocol documented in `deploy/VAST_GPU.md`.

## Operational notes

- App: `pm2` process `musicnote`; MT3 worker: `mt3-worker`.
- Local service: `http://127.0.0.1:8731`.
- GitHub remote: `https://github.com/jaewooli/MusicNote`, branch `main`.
- Git credentials are configured locally for this repository. Do not print,
  commit, or alter credential files.
