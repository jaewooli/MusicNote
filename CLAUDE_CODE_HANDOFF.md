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
     evidence: repeated aligned attacks, near-constant interval, and <= octave
     register distance. It is not performer/source separation.
- `backend/score_build.py`: beat-grid quantisation, chords/rests/ties -> ScoreDoc.
- `frontend/musicnote-core.js`: renders ScoreDoc with VexFlow and aligns playhead
  using measure start/end times.

## Important distinction

`MT3 track` = model-predicted instrument track.

`sequence` = musical pattern inferred inside that track.

`notation voice` = simultaneous written layer. Do not label sequences as
separate performers. `_score_parts` now groups same MT3 source track into one
score Part with `voices`, instead of drawing every sequence as a separate
instrument staff.

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
python3 -m py_compile backend/voices.py backend/app.py backend/quality.py backend/mt3_worker.py
pm2 restart musicnote
pm2 restart mt3-worker
```

The sequence test covers: parallel chord, chord + independent melody, one-off
chord extension, bass + chord, and short MT3 offset-overrun melody.

## Remaining priority work

1. Run the Canon clip after padding and compare MT3 raw tail / confirmed
   candidates. Tune candidate timing and pitch tolerance from this evidence.
2. Render `confirmed_missing_notes` as piano-roll markers; user approval only
   should update notes / MIDI / MusicXML.
3. Replace the legacy editor's top-note-only renderer with the ScoreDoc renderer.
4. Improve piano notation with grand staff / hands. Current score is one part
   with multiple VexFlow voices but no proper piano grand staff.
5. Implement remote GPU protocol documented in `deploy/VAST_SERVERLESS.md`.

## Operational notes

- App: `pm2` process `musicnote`; MT3 worker: `mt3-worker`.
- Local service: `http://127.0.0.1:8731`.
- GitHub remote: `https://github.com/jaewooli/MusicNote`, branch `main`.
- Git credentials are configured locally for this repository. Do not print,
  commit, or alter credential files.
