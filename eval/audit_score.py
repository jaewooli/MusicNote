#!/usr/bin/env python3
"""Audit the notation a persisted job produces, without re-running MT3.

Reads a `<job>.job.json` snapshot from the work dir, rebuilds the score exactly
as `/api/score` does, and reports what a reader would actually see: how many
transcribed notes reached the page, how many ledger lines they cost, how much of
each staff is rest, and how the durations are spelled.

    python eval/audit_score.py uploads/<job>.job.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import app as A  # noqa: E402
from score_build import build_score, _ledger_cost  # noqa: E402


def from_caches(mid: Path, shifts=("1024", "1536"), agreement: int = 2) -> dict:
    """A job-shaped dict for a reference clip, from its cached MT3 runs.

    Lets the notation metrics run over the whole reference set instead of the
    one persisted job, without touching MT3.
    """
    import mt3_ensemble as E
    import mt3_post as MP
    import meter as MT

    runs = []
    for c in [mid.with_suffix(".mt3.json")] + [
            mid.with_suffix(f".mt3.s{t}.json") for t in shifts]:
        if not c.exists():
            continue
        kept, _ = MP.gate(json.loads(c.read_text())["notes"], 0.5)
        runs.append([{"start": float(n["start"]), "end": float(n["end"]),
                      "pitch": int(n["pitch"]), "velocity": int(n["velocity"]),
                      "track": int(n["track"]), "program": int(n.get("program", 0)),
                      "is_drum": bool(n.get("is_drum", False))} for n in kept])
    merged = E.merge(runs)
    # The product moves a whole part back into its own register before anything
    # downstream sees it, so the audit has to as well — octave displacement is
    # printed on the page like any other pitch.
    wav = mid.with_suffix(".wav")
    if wav.exists():
        A._mt3_octaves(merged, str(wav))
    accepted, _ = E.split(merged, len(runs), agreement)
    # NOT applied here: the live pipeline's bass-rescue pass (app._mt3_bass_rescue)
    # needs a transposed-up MT3 run that this cache-only audit has no copy of. So
    # a notational audit run through this path under-counts recall below
    # BASS_RESCUE_CUTOFF relative to what a real job actually produces.
    met = MT.detect(accepted)
    return {"mt3_raw": merged, "mt3_runs": len(runs), "mt3_split_voices": True,
            "mt3_max_voices": None,
            "result": {"tempo": met.get("tempo") or 120.0,
                       "beats": list(met.get("beats") or []),
                       "downbeats": list(met.get("downbeats") or []),
                       "time_sig": list(met.get("time_sig") or (4, 4)),
                       "duration": max(n["end"] for n in accepted) if accepted else 0.0}}


def rebuild(job: dict):
    raw = job["mt3_raw"]
    stems = A._mt3_stems("audit", raw, 0.5,
                         split_voices=job.get("mt3_split_voices", True),
                         max_voices=job.get("mt3_max_voices"),
                         runs=int(job.get("mt3_runs", 1)))
    res = dict(job["result"])
    res["stems"] = stems
    parts = A._score_parts(res, None)
    ts = res.get("time_sig") or [4, 4]
    doc = build_score(parts, beats=res.get("beats") or None,
                      downbeats=res.get("downbeats") or None,
                      tempo=float(res.get("tempo") or 120.0),
                      time_sig=(int(ts[0]), int(ts[1])),
                      title="audit")
    return stems, parts, doc


def audit(doc, parts) -> dict:
    src = sum(len(p.get("notes") or []) + sum(len(v) for v in p.get("voices") or [])
              for p in parts)
    seen, ledger, rests, dur, tied, tuplets = 0, 0.0, collections.Counter(), \
        collections.Counter(), 0, 0
    staff_notes: dict[tuple[int, int], list[int]] = {}
    staff_slots: dict[tuple[int, int], list[int]] = {}
    for pi, part in enumerate(doc.parts):
        for voice in part.voices:
            st = int(getattr(voice, "staff", 1) or 1)
            key = (pi, st)
            for meas in voice.measures:
                for ev in meas.events:
                    if ev.notes:
                        # A tied note is several events but ONE notehead the
                        # reader counts: only the head of a tie chain is new.
                        seen += sum(1 for n in ev.notes if not n.tie_stop)
                        staff_notes.setdefault(key, []).extend(
                            n.midi for n in ev.notes if not n.unpitched)
                        rests[key] = rests.get(key, 0)
                        if any(n.tie_start or n.tie_stop for n in ev.notes):
                            tied += 1
                    else:
                        rests[key] = rests.get(key, 0) + 1
                    staff_slots.setdefault(key, []).append(1)
                    dur[(ev.type, ev.dots)] += 1
                    if getattr(ev, "tuplet", None):
                        tuplets += 1
    for (pi, st), ps in staff_notes.items():
        clefs = getattr(doc.parts[pi], "clefs", None) or ["treble"]
        clef = clefs[min(st, len(clefs)) - 1]
        if clef in ("treble", "bass"):   # a drum staff has no ledger geometry
            ledger += _ledger_cost(ps, clef)
    return {"source_notes": src, "score_notes": seen,
            "lost": src - seen, "ledger": ledger,
            "durations": dur, "tied_events": tied, "tuplet_events": tuplets,
            "staff_rest_share": {f"part{p}/staff{s}":
                                 (rests.get((p, s), 0) / max(1, len(staff_slots[(p, s)])))
                                 for (p, s) in sorted(staff_slots)},
            "staff_notes": {f"part{p}/staff{s}": len(staff_notes.get((p, s), []))
                            for (p, s) in sorted(staff_slots)}}


def timed_noteheads(doc, only_part=None):
    """Every notehead as (audio seconds, pitch, notated seconds).

    A measure carries the audio interval it stands for, so ticks inside it map
    back to time linearly — that is what the playhead already uses.

    ``only_part`` restricts the walk to one part. Without it the caller gets a
    flat list in which a kick drum (MIDI 36) and a bass C2 are the same key, and
    matching by pitch alone silently pairs one with the other.
    """
    out = []
    for part in ([only_part] if only_part is not None else doc.parts):
        for voice in part.voices:
            open_ties: dict[int, int] = {}     # midi -> index in `out`
            for meas in voice.measures:
                if meas.start is None or meas.end is None:
                    continue
                span = float(meas.end) - float(meas.start)
                total = sum(ev.dur for ev in meas.events) or 1
                t = 0
                for ev in meas.events:
                    at = float(meas.start) + span * (t / total)
                    secs = span * (ev.dur / total)
                    for n in ev.notes:
                        if n.tie_stop and n.midi in open_ties:
                            out[open_ties[n.midi]][2] += secs
                        else:
                            out.append([at, n.midi, secs])
                        if n.tie_start:
                            open_ties[n.midi] = len(out) - 1
                        elif n.midi in open_ties:
                            open_ties.pop(n.midi, None)
                    t += ev.dur
    return out


def fidelity(doc, parts) -> dict:
    """How faithfully does the page reproduce the notes it was given?

    Matching runs part by part, and length is reported for PITCHED parts only.
    A drum hit carries no sustain — MT3 emits them a few milliseconds long — so
    printing one as a 32nd is correct notation, not a length error. Pooled with
    the pitched notes they were 46% of every match on the band clips and dragged
    the median length ratio to 1.88, which said the page was stretching notes
    when it was doing the right thing with percussion.
    """
    import numpy as np
    on_err, dur_ratio, unmatched, drum_on = [], [], 0, []
    for src_part, doc_part in zip(parts, doc.parts):
        drums = bool(src_part.get("is_drum"))
        src = []
        for group in ([src_part["notes"]] if src_part.get("notes")
                      else (src_part.get("voices") or [])):
            src += [(float(n["start"]), int(n["pitch"]),
                     float(n["end"]) - float(n["start"])) for n in group]
        by_pitch: dict[int, list] = {}
        for at, midi, secs in timed_noteheads(doc, doc_part):
            by_pitch.setdefault(midi, []).append((at, secs))
        for v in by_pitch.values():
            v.sort()
        for s0, pitch, dur in sorted(src):
            cand = by_pitch.get(pitch)
            if not cand:
                unmatched += 1
                continue
            i = min(range(len(cand)), key=lambda i: abs(cand[i][0] - s0))
            (drum_on if drums else on_err).append(abs(cand[i][0] - s0))
            if dur > 1e-3 and not drums:
                dur_ratio.append(cand[i][1] / dur)
    on = np.array(on_err) if on_err else np.zeros(1)
    dr = np.array(dur_ratio) if dur_ratio else np.ones(1)
    dn = np.array(drum_on) if drum_on else np.zeros(1)
    return {"onset_median": float(np.median(on)), "onset_p90": float(np.percentile(on, 90)),
            "onset_within_50ms": float((on <= 0.05).mean()),
            "onset_within_1_16": float((on <= 0.5 * 60.0 / max(1.0, doc.tempo) / 2).mean()),
            "drum_onset_within_50ms": float((dn <= 0.05).mean()),
            "drum_notes": len(drum_on),
            "dur_median_ratio": float(np.median(dr)),
            "dur_within_25pct": float((np.abs(np.log(np.maximum(dr, 1e-6))) <= 0.223).mean()),
            "unmatched": unmatched}


def measure_lengths(doc) -> dict:
    """Every printed measure must hold exactly one bar of ticks."""
    from fractions import Fraction
    num, den = doc.time_sig
    bar = int(doc.divisions * 4 * Fraction(num, den))
    bad, total, worst, missing = 0, 0, 0, 0
    for part in doc.parts:
        for voice in part.voices:
            for meas in voice.measures:
                got = sum(ev.dur for ev in meas.events)
                total += 1
                if got != bar:
                    bad += 1
                    worst = max(worst, abs(got - bar))
                    missing += abs(got - bar)
    return {"bars": total, "wrong": bad, "worst_ticks": worst, "bar_ticks": bar,
            "missing_ticks": missing}


def against_truth(doc, parts, mid: Path) -> dict | None:
    """Score the printed page against a reference MIDI, alongside its input.

    `fidelity` asks how closely the page reproduces what MT3 handed it. That is
    the wrong question wherever MT3 is wrong, and note-offs are exactly where it
    is: printing a note out to the next attack moves it AWAY from MT3 and
    TOWARDS the truth. Only a reference says which happened, so where one exists
    this reports both the page and its own input on the same scale — notation is
    doing its job as long as it is not giving away what it was handed.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from replay_eval import ref_notes, score, _arrays
    except Exception:
        return None
    ref_iv, ref_p = ref_notes(mid)
    if len(ref_iv) == 0:
        return None
    # Is this reference's note length NOTATION, or is it articulation?
    #
    # Several references are MIDI renderings whose note-offs are a plucked or
    # staccato envelope: in eval/refs the median reference note is 45-73 ms and
    # 77-92% of them are under 100 ms. A correct score of that music prints
    # 16ths — about 190 ms at those tempos — and mir_eval's offset test then
    # marks every one of them wrong. Read naively that says notation destroys
    # offset accuracy (0.846 -> 0.152 on ref00) when the page is right and the
    # reference simply does not carry notated lengths.
    #
    # So the offset comparison is reported only where the reference plausibly
    # holds notated durations. Onsets are unaffected and always comparable.
    short = float(((ref_iv[:, 1] - ref_iv[:, 0]) < 0.1).mean())
    notational = short <= 0.6
    src, page = [], []
    for sp, dp in zip(parts, doc.parts):
        if sp.get("is_drum"):
            continue          # the references list pitched notes only
        for group in ([sp["notes"]] if sp.get("notes") else (sp.get("voices") or [])):
            src += [(float(n["start"]), float(n["end"]), int(n["pitch"]))
                    for n in group]
        for at, midi, secs in timed_noteheads(doc, dp):
            # The beat grid may start before t=0 — an anacrusis is real music —
            # but mir_eval rejects a negative interval.
            a0 = max(0.0, at)
            if at + secs > a0:
                page.append((a0, at + secs, midi))
    if not src or not page:
        return None
    a = score(ref_iv, ref_p, *_arrays(sorted(src)))
    b = score(ref_iv, ref_p, *_arrays(sorted(page)))
    return {"input": {"onset_f1": a[0][2], "note_f1": a[1][2], "offset_f1": a[2][2]},
            "page": {"onset_f1": b[0][2], "note_f1": b[1][2], "offset_f1": b[2][2]},
            "notational_lengths": notational, "short_ref_share": short}


def summarise(paths) -> int:
    """One aggregate line per metric over a whole reference directory.

    A single score says whether one page is good; tuning a notation constant
    needs the number across the set, and reading 21 reports to add them up is
    how a regression in one clip gets missed.
    """
    import numpy as np
    rows, src, lost, bars, wrong, missing, worst = [], 0, 0, 0, 0, 0, 0
    for mid in paths:
        job = from_caches(mid)
        if not job["mt3_raw"]:
            # No cached MT3 run for this clip. It must be SKIPPED, not scored:
            # an empty score has no note to get wrong, so every ratio metric
            # comes back perfect and quietly pulls the average up. Half of
            # eval/refs_meter is in this state, which is how a "21 score"
            # average came out well above what any of the 8 real ones scored.
            print(f"  skip {mid.stem}: no cached MT3 run "
                  f"({mid.with_suffix('.mt3.json').name} is missing)")
            continue
        try:
            _stems, parts, doc = rebuild(job)
        except Exception as e:
            print(f"  skip {mid.stem}: {type(e).__name__} {e}")
            continue
        f, r, ml = fidelity(doc, parts), audit(doc, parts), measure_lengths(doc)
        if not r["source_notes"]:
            print(f"  skip {mid.stem}: the run held no notes")
            continue
        t = against_truth(doc, parts, mid)
        rows.append((f, t))
        src += r["source_notes"]; lost += r["lost"]
        bars += ml["bars"]; wrong += ml["wrong"]
        missing += ml["missing_ticks"]; worst = max(worst, ml["worst_ticks"])
    if not rows:
        print("no scores to summarise")
        return 1
    k = lambda n: np.mean([f[n] for f, _ in rows])
    print(f"{len(rows)} scores")
    print(f"  lost from the page     {lost}/{src} ({lost/max(1,src):.2%})")
    print(f"  pitched onset <=50 ms  {k('onset_within_50ms'):.1%}")
    print(f"  drum onset <=50 ms     {k('drum_onset_within_50ms'):.1%}   "
          f"({sum(f['drum_notes'] for f, _ in rows)} hits)")
    print(f"  pitched length <=25%   {k('dur_within_25pct'):.1%}   "
          f"median ratio {k('dur_median_ratio'):.3f}")
    print(f"  bars exactly one bar   {(bars-wrong)/max(1,bars):.2%}   "
          f"{missing} ticks unaccounted, worst bar off by {worst}")
    truth = [t for _, t in rows if t]
    if truth:
        print(f"  vs reference: note F1  input "
              f"{np.mean([t['input']['note_f1'] for t in truth]):.3f}   "
              f"page {np.mean([t['page']['note_f1'] for t in truth]):.3f}   "
              f"(onset {np.mean([t['input']['onset_f1'] for t in truth]):.3f} -> "
              f"{np.mean([t['page']['onset_f1'] for t in truth]):.3f}, "
              f"{len(truth)} clips)")
        keep = [t for t in truth if t["notational_lengths"]]
        if keep:
            print(f"  vs reference: +offset  input "
                  f"{np.mean([t['input']['offset_f1'] for t in keep]):.3f}   "
                  f"page {np.mean([t['page']['offset_f1'] for t in keep]):.3f}   "
                  f"({len(keep)} of {len(truth)} clips; the rest are rendered "
                  f"staccato and carry no notated lengths)")
        else:
            print("  vs reference: +offset  no clip has notated lengths to "
                  "compare against")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="a <id>.job.json snapshot, a reference .mid, "
                                "or a directory of .mid references to summarise")
    a = ap.parse_args()
    path = Path(a.job)
    if path.is_dir():
        return summarise(sorted(path.glob("*.mid")))
    job = (from_caches(path) if path.suffix == ".mid"
           else json.loads(path.read_text()))
    stems, parts, doc = rebuild(job)
    r = audit(doc, parts)
    print(f"raw MT3 notes      {len(job['mt3_raw'])}")
    print(f"stems              {len(stems)}   lines {sum(len(s['notes']) and 1 for s in stems)}")
    print(f"notes into score   {r['source_notes']}")
    print(f"notes on the page  {r['score_notes']}   lost {r['lost']} "
          f"({r['lost']/max(1,r['source_notes']):.2%})")
    print(f"measures           {max(len(v.measures) for p in doc.parts for v in p.voices)}")
    print(f"ledger lines       {r['ledger']:.0f}")
    print(f"tied events        {r['tied_events']}    tuplet events {r['tuplet_events']}")
    print("durations          " + "  ".join(
        f"{t}{'.'*d}={c}" for (t, d), c in r["durations"].most_common(10)))
    ml = measure_lengths(doc)
    print(f"measure lengths    {ml['bars'] - ml['wrong']}/{ml['bars']} exact "
          f"({(ml['bars']-ml['wrong'])/max(1,ml['bars']):.1%})   worst off by "
          f"{ml['worst_ticks']} of {ml['bar_ticks']} ticks; "
          f"{ml['missing_ticks']} ticks unaccounted in all")
    f = fidelity(doc, parts)
    print(f"onset fidelity     median {f['onset_median']*1000:.0f} ms   "
          f"p90 {f['onset_p90']*1000:.0f} ms   within 50 ms {f['onset_within_50ms']:.1%}   "
          f"within a 16th {f['onset_within_1_16']:.1%}")
    print(f"length fidelity    median ratio {f['dur_median_ratio']:.2f}   "
          f"within 25% {f['dur_within_25pct']:.1%}   "
          f"(pitched only; {f['drum_notes']} drum hits scored on onset alone, "
          f"{f['drum_onset_within_50ms']:.1%} within 50 ms)")
    if path.suffix == ".mid":
        t = against_truth(doc, parts, path)
        if t:
            print(f"vs reference MIDI  input  onset {t['input']['onset_f1']:.3f}  "
                  f"note {t['input']['note_f1']:.3f}  +offset {t['input']['offset_f1']:.3f}")
            print(f"                   page   onset {t['page']['onset_f1']:.3f}  "
                  f"note {t['page']['note_f1']:.3f}  +offset {t['page']['offset_f1']:.3f}")
            if not t["notational_lengths"]:
                print(f"                   NOTE: {t['short_ref_share']:.0%} of this "
                      f"reference's notes are under 100 ms, so its note-offs are "
                      f"articulation, not\n"
                      f"                   notated length. The +offset column "
                      f"punishes correct notation here; read onset and note F1.")
    print("staff fill:")
    for k, share in r["staff_rest_share"].items():
        print(f"   {k:16s} notes={r['staff_notes'][k]:5d}  rest slots={share:5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
