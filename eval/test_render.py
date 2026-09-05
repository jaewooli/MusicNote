#!/usr/bin/env python3
"""Every measure the score builder emits must survive the renderer.

VexFlow is the only thing that knows whether the notation layer's tick
arithmetic holds up — it rejects a voice whose durations do not add up and it is
what actually has to fit three voices onto one staff. This builds the scores the
other tests build, hands them to `eval/render_check.js`, and fails on any bar
VexFlow will not format.

    python eval/test_render.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from score_build import build_score  # noqa: E402


def line(pitches, t0=0.0, step=0.5, dur=0.45):
    return [{"start": t0 + i * step, "end": t0 + i * step + dur,
             "pitch": p, "velocity": 90} for i, p in enumerate(pitches)]


def docs():
    """A crowded staff, a triplet grid, a grand staff and a drum kit."""
    yield "crowded", build_score([{"name": "Piano", "voices": [
        line([84, 83, 81], step=0.7), line([79, 77, 76], step=0.55),
        line([74, 72, 71], step=0.45), line([69, 67, 66], step=0.65),
        line([64, 62, 61], step=0.5), line([59, 57, 55], step=0.6),
        line([52, 50, 48], step=0.75)]}], tempo=120, time_sig=(4, 4))
    yield "triplets", build_score([{"name": "Piano", "voices": [
        line([72, 74, 76, 77, 79, 77, 76, 74], step=1.0 / 3, dur=0.30),
        line([48, 50, 52, 53], step=0.5, dur=0.45)]}], tempo=120, time_sig=(4, 4))
    yield "grand", build_score([{"name": "Piano", "notes":
        line([36, 48, 60, 72, 84, 72, 60, 48], step=0.4)}],
        tempo=120, time_sig=(4, 4))
    yield "drums", build_score([{"name": "Kit", "is_drum": True, "notes":
        line([36, 42, 38, 42, 36, 42, 38, 46], step=0.25, dur=0.1)}],
        tempo=120, time_sig=(4, 4))
    yield "flat_key", build_score([{"name": "Horn", "notes":
        line([61, 63, 65, 66, 68, 70, 72, 61], step=0.5)}],
        tempo=120, time_sig=(4, 4))

    # Plus real transcriptions — synthetic input cannot produce the note
    # densities a whole band does, and those are what crowd a bar until the
    # formatter gives up. The band references carry cached MT3 runs, so this
    # covers dense music on any checkout. A job snapshot, when the work dir
    # happens to hold a finished one, is used as well: it is the only input
    # that has been through the live request path.
    import audit_score as AS
    for mid in sorted((ROOT / "eval" / "refs_band").glob("*.mid")):
        if not mid.with_suffix(".mt3.json").exists():
            continue
        _, _, doc = AS.rebuild(AS.from_caches(mid))
        yield mid.stem, doc
    for snap in sorted((ROOT / "uploads").glob("*.job.json")):
        job = json.loads(snap.read_text())
        if "mt3_raw" not in job:
            continue          # a failed or in-flight job holds no transcription
        _, _, doc = AS.rebuild(job)
        yield snap.name[:8], doc
        break


def main() -> int:
    if not (ROOT / "frontend" / "vendor" / "vexflow.js").exists():
        print("vexflow.js is not vendored — skipping", file=sys.stderr)
        return 0
    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, doc in docs():
            p = Path(tmp) / f"{name}.json"
            p.write_text(json.dumps(asdict(doc)))
            paths.append(str(p))
        r = subprocess.run(["node", str(ROOT / "eval" / "render_check.js"), *paths],
                           capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode:
        raise AssertionError("VexFlow refused a measure the score builder emitted")
    print("all render checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
