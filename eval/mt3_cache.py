#!/usr/bin/env python3
"""Cache raw MT3 worker output for every clip in a reference dir.

MT3 on this CPU box runs at roughly 25x realtime, so re-running inference for
every post-processing experiment is not workable. This stores the raw note list
once; `eval/replay_eval.py` then scores any gate / voicing change in seconds.

    python eval/mt3_cache.py eval/refs
    python eval/mt3_cache.py eval/refs --shift 1.024   # half-segment ensemble run

A shifted run is stored alongside as `<stem>.mt3.s1024.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import mt3_bridge as MT3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("--force", action="store_true", help="re-run clips already cached")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="seconds of silent lead-in (YourMT3 segment is 2.048 s, "
                         "so 1.024 is a half-segment offset)")
    a = ap.parse_args()

    if not MT3.available():
        print("mt3 worker is not reachable", file=sys.stderr)
        return 2

    wavs = sorted(Path(a.refdir).glob("*.wav"))
    if not wavs:
        print(f"no *.wav in {a.refdir}", file=sys.stderr)
        return 2

    tag = "" if not a.shift else f".s{int(round(a.shift * 1000))}"
    for wav in wavs:
        out = wav.with_suffix(f".mt3{tag}.json")
        if out.exists() and not a.force:
            print(f"{wav.stem}: cached")
            continue
        t0 = time.time()
        try:
            res = MT3.transcribe(str(wav.resolve()), shift=a.shift)
        except Exception as e:  # noqa: BLE001
            print(f"{wav.stem}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            continue
        out.write_text(json.dumps({"model": res.get("model"),
                                   "duration": res.get("duration"),
                                   "shift": res.get("shift", a.shift),
                                   "tracks": res.get("tracks", []),
                                   "notes": res.get("notes", [])}))
        print(f"{wav.stem}: {len(res.get('notes', []))} raw notes "
              f"in {time.time() - t0:.0f}s -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
