"""
Offline transcription accuracy check with mir_eval — so tuning stops being blind.

    # fast: call transcribe.py directly (no Demucs, no MT3)
    python backend/eval_harness.py --engine direct:polyphonic eval/refs

    # honest: drive the real HTTP pipeline the user actually gets
    python backend/eval_harness.py --engine api:mt3 eval/refs
    python backend/eval_harness.py --engine api:polyphonic eval/refs

A reference dir holds pairs `<stem>.mid` (ground truth) + `<stem>.wav` (audio);
build one with `eval/build_refs.py`.

CAUTION: `direct:polyphonic` runs basic-pitch inside transcribe.py, but the
product's own "polyphonic" mode is served by MT3 (see app.py `is_mt3`). So
`direct:` is a fast proxy for the library, NOT for what a user receives. Use
`api:` for shipped behaviour, or `eval/replay_eval.py` (cached MT3 output) when
tuning post-processing, since MT3 here runs at roughly 25x realtime.

Reported per clip and as a mean:
  * onset F1        — onset within 50 ms (did we find the note at all)
  * note F1         — onset 50 ms + pitch (the standard AMT number)
  * note+off F1     — additionally offset within 20 % of the note (duration too)
Plus est/ref note counts, so over- and under-detection are visible separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

API = "http://127.0.0.1:8731"


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _arrays(pairs):
    iv = np.array([[a, b] for a, b, _ in pairs], dtype=float) if pairs else np.zeros((0, 2))
    p = np.array([hz(m) for _, _, m in pairs], dtype=float) if pairs else np.zeros(0)
    return iv, p


def ref_notes(mid: Path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(mid))
    out = [(n.start, n.end, n.pitch)
           for i in pm.instruments if not i.is_drum for n in i.notes]
    out.sort()
    return _arrays(out)


def est_direct(wav: Path, mode: str):
    import transcribe as T
    r, _ = T.transcribe(str(wav), mode=mode)
    return r, [(n["start"], n["end"], n["pitch"]) for n in r["notes"]]


def est_api(wav: Path, mode: str, timeout: float = 2400.0):
    import mimetypes  # noqa: F401  (kept: multipart body is hand-built below)
    boundary = "----musicnoteEval"
    body = b""
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\n{mode}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
             f'filename="{wav.name}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += wav.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API + "/api/transcribe", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    jid = json.loads(urllib.request.urlopen(req, timeout=120).read())["job_id"]

    t0 = time.time()
    while time.time() - t0 < timeout:
        p = json.loads(urllib.request.urlopen(API + f"/api/progress/{jid}", timeout=30).read())
        if p["status"] == "done":
            r = p["result"]
            return r, [(n["start"], n["end"], n["pitch"]) for n in r["notes"]]
        if p["status"] == "error":
            raise RuntimeError(p.get("error", "job failed"))
        time.sleep(3)
    raise TimeoutError(f"{wav.name}: job timed out")


def score(ref_iv, ref_p, est_iv, est_p):
    import mir_eval
    _, _, f_on = mir_eval.transcription.onset_precision_recall_f1(
        ref_iv, est_iv, onset_tolerance=0.05)
    _, _, f_note, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    _, _, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2)
    return f_on, f_note, f_off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("--engine", default="direct:polyphonic",
                    help="direct:<mode> | api:<mode>   (mode: melody|polyphonic|stems|mt3)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    kind, _, mode = a.engine.partition(":")
    mode = mode or "polyphonic"
    refs = sorted(Path(a.refdir).glob("*.mid"))
    if a.limit:
        refs = refs[:a.limit]
    if not refs:
        print(f"no *.mid in {a.refdir}")
        return 2

    print(f"engine = {kind}:{mode}   clips = {len(refs)}\n")
    print(f"{'clip':10s} {'engine':16s} {'onsetF1':>8s} {'noteF1':>7s} {'+offF1':>7s} "
          f"{'ref':>5s} {'est':>5s} {'sec':>6s}")
    rows = []
    for mid in refs:
        wav = mid.with_suffix(".wav")
        if not wav.exists():
            continue
        ref_iv, ref_p = ref_notes(mid)
        t0 = time.time()
        try:
            r, est = (est_api(wav, mode) if kind == "api" else est_direct(wav, mode))
        except Exception as e:  # noqa: BLE001
            print(f"{mid.stem:10s} FAILED: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        est_iv, est_p = _arrays(est)
        f_on, f_note, f_off = score(ref_iv, ref_p, est_iv, est_p)
        rows.append((f_on, f_note, f_off))
        print(f"{mid.stem:10s} {str(r.get('engine'))[:16]:16s} {f_on:8.3f} {f_note:7.3f} "
              f"{f_off:7.3f} {len(ref_iv):5d} {len(est_iv):5d} {dt:6.1f}")

    if rows:
        m = np.array(rows).mean(axis=0)
        print(f"\nMEAN   onset F1 = {m[0]:.3f}   note F1 = {m[1]:.3f}   "
              f"note+offset F1 = {m[2]:.3f}   over {len(rows)} clip(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
