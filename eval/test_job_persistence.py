#!/usr/bin/env python3
"""A finished job has to survive a backend restart.

Job state used to live only in `app.JOBS`, so every restart — a deploy, a
config change, a crash — silently discarded whatever the user was looking at,
and the only symptom was a 404 on a job id they still had open in the browser.
These check the snapshot round-trip and, importantly, the things that must NOT
be restored: a job that was still running, and one past its TTL.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
os.environ.setdefault("MUSICNOTE_WORKDIR", tempfile.mkdtemp(prefix="mnjobs-"))
import app as A  # noqa: E402


def _done_job(jid: str = "abc123", **extra) -> dict:
    j = {
        "status": "done", "stage": "done", "pct": 1.0, "message": "완료",
        "created": time.time(), "mode": "mt3",
        "result": {"notes": [{"start": 0.0, "end": 0.5, "pitch": 60}],
                   "tempo": 120.0, "note_count": 1},
        "mt3_raw": [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100,
                     "program": 0, "is_drum": False, "track": 0}],
        "mt3_tempo": 120.0, "mt3_beats": [0.0, 0.5],
    }
    j.update(extra)
    A.JOBS[jid] = j
    return j


def _reload() -> None:
    """Simulate a process restart: memory is gone, disk is not."""
    A.JOBS.clear()
    A._restore()


def test_finished_job_comes_back():
    _done_job()
    A._persist("abc123")
    _reload()
    assert "abc123" in A.JOBS, "a finished job was not restored"
    assert A.JOBS["abc123"]["result"]["note_count"] == 1


def test_mt3_job_stays_retunable():
    _done_job("retune")
    A._persist("retune")
    _reload()
    raw = A.JOBS["retune"].get("mt3_raw")
    assert raw and raw[0]["pitch"] == 60, \
        "mt3_raw must survive, or refine cannot re-run without the model"


def test_running_job_is_not_persisted():
    A.JOBS["live"] = {"status": "running", "pct": 0.4, "created": time.time()}
    A._persist("live")
    _reload()
    assert "live" not in A.JOBS, \
        "a running job's worker thread does not survive, so restoring it strands the user"


def test_expired_job_is_not_restored():
    _done_job("stale")
    A._persist("stale")
    p = Path(A.WORK_DIR) / "stale.job.json"
    old = time.time() - A.RESULT_TTL - 60
    os.utime(p, (old, old))
    _reload()
    assert "stale" not in A.JOBS, "a job past its TTL came back from disk"


def test_numpy_values_do_not_block_the_write():
    try:
        import numpy as np
    except ImportError:
        return
    _done_job("npy", result={"notes": [], "tempo": np.float32(128.5),
                             "beats": np.array([0.0, 0.5]), "note_count": 0})
    A._persist("npy")
    _reload()
    assert "npy" in A.JOBS, "numpy scalars/arrays in a result must not lose the job"
    assert abs(A.JOBS["npy"]["result"]["tempo"] - 128.5) < 1e-3
    assert A.JOBS["npy"]["result"]["beats"] == [0.0, 0.5]


def test_keep_alive_extends_the_file_not_just_memory():
    _done_job("alive")
    A._persist("alive")
    p = Path(A.WORK_DIR) / "alive.job.json"
    old = time.time() - A.RESULT_TTL + 30      # nearly expired
    os.utime(p, (old, old))
    A._keep_alive("alive")
    _reload()
    assert "alive" in A.JOBS, \
        "_keep_alive must refresh the snapshot too, or an active session expires"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            for f in Path(A.WORK_DIR).glob("*.job.json"):
                f.unlink()
            A.JOBS.clear()
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if fails else 'all job-persistence tests passed'}")
    sys.exit(1 if fails else 0)
