"""
Client for the MT3 transcription worker (backend/mt3_worker.py, pm2 app
`mt3-worker`, its own venv). Keeps the heavy mt3-infer / torch tree out of the
main MusicNote process.

`available()` — is the worker up?
`transcribe(wav_path, model=None, timeout=...)` — returns the worker's note list.
`map_family(program, is_drum)` — GM program -> MuseScore-ish instrument family.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

MT3_URL = os.environ.get("MUSICNOTE_MT3_URL", "http://127.0.0.1:8732").rstrip("/")
# unset -> let the worker's own MT3_MODEL choose
DEFAULT_MODEL = os.environ.get("MUSICNOTE_MT3_MODEL") or None
# MT3 on this CPU box is minutes/song; be generous.
DEFAULT_TIMEOUT = int(os.environ.get("MUSICNOTE_MT3_TIMEOUT", "2400"))

_health_cache = {"t": 0.0, "ok": False, "info": {}}


def _get(path: str, timeout: float = 4.0) -> dict:
    with urllib.request.urlopen(MT3_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def available(ttl: float = 15.0) -> bool:
    now = time.time()
    if now - _health_cache["t"] < ttl:
        return _health_cache["ok"]
    ok, info = False, {}
    try:
        info = _get("/health")
        ok = bool(info.get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    _health_cache.update(t=now, ok=ok, info=info)
    return ok


def health() -> dict:
    available()
    return _health_cache["info"]


def transcribe(wav_path: str, model: str | None = None,
               timeout: int = DEFAULT_TIMEOUT, shift: float = 0.0) -> dict:
    """POST to the worker. Returns {notes:[...], tracks:[...], model, seconds}.

    ``shift`` runs inference with that many seconds of silent lead-in; see
    ``mt3_worker.transcribe``. Times come back on the original timeline.
    """
    payload = {"wav_path": str(wav_path)}
    if shift:
        payload["shift"] = float(shift)
    m = model or DEFAULT_MODEL
    if m:
        payload["model"] = m
    body = json.dumps(payload).encode()
    req = urllib.request.Request(MT3_URL + "/transcribe", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:          # 500 -> read the JSON error body
        try:
            out = json.loads(e.read())
        except Exception:
            raise RuntimeError(f"MT3 worker HTTP {e.code}") from e
    except Exception as e:  # worker restart / socket close / local connection
        raise RuntimeError(
            "전체 악기 채보 워커가 재시작되었거나 응답하지 않습니다. 잠시 후 다시 시도하세요.") from e
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "MT3 worker error"))
    return out


# --- General-MIDI program -> instrument family (matches stems._STEM_FAMILY) --
_GM_FAMILY: list[tuple[int, str, str]] = [
    (7,   "keyboard", "피아노"),
    (15,  "keyboard", "건반 (크로마틱 타악)"),
    (23,  "keyboard", "오르간"),
    (31,  "plucked",  "기타"),
    (39,  "bass",     "베이스"),
    (47,  "strings",  "현악"),
    (55,  "strings",  "앙상블·현악"),
    (63,  "winds",    "금관"),
    (71,  "winds",    "목관 (리드)"),
    (79,  "winds",    "관악 (파이프)"),
    (103, "synth",    "신스"),
    (111, "plucked",  "민속 현악"),
    (119, "percussion", "타악"),
    (127, "other",    "효과음"),
]

# General MIDI Level 1 programs.  MT3 predicts these program numbers, so keep
# that concrete prediction in the product instead of collapsing it to a broad
# family such as "bass" or "winds".
_GM_PROGRAM_NAMES = [
    "어쿠스틱 그랜드 피아노", "브라이트 피아노", "일렉트릭 그랜드 피아노", "혼키통크 피아노",
    "일렉트릭 피아노 1", "일렉트릭 피아노 2", "하프시코드", "클라비넷",
    "첼레스타", "글로켄슈필", "뮤직 박스", "비브라폰", "마림바", "실로폰", "튜블러 벨", "덜시머",
    "드로바 오르간", "퍼커시브 오르간", "록 오르간", "처치 오르간", "리드 오르간", "아코디언", "하모니카", "탱고 아코디언",
    "나일론 기타", "스틸 기타", "재즈 기타", "클린 기타", "뮤트 기타", "오버드라이브 기타", "디스토션 기타", "기타 하모닉스",
    "어쿠스틱 베이스", "핑거 베이스", "픽 베이스", "프렛리스 베이스", "슬랩 베이스 1", "슬랩 베이스 2", "신스 베이스 1", "신스 베이스 2",
    "바이올린", "비올라", "첼로", "콘트라베이스", "트레몰로 현악", "피치카토 현악", "하프", "팀파니",
    "현악 앙상블 1", "현악 앙상블 2", "신스 현악 1", "신스 현악 2", "합창 아", "보이스 우", "신스 보이스", "오케스트라 히트",
    "트럼펫", "트롬본", "튜바", "뮤트 트럼펫", "프렌치 호른", "브라스 섹션", "신스 브라스 1", "신스 브라스 2",
    "소프라노 색소폰", "알토 색소폰", "테너 색소폰", "바리톤 색소폰", "오보에", "잉글리시 호른", "바순", "클라리넷",
    "피콜로", "플루트", "리코더", "팬 플루트", "블로운 보틀", "샤쿠하치", "휘슬", "오카리나",
    "리드 1 (스퀘어)", "리드 2 (쏘우)", "리드 3 (칼리오페)", "리드 4 (치프)", "리드 5 (차랑)", "리드 6 (보이스)", "리드 7 (피프스)", "리드 8 (베이스+리드)",
    "패드 1 (뉴에이지)", "패드 2 (웜)", "패드 3 (폴리신스)", "패드 4 (콰이어)", "패드 5 (보우드)", "패드 6 (메탈릭)", "패드 7 (헤일로)", "패드 8 (스윕)",
    "FX 1 (레인)", "FX 2 (사운드트랙)", "FX 3 (크리스털)", "FX 4 (애트모스피어)", "FX 5 (브라이트니스)", "FX 6 (고블린)", "FX 7 (에코즈)", "FX 8 (사이파이)",
    "시타르", "밴조", "샤미센", "코토", "칼림바", "백파이프", "피들", "샤나이",
    "팅클 벨", "아고고", "스틸 드럼", "우드블록", "타이코 드럼", "멜로딕 탐", "신스 드럼", "리버스 심벌",
    "기타 프렛 노이즈", "브레스 노이즈", "시쇼어", "버드 트윗", "텔레폰", "헬리콥터", "박수", "건샷",
]


def program_label(program: int, is_drum: bool) -> str:
    """Human label for MT3's exact General-MIDI prediction."""
    if is_drum:
        return "드럼 키트 (GM percussion)"
    if 0 <= int(program) < len(_GM_PROGRAM_NAMES):
        return f"{_GM_PROGRAM_NAMES[int(program)]} (GM {int(program) + 1})"
    return f"악기 미확정 (예측 프로그램 {int(program) + 1})"


def map_family(program: int, is_drum: bool) -> tuple[str, str]:
    """-> (family_key, korean_label)."""
    if is_drum:
        return "percussion", "타악 (드럼)"
    for hi, fam, label in _GM_FAMILY:
        if program <= hi:
            return fam, label
    return "other", "기타 악기"
