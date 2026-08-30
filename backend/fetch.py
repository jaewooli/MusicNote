"""
URL -> local audio file, via yt-dlp.

Only YouTube URLs are accepted (keeps this away from being an open SSRF proxy).
Downloads are capped by duration so a pasted link can't pull a multi-hour file.

YouTube blocks anonymous requests from datacenter IPs ("Sign in to confirm
you're not a bot"). To make URL input work on a server you must supply a
cookies file exported from a logged-in browser:

    MUSICNOTE_YT_COOKIES=/path/to/cookies.txt   (Netscape format)

If unset, the default is backend/cookies.txt when that file exists.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# pm2 runs this app as a "Node" child and exports NODE_CHANNEL_FD (its IPC
# socket) into the environment. yt-dlp spawns Deno to solve YouTube's JS "n"
# challenge; Deno inherits NODE_CHANNEL_FD, tries to open fd 3 as a Node IPC
# pipe, and dies ("fd is not from BiPipe") -> no formats -> "The page needs to
# be reloaded". uvicorn is Python, so these vars are vestigial here; drop them.
for _v in ("NODE_CHANNEL_FD", "NODE_CHANNEL_SERIALIZATION_MODE",
           "NODE_APP_INSTANCE", "NODE_UNIQUE_ID", "NODE_OPTIONS"):
    os.environ.pop(_v, None)

_DEBUG = os.environ.get("MUSICNOTE_YTDL_DEBUG") == "1"

_YT_RE = re.compile(
    r"^https?://(?:www\.|m\.|music\.)?(?:youtube\.com/(?:watch\?|shorts/|live/|embed/)|youtu\.be/)",
    re.IGNORECASE,
)

MAX_DURATION = int(os.environ.get("MUSICNOTE_MAX_DURATION", "1200"))  # seconds

_DEFAULT_COOKIES = Path(__file__).resolve().parent / "cookies.txt"


class TooLong(Exception):
    pass


class NeedsCookies(Exception):
    pass


def _cookiefile() -> str | None:
    env = os.environ.get("MUSICNOTE_YT_COOKIES", "").strip()
    if env and Path(env).is_file():
        return env
    if _DEFAULT_COOKIES.is_file():
        return str(_DEFAULT_COOKIES)
    return None


def is_supported_url(url: str) -> bool:
    return bool(_YT_RE.match((url or "").strip()))


def cookies_available() -> bool:
    return _cookiefile() is not None


POT_BASEURL = os.environ.get("MUSICNOTE_POT_BASEURL", "http://127.0.0.1:4416")


def pot_server_up() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(POT_BASEURL.rstrip("/") + "/ping", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def download_audio(url: str, dest_dir: Path,
                   max_duration: int = MAX_DURATION,
                   progress_hook=None) -> tuple[Path, str, float]:
    """Return (wav_path, title, duration_seconds).

    ``progress_hook`` (optional) is a yt-dlp progress hook: it is called with a
    dict carrying ``status`` ("downloading"/"finished") and byte counters.

    Raises TooLong / NeedsCookies / RuntimeError.
    """
    import yt_dlp

    url = url.strip()
    out_tmpl = str(dest_dir / "yt_%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": not _DEBUG,
        "no_warnings": not _DEBUG,
        "verbose": _DEBUG,
        "restrictfilenames": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        # Let yt-dlp pick its default player clients; forcing a list here
        # (esp. tv / web_safari) breaks on the current YouTube backend
        # ("The page needs to be reloaded"). The JS "n" challenge is solved
        # locally by yt-dlp-ejs + Deno, and GVS PO tokens come from the
        # bgutil HTTP provider below.
        "extractor_args": {
            "youtubepot-bgutilhttp": {"base_url": [POT_BASEURL]},
        },
    }
    cf = _cookiefile()
    if cf:
        ydl_opts["cookiefile"] = cf
    if progress_hook is not None:
        def _safe_hook(d):
            try:
                progress_hook(d)
            except Exception:
                pass
        ydl_opts["progress_hooks"] = [_safe_hook]
        ydl_opts["postprocessor_hooks"] = [_safe_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise RuntimeError("영상 정보를 가져올 수 없습니다.")
            dur = float(info.get("duration") or 0)
            if dur and dur > max_duration:
                raise TooLong(
                    f"영상이 너무 깁니다 ({int(dur // 60)}분). "
                    f"최대 {max_duration // 60}분까지 지원합니다.")
            info = ydl.extract_info(url, download=True)
            base = os.path.splitext(ydl.prepare_filename(info))[0]
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "confirm you" in msg or "Sign in" in msg or "cookies" in msg.lower():
            raise NeedsCookies(
                "YouTube가 이 서버 IP를 봇으로 차단했습니다. PO-token provider·Deno 는 "
                "설치되어 있지만, 현재 YouTube는 로그인 쿠키 없이는 통과되지 않습니다. "
                "전용 계정으로 내보낸 cookies.txt 를 backend/cookies.txt 에 두거나 "
                "MUSICNOTE_YT_COOKIES 로 지정한 뒤 pm2 restart 하세요. "
                "(파일 업로드는 쿠키 없이도 정상 동작합니다.)")
        raise RuntimeError(msg.splitlines()[-1]) from e

    wav_path = Path(base + ".wav")
    if not wav_path.exists():
        cands = list(dest_dir.glob(f"yt_{info.get('id', '')}*"))
        if not cands:
            raise RuntimeError("다운로드된 오디오 파일을 찾을 수 없습니다.")
        wav_path = cands[0]

    title = info.get("title") or info.get("id") or "youtube"
    return wav_path, title, float(info.get("duration") or 0)
