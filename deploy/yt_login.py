"""
Headless-ish Google login -> export youtube.com cookies as Netscape cookies.txt.

Runs a HEADED Chromium under xvfb with a persistent profile so Google is less
likely to throw "this browser or app may not be secure". Best-effort: Google
often challenges datacenter IP logins with "verify it's you".

Usage:
    xvfb-run -a .venv/bin/python deploy/yt_login.py <email> <password> <out_cookies.txt>
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

EMAIL, PASSWORD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PROFILE = Path.home() / ".cache" / "yt-login-profile"
SHOTS = Path("/home/ubuntu/MusicNote/logs")
SHOTS.mkdir(exist_ok=True)
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/151.0.0.0 Safari/537.36")


def shot(page, name):
    try:
        page.screenshot(path=str(SHOTS / f"login_{name}.png"), full_page=True)
    except Exception as e:
        print(f"  (screenshot {name} failed: {e})")


def type_slow(page, selectors, text):
    if isinstance(selectors, str):
        selectors = [selectors]
    loc = None
    deadline = time.time() + 60
    while time.time() < deadline and loc is None:
        for sel in selectors:
            try:
                cand = page.locator(sel).first
                cand.wait_for(state="visible", timeout=3000)
                loc = cand
                break
            except Exception:
                continue
    if loc is None:
        raise RuntimeError(f"none of {selectors} became visible")
    loc.click()
    for ch in text:
        page.keyboard.type(ch)
        time.sleep(0.06 + 0.04 * (hash(ch) % 3))
    return loc


def netscape_from_cookies(cookies):
    lines = ["# Netscape HTTP Cookie File", ""]
    for c in cookies:
        dom = c["domain"]
        if "youtube.com" not in dom and "google.com" not in dom:
            continue
        flag = "TRUE" if dom.startswith(".") else "FALSE"
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = int(c.get("expires") or 0)
        if expiry <= 0:
            expiry = int(time.time()) + 180 * 86400
        lines.append("\t".join([dom, flag, c.get("path", "/"), secure,
                                str(expiry), c["name"], c["value"]]))
    return "\n".join(lines) + "\n"


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled",
                  "--disable-dev-shm-usage", "--window-size=1280,900"],
            user_agent=UA,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="Asia/Seoul",
        )
        page = ctx.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")

        print("1. opening accounts.google.com")
        page.goto("https://accounts.google.com/ServiceLogin?continue=https://www.youtube.com/",
                  wait_until="load", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        time.sleep(4)
        shot(page, "1_email")

        print("2. entering email")
        try:
            type_slow(page, ['#identifierId', 'input[type="email"]',
                             'input[name="identifier"]'], EMAIL)
            time.sleep(1)
            page.keyboard.press("Enter")
        except Exception as e:
            print(f"  email step: {e}")
        time.sleep(5)
        shot(page, "2_after_email")

        print("3. entering password")
        try:
            type_slow(page, ['input[name="Passwd"]',
                             'input[type="password"]:not([aria-hidden="true"])',
                             '#password input'], PASSWORD)
            time.sleep(1)
            page.keyboard.press("Enter")
        except Exception as e:
            print(f"  password step: {e}")
        time.sleep(6)
        shot(page, "3_after_password")

        # detect the "Verify it's you" device-prompt number and wait for the
        # user to approve on their phone/tablet.
        numfile = SHOTS / "challenge_number.txt"
        numfile.unlink(missing_ok=True)
        deadline = time.time() + 360
        logged_num = False
        import re as _re
        while time.time() < deadline:
            url = page.url
            signed_in = (
                (url.startswith("https://www.youtube.com/")
                 or url.startswith("https://myaccount.google.com/")
                 or url.startswith("https://accounts.google.com/ManageAccount"))
                and "signin" not in url and "challenge" not in url
            )
            if signed_in:
                print(f"   signed in, url={url}")
                break
            try:
                body = page.inner_text("body")
            except Exception:
                body = ""
            if not logged_num and ("Check your" in body or "tap " in body.lower()):
                m = _re.search(r"\btap\s+\*?\*?(\d{1,3})\b", body)
                num = m.group(1) if m else "?"
                dm = _re.search(r"Check your ([^\n.]+)", body)
                dev = dm.group(1).strip() if dm else ""
                numfile.write_text(f"{num}\t{dev}\n")
                print(f"   >>> DEVICE PROMPT: tap YES then {num} on '{dev}'")
                logged_num = True
                shot(page, "4_challenge")
            elif not logged_num and "another way" in body.lower():
                numfile.write_text("?\t(see logs/login_4_challenge.png)\n")
                print("   >>> challenge shown (not a number prompt)")
                logged_num = True
                shot(page, "4_challenge")
            time.sleep(4)
        shot(page, "4_final")
        print("   body text (first 400):")
        try:
            print("   " + page.inner_text("body")[:400].replace("\n", " | "))
        except Exception:
            pass

        print("5. visiting youtube.com to settle cookies")
        try:
            page.goto("https://www.youtube.com/", wait_until="load", timeout=60000)
            time.sleep(4)
        except Exception as e:
            print(f"  yt visit: {e}")
        shot(page, "5_youtube")

        cookies = ctx.cookies()
        names = sorted({c["name"] for c in cookies
                        if "youtube.com" in c["domain"] or "google.com" in c["domain"]})
        print(f"6. cookies captured: {names}")
        have_login = any(n in names for n in ("SID", "__Secure-1PSID", "__Secure-3PSID"))
        Path(OUT).write_text(netscape_from_cookies(cookies))
        print(f"   wrote {OUT}  (login cookies present: {have_login})")
        ctx.close()
        sys.exit(0 if have_login else 2)


if __name__ == "__main__":
    main()
