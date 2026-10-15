"""
Bharat Online Cabs - Phase 4 Web & Frontend Integration Verification.

Verifies:
  1. Static asset mounting in FastAPI (/ -> index.html, /static/style.css, /static/app.js)
  2. HTTP 200 OK responses and proper content types
  3. Live WebSocket upgrade and handshake on /ws/voice
  4. Memory stability on Apple Silicon M1
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
import psutil
import websockets
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PORT = int(os.getenv("WEB_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/ws/voice"
SERVER_CMD = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)]

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    status_icon = "✅" if passed else "❌"
    print(f"  {status_icon} {name:<46} [{detail}]")


async def wait_for_server(timeout: float = 30.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    return data
        except Exception:
            await asyncio.sleep(0.5)
    return None


def fetch_url(path: str) -> tuple[int, str, bytes]:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        status = resp.status
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()
        return status, content_type, body


async def run_verification():
    print(f"\n{'═' * 78}")
    print("  Bharat Online Cabs: Phase 4 Frontend & Web Verification")
    print(f"{'═' * 78}\n")

    print(f"🚀 Starting voice server process on port {PORT}...")
    proc = subprocess.Popen(
        SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        cwd=str(REPO_ROOT),
    )

    server_ps = psutil.Process(proc.pid)

    try:
        health = await wait_for_server()
        if not health:
            record("Server startup and /health probe", False, "Timeout waiting for /health")
            return False

        rss_mb = health.get("process_rss_mb", 0)
        record("Server startup and /health probe", True, f"Port {PORT}, RSS: {rss_mb} MB")

        # 1. Verify Root Route (index.html)
        status, ctype, body = fetch_url("/")
        body_text = body.decode("utf-8", errors="ignore")
        has_title = "Bharat Online Cabs" in body_text
        has_orb = 'id="voiceOrb"' in body_text
        has_map = 'id="map"' in body_text

        record(
            "HTTP GET / (index.html)",
            status == 200 and "text/html" in ctype and has_title and has_orb and has_map,
            f"{status} OK, {len(body)} bytes, HTML verified"
        )

        # 2. Verify /static/style.css
        status, ctype, body = fetch_url("/static/style.css")
        body_text = body.decode("utf-8", errors="ignore")
        has_css = "voice-orb" in body_text and "--accent-cyan" in body_text

        record(
            "HTTP GET /static/style.css",
            status == 200 and "text/css" in ctype and has_css,
            f"{status} OK, {len(body)} bytes, CSS verified"
        )

        # 3. Verify /static/app.js
        status, ctype, body = fetch_url("/static/app.js")
        body_text = body.decode("utf-8", errors="ignore")
        has_js = "downsampleBuffer" in body_text and "handleServerAudioBuffer" in body_text

        record(
            "HTTP GET /static/app.js",
            status == 200 and ("javascript" in ctype or "text/" in ctype) and has_js,
            f"{status} OK, {len(body)} bytes, JS verified"
        )

        # 4. Verify WebSocket Upgrade & Handshake
        print(f"\n📡 Verifying WebSocket Upgrade at {WS_URL} ...")
        async with websockets.connect(WS_URL) as ws:
            record("WebSocket /ws/voice upgrade", True, "101 Switching Protocols")

            # Send CLIENT_INIT
            await ws.send(json.dumps({
                "type": "CLIENT_INIT",
                "lat": 21.1458,
                "lon": 79.0882
            }))

            session_ready = False
            got_greeting_audio = False

            deadline = time.time() + 10.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.5)
                    if isinstance(msg, bytes):
                        got_greeting_audio = True
                    else:
                        payload = json.loads(msg)
                        if payload.get("type") == "SESSION_READY":
                            session_ready = True
                    if session_ready and got_greeting_audio:
                        break
                except asyncio.TimeoutError:
                    break

            record("WebSocket CLIENT_INIT -> SESSION_READY", session_ready, "Session initialized")
            record("Piper TTS audio stream handshake", got_greeting_audio, "Initial speech frames received")

        # 5. Check Memory Footprint
        peak_rss = server_ps.memory_info().rss / (1024 * 1024)
        record("Server memory footprint (M1 8GB)", peak_rss < 1500, f"Process RSS: {peak_rss:.1f} MB")

    finally:
        print("\n🛑 Terminating verification server process...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run_verification())

    print(f"\n{'═' * 78}")
    print("  PHASE 4 FRONTEND VERIFICATION SUMMARY")
    print(f"{'═' * 78}")
    passed_count = sum(1 for _, ok, _ in results if ok)
    total_count = len(results)

    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL':<6} | {name:<46} | {detail}")

    print(f"{'═' * 78}")
    all_passed = (passed_count == total_count and total_count > 0)
    print(f"  Result: {passed_count}/{total_count} tests passed")
    print("  ✅ PHASE 4 COMPLETE: Standalone Siri-style frontend verified and ready!" if all_passed else "  ❌ Some tests failed.")
    print(f"{'═' * 78}\n")

    sys.exit(0 if all_passed else 1)
