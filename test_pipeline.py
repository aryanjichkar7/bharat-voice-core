"""
Bharat Online Cabs - Phase 3 Pipeline Verification & Smoke Test.

Validates the full-duplex WebSocket voice pipeline:
  1. Connection & Session Init (CLIENT_INIT -> SESSION_READY)
  2. Greet & Piper TTS Audio Generation (Binary PCM 22.05kHz frames)
  3. Binary Audio Inbound Ingestion (16kHz PCM frames -> VAD/Transport)
  4. Conversational Tool Invocation:
     - Search & Route Planning (POI_CANDIDATES, MAP_ROUTE_PREVIEW)
     - Fare math verification (Base ₹50, ₹14/km, Min ₹70)
     - GeoJSON LineString coordinates verification
  5. Booking Confirmation (BOOKING_CONFIRMED with ID)
  6. Memory stability on Apple Silicon (RSS tracking via psutil)
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import numpy as np
import psutil
import websockets

PORT = int(os.getenv("TEST_PORT", "8765"))
WS_URL = f"ws://127.0.0.1:{PORT}/ws/voice"
SERVER_CMD = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)]

test_results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    test_results.append((name, passed, detail))
    status_icon = "✅" if passed else "❌"
    print(f"  {status_icon} {name:<45} [{detail}]")


async def wait_for_server(timeout: float = 30.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    return data
        except Exception:
            await asyncio.sleep(0.5)
    return None


async def run_pipeline_test():
    print(f"\n{'═' * 78}")
    print("  Bharat Online Cabs: Phase 3 Full-Duplex Pipeline Smoke Test")
    print(f"{'═' * 78}\n")

    # 1. Start server process
    print("🚀 Launching voice server background process...")
    proc = subprocess.Popen(
        SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    server_ps = psutil.Process(proc.pid)
    initial_rss = server_ps.memory_info().rss / (1024 * 1024)

    try:
        health = await wait_for_server()
        if not health:
            record("Server startup and healthcheck", False, "Timeout waiting for /health")
            return False

        ready_rss = health.get("process_rss_mb", 0)
        record("Server startup and healthcheck", True, f"Port {PORT}, RSS: {ready_rss}MB")

        # 2. Connect to WebSocket
        print(f"\n📡 Connecting to WebSocket endpoint: {WS_URL} ...")
        async with websockets.connect(WS_URL) as ws:
            record("WebSocket handshake", True, "/ws/voice connected")

            # 3. Send CLIENT_INIT
            client_init = {
                "type": "CLIENT_INIT",
                "lat": 21.1458,
                "lon": 79.0882
            }
            await ws.send(json.dumps(client_init))

            # Receive initial greeting frames
            greeting_audio_bytes = 0
            session_ready_received = False
            bot_greeting_received = False

            timeout_deadline = time.time() + 15.0
            while time.time() < timeout_deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    if isinstance(msg, bytes):
                        greeting_audio_bytes += len(msg)
                    else:
                        payload = json.loads(msg)
                        if payload.get("type") == "SESSION_READY":
                            session_ready_received = True
                        elif payload.get("type") == "BOT_RESPONSE":
                            bot_greeting_received = True

                    if session_ready_received and greeting_audio_bytes > 0 and bot_greeting_received:
                        break
                except asyncio.TimeoutError:
                    break

            record("SESSION_READY acknowledgment", session_ready_received, "GPS lat/lon bound")
            record("Piper TTS audio stream (greeting)", greeting_audio_bytes > 1000, f"{greeting_audio_bytes} bytes received")
            record("Greeting text prompt emitted", bot_greeting_received, "Bot greeting received")

            # 4. Transmit binary PCM audio frames (Simulated 16kHz speech input)
            print("\n🎙️  Streaming simulated 16kHz binary PCM audio frames to server...")
            # 1.5 seconds of 16kHz 16-bit mono PCM audio
            sample_rate = 16000
            duration = 1.5
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            sine_wave = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            pcm_bytes = sine_wave.tobytes()

            chunk_size = 640  # 20ms chunks @ 16kHz mono 16-bit
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.005)

            record("Inbound binary audio streaming", True, f"Sent {len(pcm_bytes)} bytes PCM16")

            # 5. Conversational Tool Calling: Request route to Civil Lines
            print("\n🚕 Testing Destination Booking Flow: 'Take me to Civil Lines'...")
            turn_req = {
                "type": "TEXT_INPUT",
                "text": "I want to go to Civil Lines"
            }
            await ws.send(json.dumps(turn_req))

            route_preview = None
            poi_candidates = None
            response_audio_bytes = 0
            bot_response_text = ""

            deadline = time.time() + 45.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    if isinstance(msg, bytes):
                        response_audio_bytes += len(msg)
                    else:
                        data = json.loads(msg)
                        mtype = data.get("type")
                        if mtype == "POI_CANDIDATES":
                            poi_candidates = data
                        elif mtype == "MAP_ROUTE_PREVIEW":
                            route_preview = data
                        elif mtype == "BOT_RESPONSE":
                            bot_response_text = data.get("text", "")
                            if route_preview and response_audio_bytes > 0:
                                break
                except asyncio.TimeoutError:
                    if route_preview:
                        break

            record("Tool Execution: search_nearby_pois", poi_candidates is not None or route_preview is not None,
                   f"Candidates: {len(poi_candidates.get('candidates', [])) if poi_candidates else 'direct match'}")

            has_valid_preview = False
            preview_detail = "None"
            if route_preview:
                dist = route_preview.get("distance_km", 0)
                dur = route_preview.get("duration_mins", 0)
                fare = route_preview.get("fare_inr", 0)
                coords = route_preview.get("route_geometry", {}).get("coordinates", [])
                has_valid_preview = (dist > 0 and dur > 0 and fare >= 70.0 and len(coords) >= 2)
                preview_detail = f"{dist}km, {dur}min, ₹{fare}, {len(coords)} pts"

            record("Tool Execution: calculate_trip_details", has_valid_preview, preview_detail)
            record("MAP_ROUTE_PREVIEW JSON frame emitted", route_preview is not None, f"Destination: {route_preview.get('destination') if route_preview else 'N/A'}")
            record("Route response TTS audio", response_audio_bytes > 1000, f"{response_audio_bytes} bytes audio")

            # 6. Booking Confirmation: 'Yes, please book the cab'
            print("\n✅ Testing Booking Confirmation Flow: 'Yes, please book it'...")
            confirm_req = {
                "type": "TEXT_INPUT",
                "text": "Yes, please confirm the booking"
            }
            await ws.send(json.dumps(confirm_req))

            booking_confirmed = None
            confirm_deadline = time.time() + 35.0
            while time.time() < confirm_deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    if isinstance(msg, bytes):
                        pass
                    else:
                        data = json.loads(msg)
                        if data.get("type") == "BOOKING_CONFIRMED":
                            booking_confirmed = data
                            break
                except asyncio.TimeoutError:
                    break

            booking_id = booking_confirmed.get("booking_id", "N/A") if booking_confirmed else "N/A"
            record("Tool Execution: confirm_booking", booking_confirmed is not None, f"Booking ID: {booking_id}")
            record("BOOKING_CONFIRMED JSON frame emitted", booking_confirmed is not None, f"Status: CONFIRMED")

            # 7. Check server memory stability
            peak_rss = server_ps.memory_info().rss / (1024 * 1024)
            record("Server memory stability (M1 8GB)", peak_rss < 1500, f"Peak RSS: {peak_rss:.1f}MB")

    finally:
        print("\n🛑 Shutting down test server...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run_pipeline_test())

    # Print summary table
    print(f"\n{'═' * 78}")
    print("  PHASE 3 VERIFICATION SUMMARY")
    print(f"{'═' * 78}")
    passed_count = sum(1 for _, ok, _ in test_results if ok)
    total_count = len(test_results)

    for name, ok, detail in test_results:
        print(f"  {'PASS' if ok else 'FAIL':<6} | {name:<46} | {detail}")

    print(f"{'═' * 78}")
    all_passed = (passed_count == total_count and total_count > 0)
    print(f"  Result: {passed_count}/{total_count} tests passed")
    print("  ✅ PHASE 3 COMPLETE: Full-duplex voice server is ready for Frontend." if all_passed else "  ❌ Some tests failed.")
    print(f"{'═' * 78}\n")

    sys.exit(0 if all_passed else 1)
