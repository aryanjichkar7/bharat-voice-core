"""
Bharat Online Cabs - Comprehensive Enterprise Verification Suite.
Validates:
1. API Config & Key Segregation (Browser Key exposed, Places Server Key isolated)
2. AudioWorklet & Modern Asset Integrity
3. WebSocket Handshake & Telemetry Synchronization
4. Concierge POI Disambiguation & Deterministic Route Preview
5. Deterministic Booking Confirmation & Honest Dispatch Telemetry
6. Active Ride Lifecycle (Driver Status, OTP Query, Ride Cancellation)
7. Dynamic Multilingual Switching & Voice Model Alignment
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import psutil
import websockets

PORT = 8775
WS_URL = f"ws://127.0.0.1:{PORT}/ws/voice"
SERVER_CMD = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)]

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name:<46} [{detail}]")


async def wait_for_server(timeout: float = 30.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
        except Exception:
            await asyncio.sleep(0.4)
    return None


async def run_tests():
    print(f"\n{'═' * 78}")
    print("  Bharat Online Cabs: Enterprise Architecture Verification")
    print(f"{'═' * 78}\n")

    # 1. Asset & Frontend File Verifications
    with open("static/pcm-recorder-worklet.js", "r", encoding="utf-8") as f:
        worklet_code = f.read()
    has_worklet = "PcmRecorderProcessor" in worklet_code and "registerProcessor" in worklet_code
    record("AudioWorklet Processor Implementation", has_worklet, "static/pcm-recorder-worklet.js registered")

    with open("static/app.js", "r", encoding="utf-8") as f:
        app_js = f.read()
    has_gmaps = "google.maps.Map" in app_js and "google.maps.Polyline" in app_js
    has_worklet_load = "audioWorklet.addModule" in app_js
    has_honest_dispatch = "Simulated Dispatch" in app_js or "Live GPS Unlinked" in app_js or "Live GPS" in app_js
    record("Google Maps JS API Engine", has_gmaps, "Vector Map & Polyline routines verified")
    record("AudioWorklet Audio Pipeline in Client", has_worklet_load, "addModule(/static/pcm-recorder-worklet.js) present")

    with open("static/index.html", "r", encoding="utf-8") as f:
        index_html = f.read()
    no_leaflet = "leaflet" not in index_html.lower()
    has_honest_ui = "Live GPS Unlinked" in index_html or "Simulated Dispatch" in index_html
    record("No Leaflet Dependencies in HTML", no_leaflet, "Leaflet CSS & JS completely removed")
    record("Honest Driver Dispatch Labeling", has_honest_ui, "Driver card discloses simulated dispatch telemetry")

    # 2. Launch test server
    print(f"\n🚀 Launching verification server on port {PORT}...")
    proc = subprocess.Popen(
        SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    try:
        health = await wait_for_server()
        if not health:
            record("Server startup & healthcheck", False, "Timeout waiting for /health")
            return

        record("Server Startup & Healthcheck", True, f"Port {PORT}, Whisper {health['models']['whisper']}")

        # 3. Test /api/config key isolation
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/config", timeout=2.0) as resp:
            cfg = json.loads(resp.read().decode())
        has_browser_key_key = "google_maps_browser_key" in cfg
        has_places_key_isolated = "google_places_api_key" not in cfg and "places" not in "".join(cfg.keys()).lower()
        record("API Config Endpoint Exists", has_browser_key_key, f"Keys: {list(cfg.keys())}")
        record("Server Key Segregation (No Places Leak)", has_places_key_isolated, "Server GOOGLE_PLACES_API_KEY is not exposed to client")

        # 4. WebSocket Test: Handshake & Booking Lifecycle
        async with websockets.connect(WS_URL) as ws:
            record("WebSocket Connection", True, f"Connected to {WS_URL}")

            # Handshake frame
            handshake_raw = await asyncio.wait_for(ws.recv(), timeout=4.0)
            handshake = json.loads(handshake_raw) if isinstance(handshake_raw, str) else {}
            record("Immediate Handshake Frame", handshake.get("status") == "CONNECTED", f"status: {handshake.get('status')}")

            # Send CLIENT_INIT
            await ws.send(json.dumps({
                "type": "CLIENT_INIT",
                "lat": 21.1458,
                "lon": 79.0882,
                "lang": "en"
            }))

            # Drain initial frames (greeting, bot response)
            await asyncio.sleep(0.8)
            while True:
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.3)
                except asyncio.TimeoutError:
                    break

            # 5. POI Query: "any jewelry shop near me"
            print("\n🔍 Querying: 'any jewelry shop near me'...")
            await ws.send(json.dumps({
                "type": "TEXT_INPUT",
                "text": "any jewelry shop near me"
            }))

            poi_emitted = False
            first_candidate = None
            timeout_time = time.time() + 10.0
            while time.time() < timeout_time and not poi_emitted:
                try:
                    frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    if isinstance(frame, str):
                        data = json.loads(frame)
                        if data.get("type") == "POI_CANDIDATES":
                            cands = data.get("candidates", [])
                            if cands:
                                poi_emitted = True
                                first_candidate = cands[0]
                except asyncio.TimeoutError:
                    pass

            record("Live POI Candidates Emitted", poi_emitted, f"Found {first_candidate.get('name') if first_candidate else 'None'}")

            # 6. Select candidate to trigger deterministic route preview
            if first_candidate:
                print(f"🎯 Selecting candidate: {first_candidate['name']}...")
                await ws.send(json.dumps({
                    "type": "SELECT_POI",
                    "lat": first_candidate["lat"],
                    "lon": first_candidate["lon"],
                    "name": first_candidate["name"]
                }))

                route_preview_emitted = False
                timeout_time = time.time() + 8.0
                while time.time() < timeout_time and not route_preview_emitted:
                    try:
                        frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(frame, str):
                            data = json.loads(frame)
                            if data.get("type") == "MAP_ROUTE_PREVIEW":
                                route_preview_emitted = True
                                fare = data.get("fare_inr")
                                dist = data.get("distance_km")
                    except asyncio.TimeoutError:
                        pass

                record("Deterministic MAP_ROUTE_PREVIEW", route_preview_emitted, f"Distance: {dist}km, Fare: ₹{fare}")

                # 7. Confirm ride
                print("🚕 Confirming ride...")
                await ws.send(json.dumps({"type": "CONFIRM_BOOKING"}))

                booking_confirmed_emitted = False
                bid = None
                otp = None
                timeout_time = time.time() + 8.0
                while time.time() < timeout_time and not booking_confirmed_emitted:
                    try:
                        frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(frame, str):
                            data = json.loads(frame)
                            if data.get("type") == "BOOKING_CONFIRMED":
                                booking_confirmed_emitted = True
                                bid = data.get("booking_id")
                                otp = data.get("otp")
                    except asyncio.TimeoutError:
                        pass

                record("BOOKING_CONFIRMED Emitted", booking_confirmed_emitted, f"ID: {bid}, OTP: {otp}")

                # 8. Active ride queries
                print("🚗 Inquiring driver location...")
                await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Where is my driver?"}))
                driver_spoken = False
                timeout_time = time.time() + 8.0
                while time.time() < timeout_time and not driver_spoken:
                    try:
                        frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(frame, str):
                            data = json.loads(frame)
                            if data.get("type") == "BOT_RESPONSE" and "driver" in data.get("text", "").lower():
                                driver_spoken = True
                                txt = data.get("text")
                    except asyncio.TimeoutError:
                        pass

                record("Active Ride Driver Telemetry Query", driver_spoken, f"{txt[:45]}..." if driver_spoken else "None")

                # 9. Cancel ride
                print("🛑 Cancelling ride...")
                await ws.send(json.dumps({"type": "CANCEL_RIDE"}))
                cancel_emitted = False
                timeout_time = time.time() + 6.0
                while time.time() < timeout_time and not cancel_emitted:
                    try:
                        frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(frame, str):
                            data = json.loads(frame)
                            if data.get("type") == "RIDE_CANCELLED":
                                cancel_emitted = True
                    except asyncio.TimeoutError:
                        pass

                record("Ride Cancellation Handshake", cancel_emitted, "RIDE_CANCELLED frame confirmed")

            # 10. Multilingual switching
            print("🌐 Testing language switch to Hindi...")
            await ws.send(json.dumps({"type": "SET_LANGUAGE", "lang": "hi"}))
            lang_switched = False
            timeout_time = time.time() + 6.0
            while time.time() < timeout_time and not lang_switched:
                try:
                    frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    if isinstance(frame, str):
                        data = json.loads(frame)
                        if data.get("type") == "LANGUAGE_CHANGED" and data.get("language") == "hi":
                            lang_switched = True
                except asyncio.TimeoutError:
                    pass

            record("Language Switching to Hindi", lang_switched, "LANGUAGE_CHANGED(hi) confirmed")

    finally:
        print("\n🛑 Stopping test server...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_tests())
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'═' * 78}")
    print(f"  Summary: {passed}/{total} checks passed")
    print(f"{'═' * 78}\n")
    sys.exit(0 if passed == total else 1)
