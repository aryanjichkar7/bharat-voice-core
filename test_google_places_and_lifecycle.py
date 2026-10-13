"""
Test suite for Google Places Integration, Local Nagpur Intelligence,
and Persistent Active Ride Lifecycle (RIDE_ACTIVE).
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import psutil
import websockets

import spatial_engine as se

PORT = 8771
WS_URL = f"ws://127.0.0.1:{PORT}/ws/voice"
SERVER_CMD = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)]

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name:<48} [{detail}]")


async def wait_for_server(timeout: float = 35.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
        except Exception:
            await asyncio.sleep(0.5)
    return None


async def run_tests():
    print(f"\n{'═' * 78}")
    print("  Bharat Online Cabs: Google Places & Active Ride Lifecycle Verification")
    print(f"{'═' * 78}\n")

    # 1. Direct Spatial Engine Test: Google Places API Integration
    print("📍 Testing Google Places API Engine on Nagpur coordinates...")
    jewelers = se.search_nearby_pois("any jewelry shop near me", se.DEFAULT_LAT, se.DEFAULT_LON)
    has_local_jeweler = len(jewelers) > 0 and jewelers[0]["distance_km"] < 2.0
    top_jeweler = jewelers[0] if jewelers else {}
    record(
        "Google Places: Local Shop Resolution",
        has_local_jeweler,
        f"{top_jeweler.get('name', 'None')} ({top_jeweler.get('short_address', '')}) @ {top_jeweler.get('distance_km', 'N/A')} km"
    )

    intent_res = se.evaluate_destination_intent("any jewelry shop near me", se.DEFAULT_LAT, se.DEFAULT_LON)
    is_disambig = intent_res.get("status") == "DISAMBIGUATION_NEEDED"
    clarif_prompt = intent_res.get("clarification_prompt", "")
    record("Concierge Disambiguation for Generic Category", is_disambig, f"{clarif_prompt[:50]}...")

    # 2. Launch server
    print("\n🚀 Starting test server on port", PORT, "...")
    proc = subprocess.Popen(
        SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    server_ps = psutil.Process(proc.pid)

    try:
        health = await wait_for_server()
        if not health:
            record("Server startup & healthcheck", False, "Timeout waiting for /health")
            return
        record("Server startup & healthcheck", True, f"Port {PORT}, Whisper {health.get('models', {}).get('whisper')}")

        # 3. Connect WebSocket
        async with websockets.connect(WS_URL) as ws:
            record("WebSocket Connection", True, "Connected to /ws/voice")

            # Check immediate status handshake
            init_frame = await asyncio.wait_for(ws.recv(), timeout=3.0)
            conn_ok = False
            if isinstance(init_frame, str):
                d = json.loads(init_frame)
                conn_ok = (d.get("type") == "CONNECTION_STATUS" and d.get("status") == "CONNECTED")
            record("Immediate CONNECTION_STATUS Handshake", conn_ok, "Status: CONNECTED")

            # CLIENT_INIT
            await ws.send(json.dumps({"type": "CLIENT_INIT", "lat": 21.1458, "lon": 79.0882, "lang": "en"}))

            # Receive greeting
            deadline = time.time() + 15.0
            while time.time() < deadline:
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    if isinstance(m, str) and json.loads(m).get("type") == "BOT_RESPONSE":
                        break
                except asyncio.TimeoutError:
                    break

            # 4. Search Category Query: "any jewelry shop near me"
            print("\n💍 Testing Category Query: 'any jewelry shop near me'...")
            await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "any jewelry shop near me"}))

            poi_frame = None
            disambig_bot_text = ""
            deadline = time.time() + 45.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "POI_CANDIDATES":
                            poi_frame = d
                        elif d.get("type") == "BOT_RESPONSE":
                            disambig_bot_text = d.get("text", "")
                        if poi_frame and disambig_bot_text:
                            break
                except asyncio.TimeoutError:
                    if poi_frame:
                        break

            cands = poi_frame.get("candidates", []) if poi_frame else []
            record("POI_CANDIDATES emitted from Google Places", len(cands) >= 1, f"Found {len(cands)} local shops")
            record("Concierge Voice Recommendation", len(disambig_bot_text) > 0, f"Spoken: {disambig_bot_text[:50]}...")

            # 5. Deterministic Selection
            chosen = cands[0] if cands else top_jeweler
            print(f"\n🎯 Selecting closest candidate: {chosen['name']}...")
            await ws.send(json.dumps({
                "type": "SELECT_POI",
                "lat": chosen["lat"],
                "lon": chosen["lon"],
                "name": chosen["name"]
            }))

            route_preview = None
            route_bot_text = ""
            deadline = time.time() + 30.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=6.0)
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "MAP_ROUTE_PREVIEW":
                            route_preview = d
                        elif d.get("type") == "BOT_RESPONSE":
                            route_bot_text = d.get("text", "")
                        if route_preview and route_bot_text:
                            break
                except asyncio.TimeoutError:
                    if route_preview:
                        break

            has_route = route_preview is not None and route_preview.get("fare_inr", 0) >= 70.0
            record("Deterministic MAP_ROUTE_PREVIEW", has_route, f"Fare: ₹{route_preview.get('fare_inr') if route_preview else 'N/A'}")

            # 6. Booking Confirmation & Active Ride Handshake
            print("\n🚕 Confirming booking...")
            await ws.send(json.dumps({"type": "CONFIRM_BOOKING"}))

            booking_confirmed = None
            booking_bot_text = ""
            deadline = time.time() + 25.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "BOOKING_CONFIRMED":
                            booking_confirmed = d
                        elif d.get("type") == "BOT_RESPONSE":
                            booking_bot_text = d.get("text", "")
                        if booking_confirmed and booking_bot_text:
                            break
                except asyncio.TimeoutError:
                    if booking_confirmed and booking_bot_text:
                        break

            bid = booking_confirmed.get("booking_id", "N/A") if booking_confirmed else "N/A"
            otp = booking_confirmed.get("otp", "N/A") if booking_confirmed else "N/A"
            driver = booking_confirmed.get("driver", {}) if booking_confirmed else {}
            record("BOOKING_CONFIRMED with Driver & OTP", booking_confirmed is not None and "name" in driver, f"ID: {bid}, OTP: {otp}, Driver: {driver.get('name')}")

            # 7. Post-Booking Lifecycle: Where is my driver?
            print("\n🚗 Testing Post-Booking Query: 'Where is my driver?'...")
            await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Where is my driver?"}))

            driver_bot_text = ""
            deadline = time.time() + 20.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "BOT_RESPONSE":
                            driver_bot_text = d.get("text", "")
                            break
                except asyncio.TimeoutError:
                    break

            has_driver_response = "rajesh" in driver_bot_text.lower() or "swift" in driver_bot_text.lower() or "minute" in driver_bot_text.lower()
            record("Post-Booking: Driver Location Tracking", has_driver_response, f"Spoken: {driver_bot_text[:50]}...")

            # 8. Post-Booking Lifecycle: What is my OTP?
            print("\n🔑 Testing Post-Booking Query: 'What is my OTP?'...")
            await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "What is my OTP?"}))

            otp_bot_text = ""
            deadline = time.time() + 20.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "BOT_RESPONSE":
                            otp_bot_text = d.get("text", "")
                            break
                except asyncio.TimeoutError:
                    break

            has_otp_response = "4821" in otp_bot_text or "otp" in otp_bot_text.lower()
            record("Post-Booking: Ride OTP Query", has_otp_response, f"Spoken: {otp_bot_text[:50]}...")

            # 9. Post-Booking Lifecycle: Ride Cancellation
            print("\n🛑 Testing Post-Booking: 'Cancel my ride'...")
            await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Cancel my ride"}))

            cancel_frame = None
            cancel_bot_text = ""
            deadline = time.time() + 20.0
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(msg, str):
                        d = json.loads(msg)
                        if d.get("type") == "RIDE_CANCELLED":
                            cancel_frame = d
                        elif d.get("type") == "BOT_RESPONSE":
                            cancel_bot_text = d.get("text", "")
                        if cancel_frame and cancel_bot_text:
                            break
                except asyncio.TimeoutError:
                    if cancel_frame:
                        break

            record("Post-Booking: RIDE_CANCELLED Handshake", cancel_frame is not None, f"Spoken: {cancel_bot_text[:50]}...")

    finally:
        print("\n🛑 Terminating test server...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run_tests())
    print(f"\n{'═' * 78}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL':<48} | {detail}")
    print(f"{'═' * 78}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'═' * 78}\n")
    sys.exit(0 if passed == total else 1)
