"""
Phase 3 & Phase 2 Rigorous Reliability Verification Suite for Bharat Online Cabs.
Exercises all 10 explicit Phase 3 booking & state scenarios plus Phase 2 lifecycle
against the running backend (ws://127.0.0.1:8000/ws/voice).
"""
import asyncio
import json
import time
import websockets
from loguru import logger
import spatial_engine as se

WS_URL = "ws://127.0.0.1:8000/ws/voice?lat=21.1458&lon=79.0882&lang=en"

results = []

def record(scenario_num: str, name: str, passed: bool, detail: str = ""):
    results.append((scenario_num, name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{scenario_num}] {name:<48} [{detail}]")
    assert passed, f"Failed [{scenario_num}] {name}: {detail}"

async def drain_frames_until(ws, target_type: str, timeout: float = 12.0):
    start = time.time()
    frames = []
    while time.time() - start < timeout:
        try:
            f = await asyncio.wait_for(ws.recv(), timeout=1.5)
            if isinstance(f, str):
                d = json.loads(f)
                frames.append(d)
                if d.get("type") == target_type:
                    return d, frames
        except asyncio.TimeoutError:
            continue
    return None, frames

async def test_all_scenarios():
    print(f"\n{'═' * 80}")
    print("  BHARAT ONLINE CABS: PHASE 3 EVIDENCE-DRIVEN RUNTIME AUDIT (10 SCENARIOS)")
    print(f"{'═' * 80}\n")

    # -------------------------------------------------------------------------
    # SCENARIO 1: POI Deduplication & Multiple Outlets (Haldiram / Batukbhai)
    # -------------------------------------------------------------------------
    print("--- Scenario 1: POI Deduplication & Disambiguation ---")
    candidates = se.search_nearby_pois("Haldiram", 21.1458, 79.0882, radius_km=15.0)
    record("1.1", "POI search returns candidates", len(candidates) >= 2, f"Found {len(candidates)} branches")
    # Verify deduplication: no two candidates within 200m with same name
    dedup_clean = True
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            dist = se.calculate_distance(c1["lat"], c1["lon"], c2["lat"], c2["lon"])
            if dist < 0.2 and c1["name"].lower() == c2["name"].lower():
                dedup_clean = False
    record("1.2", "No duplicate pins within 200m", dedup_clean, "Spatial deduplication verified")

    eval_res = se.evaluate_destination_intent("Haldiram", 21.1458, 79.0882)
    record("1.3", "Intent flags DISAMBIGUATION_NEEDED", eval_res.get("status") == "DISAMBIGUATION_NEEDED", 
           f"Outlets: {eval_res.get('outlet_count')}")

    # -------------------------------------------------------------------------
    # SCENARIO 2: Quote Generation (Fare, Distance, ETA, OSRM Geometry)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: Deterministic Fare Quote Calculation ---")
    trip = se.calculate_trip_details(21.1458, 79.0882, 21.1549, 79.0789, "Civil Lines")
    expected_fare = round(max(50.0, 50.0 + 14.0 * trip["distance_km"]), 2)
    record("2.1", "Fare calculation tariff match", trip["fare_inr"] == expected_fare, 
           f"Distance: {trip['distance_km']}km, Fare: ₹{trip['fare_inr']}")
    record("2.2", "Route geometry is GeoJSON LineString", 
           trip["route_geometry"]["type"] == "LineString" and len(trip["route_geometry"]["coordinates"]) > 2,
           f"{len(trip['route_geometry']['coordinates'])} coordinates")

    # -------------------------------------------------------------------------
    # SCENARIO 3: Confirm Booking Flow (Proposal -> Confirm -> Active Ride)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 3: End-to-End Booking Confirmation ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE") # Greeting
        
        # Request destination
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Take me to Civil Lines"}))
        route_f, _ = await drain_frames_until(ws, "MAP_ROUTE_PREVIEW")
        record("3.1", "Route preview received", route_f is not None, f"Destination: {route_f.get('destination')}")

        # Confirm booking
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Yes, Book it"}))
        conf_f, _ = await drain_frames_until(ws, "BOOKING_CONFIRMED")
        record("3.2", "Booking confirmed frame received", conf_f is not None, 
               f"BID: {conf_f.get('booking_id')}, OTP: {conf_f.get('otp')}")
        record("3.3", "Simulated dispatch disclosure present", "SIMULATION" in conf_f.get("dispatch_mode", ""),
               "Honest simulation flag verified")

    # -------------------------------------------------------------------------
    # SCENARIO 4: "Cancel my booking" on Active Ride
    # -------------------------------------------------------------------------
    print("\n--- Scenario 4: Active Ride Cancellation ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Take me to Civil Lines"}))
        await drain_frames_until(ws, "MAP_ROUTE_PREVIEW")
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Yes, please confirm"}))
        await drain_frames_until(ws, "BOOKING_CONFIRMED")

        # Now cancel active ride
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Cancel my booking"}))
        canc_f, _ = await drain_frames_until(ws, "RIDE_CANCELLED")
        record("4.1", "Active ride cancelled frame received", canc_f is not None, 
               f"Reason: {canc_f.get('reason') if canc_f else 'None'}")
        canc_bot, _ = await drain_frames_until(ws, "BOT_RESPONSE")
        bot_reply = canc_bot.get("text", "") if canc_bot else ""
        record("4.2", "Spoken cancellation prompt", "cancelled" in bot_reply.lower(), bot_reply[:45])

    # -------------------------------------------------------------------------
    # SCENARIO 5: "Don't book it" / "Cancel" during Proposal (Pre-Confirmation)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 5: Cancel Fare Proposal Before Confirmation ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Take me to Civil Lines"}))
        await drain_frames_until(ws, "MAP_ROUTE_PREVIEW")

        # Reject proposal
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Don't book it"}))
        canc_f, _ = await drain_frames_until(ws, "RIDE_CANCELLED")
        record("5.1", "Proposal cancellation frame received", canc_f is not None and canc_f.get("reason") == "PROPOSAL_CANCELLED",
               f"Reason: {canc_f.get('reason') if canc_f else 'None'}")
        canc_bot, _ = await drain_frames_until(ws, "BOT_RESPONSE")
        bot_reply = canc_bot.get("text", "") if canc_bot else ""
        record("5.2", "Spoken proposal cancellation prompt", "proposal cancelled" in bot_reply.lower(), bot_reply[:45])

    # -------------------------------------------------------------------------
    # SCENARIO 6: "Yes, book it" when NO Proposal Exists
    # -------------------------------------------------------------------------
    print("\n--- Scenario 6: 'Yes, book it' Without Proposal ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")

        # Customer says "Yes, book it" before picking a destination
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Yes, book it"}))
        bot_f, frames = await drain_frames_until(ws, "BOT_RESPONSE")
        has_conf = any(f.get("type") == "BOOKING_CONFIRMED" for f in frames)
        record("6.1", "No fake booking confirmation emitted", not has_conf, "Zero false BOOKING_CONFIRMED")
        record("6.2", "Spoken prompt instructs destination selection", 
               "haven't selected a destination" in bot_f.get("text", "").lower(), bot_f.get("text", "")[:50])

    # -------------------------------------------------------------------------
    # SCENARIO 7: "Cancel my booking" when NO Active Ride Exists
    # -------------------------------------------------------------------------
    print("\n--- Scenario 7: 'Cancel my booking' Without Active Ride ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")

        # Customer asks to cancel right at greeting
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Cancel my booking"}))
        bot_f, frames = await drain_frames_until(ws, "BOT_RESPONSE")
        record("7.1", "Informs user no active ride to cancel", 
               "don't have an active ride" in bot_f.get("text", "").lower(), bot_f.get("text", "")[:50])

    # -------------------------------------------------------------------------
    # SCENARIO 8: Destination Change After Fare Proposal
    # -------------------------------------------------------------------------
    print("\n--- Scenario 8: Change Destination After Quote Proposal ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Take me to Civil Lines"}))
        route1, _ = await drain_frames_until(ws, "MAP_ROUTE_PREVIEW")
        record("8.1", "Initial route preview (Civil Lines)", "Civil Lines" in route1.get("destination", ""),
               f"Dest: {route1.get('destination')}, Fare: ₹{route1.get('fare_inr')}")

        # Customer changes mind to Sadar
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Actually take me to Sadar"}))
        route2, frames = await drain_frames_until(ws, "MAP_ROUTE_PREVIEW")
        record("8.2", "Updated route preview emitted", route2 is not None and "sadar" in route2.get("destination", "").lower(),
               f"Dest: {route2.get('destination') if route2 else 'None'}, Fare: ₹{route2.get('fare_inr') if route2 else 'None'}")
        
        # Confirm the new destination
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Yes, book it"}))
        conf_f, _ = await drain_frames_until(ws, "BOOKING_CONFIRMED")
        record("8.3", "Booking confirms updated destination", conf_f is not None and "sadar" in conf_f.get("destination", "").lower(),
               f"Confirmed for: {conf_f.get('destination')}")

    # -------------------------------------------------------------------------
    # SCENARIO 9: Duplicate Confirmation on Active Ride
    # -------------------------------------------------------------------------
    print("\n--- Scenario 9: Duplicate Confirmation Idempotency ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Take me to Civil Lines"}))
        await drain_frames_until(ws, "MAP_ROUTE_PREVIEW")
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Yes, book it"}))
        conf1, _ = await drain_frames_until(ws, "BOOKING_CONFIRMED")
        orig_bid = conf1.get("booking_id")
        await drain_frames_until(ws, "BOT_RESPONSE") # Drain first confirmation speech

        # Second confirmation attempt
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Confirm ride again"}))
        bot_f, frames = await drain_frames_until(ws, "BOT_RESPONSE")
        conf_frames = [f for f in frames if f.get("type") == "BOOKING_CONFIRMED"]
        record("9.1", "Zero duplicate BOOKING_CONFIRMED frames", len(conf_frames) == 0, "No duplicate booking frame")
        record("9.2", "Replies ride is already confirmed", "already confirmed" in bot_f.get("text", "").lower(),
               bot_f.get("text", "")[:50])

    # -------------------------------------------------------------------------
    # SCENARIO 10: Failure Modes & Unknown POI Clarification
    # -------------------------------------------------------------------------
    print("\n--- Scenario 10: Unknown Destination & Failure Modes ---")
    async with websockets.connect(WS_URL) as ws:
        await drain_frames_until(ws, "BOT_RESPONSE")

        # Search for completely unknown / nonexistent location
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Take me to Atlantis Floating Castle 999"}))
        bot_f, frames = await drain_frames_until(ws, "BOT_RESPONSE")
        record("10.1", "Deterministic clarification prompt on unknown place",
               "couldn't find" in bot_f.get("text", "").lower() or "landmark" in bot_f.get("text", "").lower(),
               bot_f.get("text", "")[:50])
        no_route = not any(f.get("type") == "MAP_ROUTE_PREVIEW" for f in frames)
        record("10.2", "No bogus route preview generated", no_route, "Clean fallback without hallucination")

    # -------------------------------------------------------------------------
    # PHASE 2 EXTENSION: Multilingual Marathi & Inactivity Heartbeat Check
    # -------------------------------------------------------------------------
    print("\n--- Multilingual Verification: Marathi (mr) ---")
    mr_url = "ws://127.0.0.1:8000/ws/voice?lat=21.1458&lon=79.0882&lang=mr"
    async with websockets.connect(mr_url) as ws:
        # Check Marathi greeting
        bot_f, _ = await drain_frames_until(ws, "BOT_RESPONSE")
        record("MR.1", "Marathi greeting received", "नमस्कार" in bot_f.get("text", ""), bot_f.get("text", "")[:45])

        # Marathi confirmation guard
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "हो बुक करा"}))
        bot_f, _ = await drain_frames_until(ws, "BOT_RESPONSE")
        record("MR.2", "Marathi pre-proposal guard speaks Marathi", "आपण अद्याप कोणतेही ठिकाण निवडलेले नाही" in bot_f.get("text", ""),
               bot_f.get("text", "")[:50])

    print(f"\n{'═' * 80}")
    print(f"  PHASE 3 AUDIT SUITE: ALL {len(results)} CHECKS PASSED WITH EMPIRICAL EVIDENCE! ✅")
    print(f"{'═' * 80}\n")

if __name__ == "__main__":
    asyncio.run(test_all_scenarios())
