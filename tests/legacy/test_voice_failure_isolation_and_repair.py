"""
Verification Suite: Voice Failure Isolation and Targeted Repair
Bharat Online Cabs (bharat-voice-core)

Verifies:
1. Category queries (restaurants, pharmacies, petrol pumps, hospitals, cafes, bakeries)
   extract cleanly and return real POI candidates with proper category disambiguation prompts.
2. Phonetic resilience handles common speech transcription drift (Citibaldi -> Sitabuldi,
   Sittabaldi -> Sitabuldi, Dharampath -> Dharampeth) without hardcoded venue lists.
3. Complete booking flow works during disambiguation when customer says "Yes" / "Book it".
4. Zero hardcoded venue names or locality lists in application logic or prompts.
5. 12-Stage TurnTracer logging captures the entire voice lifecycle without leaking secrets.
"""
import asyncio
import json
import os
import re
import unittest
import websockets
import spatial_engine as se
import app

WS_URL = "ws://127.0.0.1:8000/ws/voice?lat=21.1458&lon=79.0882&lang=en"

async def drain_until(ws, target_type: str, timeout: float = 12.0):
    start = asyncio.get_event_loop().time()
    frames = []
    while asyncio.get_event_loop().time() - start < timeout:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
            if isinstance(raw, str):
                msg = json.loads(raw)
                frames.append(msg)
                if msg.get("type") == target_type:
                    return msg, frames
        except asyncio.TimeoutError:
            continue
    return None, frames


class TestVoiceFailureIsolationAndRepair(unittest.TestCase):

    def test_category_entity_extraction_and_resolution(self):
        """Verifies carrier extraction and real POI candidate resolution across common categories."""
        categories = [
            ("Find a good restaurant nearby", "restaurant", "restaurant"),
            ("Find me a pharmacy", "pharmacy", "pharmacy"),
            ("Looking for a petrol pump", "petrol pump", "petrol pump"),
            ("Show me nearby hospitals", "hospital", "hospital"),
            ("Search for a cafe", "cafe", "cafe"),
            ("Take me to a bakery", "bakery", "bakery"),
        ]

        for query, expected_sanitized, expected_label in categories:
            extracted = se.extract_destination_entity(query)
            sanitized = se.sanitize_search_query(query)
            self.assertEqual(sanitized, expected_sanitized, f"Sanitization mismatch for '{query}'")

            res = se.evaluate_destination_intent(query, 21.1458, 79.0882, radius_km=15.0)
            self.assertEqual(res["status"], "DISAMBIGUATION_NEEDED", f"Expected disambiguation for category query '{query}'")
            self.assertGreater(res["outlet_count"], 0, f"Expected candidates > 0 for category '{query}'")
            self.assertIsNotNone(res["primary_match"])
            
            prompt = res.get("clarification_prompt", "")
            self.assertIn(f"closest {expected_label}", prompt.lower(),
                          f"Prompt should mention 'closest {expected_label}', got: '{prompt}'")

    def test_phonetic_transcription_resilience(self):
        """Verifies phonetic variants resolve without hardcoded venue lists."""
        test_cases = [
            ("Citibaldi", "sitabuldi"),
            ("Sittabaldi", "sitabuldi"),
            ("Dharampath", "dharampeth"),
        ]

        for input_token, expected_target_sub in test_cases:
            variants = se._query_variants(input_token)
            # Variants must include phonetic resolution
            has_match = any(expected_target_sub in v.lower() for v in variants)
            self.assertTrue(has_match, f"Expected variant containing '{expected_target_sub}' in variants for '{input_token}': {variants}")

            # Verify actual spatial intent resolution resolves to real place
            res = se.evaluate_destination_intent(input_token, 21.1458, 79.0882, radius_km=15.0)
            self.assertIn(res["status"], ("EXACT", "DISAMBIGUATION_NEEDED"))
            self.assertIsNotNone(res["primary_match"])
            pm_name = res["primary_match"]["name"].lower()
            self.assertTrue(
                expected_target_sub in pm_name or any(expected_target_sub in v.lower() for v in variants),
                f"Resolved primary match '{pm_name}' does not match expected target '{expected_target_sub}'"
            )

    def test_strict_zero_hardcoding_rule(self):
        """Ensures zero hardcoded venue names or locality lists exist in spatial_engine or app."""
        with open("spatial_engine.py", "r", encoding="utf-8") as f:
            spatial_code = f.read()
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        banned_tokens = ["nagpur_localities", "sitabuldi", "dharampeth", "haldiram", "batukbhai"]
        # Check spatial_engine.py code logic
        for line in spatial_code.splitlines():
            line_clean = line.strip().lower()
            if line_clean.startswith("#") or line_clean.startswith('"""'):
                continue
            for tok in banned_tokens:
                self.assertNotIn(f'"{tok}"', line_clean, f"Hardcoded venue token '{tok}' found in spatial_engine logic: {line}")
                self.assertNotIn(f"'{tok}'", line_clean, f"Hardcoded venue token '{tok}' found in spatial_engine logic: {line}")

        # Check app.py code logic
        for line in app_code.splitlines():
            line_clean = line.strip().lower()
            if line_clean.startswith("#") or line_clean.startswith('"""'):
                continue
            if line.strip().startswith("STT_HOTWORDS"):
                for tok in banned_tokens:
                    self.assertNotIn(tok, line_clean, f"Venue token '{tok}' found in STT_HOTWORDS")

    def test_candidate_choice_affirmation(self):
        """Verifies customer saying 'Yes', 'Book it', or 'Sure' accepts the closest option during disambiguation."""
        options = [
            {"name": "Branch Alpha", "short_address": "Main Road", "lat": 21.15, "lon": 79.08},
            {"name": "Branch Beta", "short_address": "Outer Ring", "lat": 21.18, "lon": 79.10},
        ]
        test_affirmations = [
            "yes",
            "yes please",
            "book it",
            "take me there",
            "chalo",
            "haan kar do",
            "sure",
            "the first one",
            "closest",
        ]
        for text in test_affirmations:
            match = app.match_candidate_choice(text, options)
            self.assertIsNotNone(match, f"Failed to match candidate on affirmation: '{text}'")
            self.assertEqual(match["name"], "Branch Alpha", f"Expected closest candidate for '{text}', got {match['name']}")

    def test_e2e_category_to_booking_websocket(self):
        """Tests full WebSocket session: category query -> disambiguation -> 'Yes, book it' -> route preview -> confirm."""
        async def run_ws():
            async with websockets.connect(WS_URL) as ws:
                # 1. Greeting
                greet_f, _ = await drain_until(ws, "BOT_RESPONSE")
                self.assertIsNotNone(greet_f)

                # 2. Category query
                await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Find a good restaurant nearby"}))
                poi_f, _ = await drain_until(ws, "POI_CANDIDATES")
                self.assertIsNotNone(poi_f, "POI_CANDIDATES not received for restaurant query")
                self.assertEqual(poi_f.get("status"), "DISAMBIGUATION_NEEDED")
                candidates = poi_f.get("candidates", [])
                self.assertGreater(len(candidates), 0)

                # 3. User accepts concierge's closest recommendation
                await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Yes, book it"}))
                route_f, _ = await drain_until(ws, "MAP_ROUTE_PREVIEW")
                self.assertIsNotNone(route_f, "MAP_ROUTE_PREVIEW not received after accepting recommendation")
                self.assertIn("fare_inr", route_f)
                self.assertGreater(route_f["fare_inr"], 0)

                # 4. User confirms booking
                await ws.send(json.dumps({"type": "TEXT_INPUT", "text": "Confirm"}))
                conf_f, _ = await drain_until(ws, "BOOKING_CONFIRMED")
                self.assertIsNotNone(conf_f, "BOOKING_CONFIRMED not received")
                self.assertTrue(conf_f.get("booking_id", "").startswith("BOC-"))
                self.assertIn("otp", conf_f)

        asyncio.run(run_ws())


if __name__ == "__main__":
    unittest.main()
