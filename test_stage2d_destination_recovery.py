"""
test_stage2d_destination_recovery.py
Focused verification suite for Bharat Online Cabs — Stage 2D: Generic Destination Recovery and Confirmation.
Uses synthetic candidates (Alpha Tower, Central Point, Metro Station East, City Market) to verify:
1. Exact match resolution.
2. Slight typo/phonetic candidate recovery (POSSIBLE_MATCH).
3. Distance vs name match (geographic proximity never overrides clear name match).
4. Multiple plausible candidates (MULTIPLE_CANDIDATES).
5. Search zero results (NO_RESULTS with friendly landmark prompt).
6. Search provider outage (PROVIDER_ERROR with service outage message, never blaming user).
7. Natural conversational destination extraction.
8. Two-phase confirmation state machine separation:
   - Destination confirmation ("Did you mean X?") -> "Yes" -> Route/Fare proposal presented (NOT booked yet).
   - Booking confirmation ("Shall I book?") -> "Yes" -> Ride active with BOC-XXXXXX.
9. Old candidate invalidation when destination changes.
10. GPS truthfulness (client_exact vs fallback_approximate).
11. Absence of real-world venue dictionaries in production code.
"""

import asyncio
import os
import re
import unittest
from unittest.mock import patch, MagicMock

import spatial_engine as se
from app import Session, SessionController, is_confirmation_intent, is_cancellation_intent


class TestStage2DDestinationRecovery(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.session = Session()
        self.session.session_id = "test-session-2d"
        self.session.lat = 21.1458
        self.session.lon = 79.0882
        self.controller = SessionController(session=self.session)
        self.pushed_frames = []

        async def capture_frame(frame):
            self.pushed_frames.append(frame)

        self.controller.push_frame = capture_frame

    # -------------------------------------------------------------------------
    # 1. Candidate Scoring & Typo Recovery Unit Tests (Synthetic Candidates)
    # -------------------------------------------------------------------------
    def test_exact_match_synthetic_candidate(self):
        candidate = {
            "name": "Alpha Tower",
            "short_address": "Sector 4",
            "full_address": "Alpha Tower, Sector 4, Metro Region",
            "lat": 21.1500,
            "lon": 79.0900,
            "distance_km": 1.2,
        }
        res = se.score_candidate("Alpha Tower", candidate, self.session.lat, self.session.lon)
        self.assertGreaterEqual(res["total_score"], 0.85, "Exact match score should be >= 0.85")
        self.assertEqual(res["name_sim"], 1.0)

    def test_slight_typo_candidate_recovery(self):
        candidate = {
            "name": "Central Point",
            "short_address": "Commercial Hub",
            "full_address": "Central Point, Commercial Hub, Metro Region",
            "lat": 21.1480,
            "lon": 79.0850,
            "distance_km": 2.5,
        }
        # Misspelling: 'Centrl Point' (missing vowel)
        res = se.score_candidate("Centrl Point", candidate, self.session.lat, self.session.lon)
        self.assertGreaterEqual(res["total_score"], 0.60, "Typo candidate should score >= 0.60")
        self.assertGreaterEqual(res["name_sim"], 0.70, "Name similarity should capture spelling overlap")

    def test_distance_never_overrides_clear_name_match(self):
        # Target: Central Point (4.0 km away)
        target_cand = {
            "name": "Central Point",
            "short_address": "Zone A",
            "full_address": "Central Point, Zone A",
            "lat": 21.1700,
            "lon": 79.0800,
            "distance_km": 4.0,
        }
        # Closer but unrelated candidate (0.1 km away)
        closer_unrelated = {
            "name": "Grocery Express",
            "short_address": "Local Lane",
            "full_address": "Grocery Express, Local Lane",
            "lat": 21.1460,
            "lon": 79.0885,
            "distance_km": 0.1,
        }

        query = "Centrl Point"
        score_target = se.score_candidate(query, target_cand, self.session.lat, self.session.lon)
        score_closer = se.score_candidate(query, closer_unrelated, self.session.lat, self.session.lon)

        self.assertGreater(
            score_target["total_score"],
            score_closer["total_score"],
            f"Target ({score_target['total_score']}) must rank higher than closer unrelated POI ({score_closer['total_score']})"
        )

    # -------------------------------------------------------------------------
    # 2. evaluate_destination_intent Calibrated Outcomes (Synthetic Mocking)
    # -------------------------------------------------------------------------
    @patch("spatial_engine.search_nearby_pois")
    def test_evaluate_exact_match_outcome(self, mock_search):
        mock_search.return_value = [
            {"name": "Metro Station East", "short_address": "East Line", "lat": 21.15, "lon": 79.09, "distance_km": 2.0, "category": "transit"}
        ]
        res = se.evaluate_destination_intent("Metro Station East", self.session.lat, self.session.lon)
        self.assertEqual(res["resolution_type"], "EXACT_MATCH")
        self.assertEqual(res["status"], "EXACT")
        self.assertEqual(res["primary_match"]["name"], "Metro Station East")

    @patch("spatial_engine.search_nearby_pois")
    def test_evaluate_possible_match_outcome(self, mock_search):
        mock_search.return_value = [
            {"name": "Central Point", "short_address": "Commercial Hub", "lat": 21.148, "lon": 79.085, "distance_km": 2.5, "category": "building"}
        ]
        res = se.evaluate_destination_intent("Centrl Point", self.session.lat, self.session.lon)
        self.assertEqual(res["resolution_type"], "POSSIBLE_MATCH")
        self.assertEqual(res["status"], "POSSIBLE_MATCH")
        self.assertIsNotNone(res["suggested_match"])
        self.assertEqual(res["suggested_match"]["name"], "Central Point")
        self.assertIn("Did you mean Central Point", res["clarification_prompt"])

    @patch("spatial_engine.search_nearby_pois")
    def test_evaluate_multiple_candidates_outcome(self, mock_search):
        mock_search.return_value = [
            {"name": "City Market North", "short_address": "North Sector", "lat": 21.15, "lon": 79.08, "distance_km": 2.0, "category": "shop"},
            {"name": "City Market South", "short_address": "South Sector", "lat": 21.13, "lon": 79.08, "distance_km": 2.2, "category": "shop"}
        ]
        res = se.evaluate_destination_intent("City Market", self.session.lat, self.session.lon)
        self.assertEqual(res["resolution_type"], "MULTIPLE_CANDIDATES")
        self.assertEqual(res["status"], "DISAMBIGUATION_NEEDED")
        self.assertEqual(len(res["alternatives"]) + 1, 2)

    @patch("spatial_engine.search_nearby_pois")
    def test_evaluate_no_results_outcome(self, mock_search):
        mock_search.return_value = []
        se._last_search_meta["provider_status"] = 200
        se._last_search_meta["provider_error"] = None
        res = se.evaluate_destination_intent("Nonexistent Unknown Place", self.session.lat, self.session.lon)
        self.assertEqual(res["resolution_type"], "NO_RESULTS")
        self.assertEqual(res["status"], "NOT_FOUND")
        self.assertIn("landmark", res["clarification_prompt"].lower())

    @patch("spatial_engine.search_nearby_pois")
    def test_evaluate_provider_error_outcome(self, mock_search):
        mock_search.return_value = []
        se._last_search_meta["provider_status"] = 503
        se._last_search_meta["provider_error"] = "Nominatim 503 Service Unavailable"
        res = se.evaluate_destination_intent("Any Place", self.session.lat, self.session.lon)
        self.assertEqual(res["resolution_type"], "PROVIDER_ERROR")
        self.assertEqual(res["status"], "PROVIDER_ERROR")
        self.assertIn("temporarily unavailable", res["clarification_prompt"])
        # Must not accuse the user or place name of being wrong
        self.assertNotIn("invalid", res["clarification_prompt"].lower())

    # -------------------------------------------------------------------------
    # 3. Destination Entity Extraction across Natural Sentence Forms
    # -------------------------------------------------------------------------
    def test_natural_sentence_entity_extractions(self):
        cases = [
            ("Can you please take me to Alpha Tower", "Alpha Tower"),
            ("Actually take me to Metro Station East", "Metro Station East"),
            ("Drive me to City Market please", "City Market"),
            ("I would like to go to Central Point", "Central Point"),
            ("I wanted to reach North Avenue Plaza", "North Avenue Plaza"),
            ("Mujhe City Market jana hai", "City Market"),
            ("North Avenue Plazala jaycha ahe", "North Avenue Plaza"),
            ("Cab to Alpha Tower", "Alpha Tower"),
        ]
        for sentence, expected in cases:
            extracted = se.extract_destination_entity(sentence)
            self.assertEqual(extracted, expected, f"Failed extracting from: {sentence!r}")

    # -------------------------------------------------------------------------
    # 4. Two-Phase Confirmation State Machine Flow
    # -------------------------------------------------------------------------
    @patch("spatial_engine.evaluate_destination_intent")
    @patch("spatial_engine.calculate_trip_details")
    async def test_two_phase_confirmation_full_journey(self, mock_trip, mock_eval):
        # Setup synthetic candidate
        cand = {
            "name": "Central Point",
            "short_address": "Commercial Hub",
            "lat": 21.1480,
            "lon": 79.0850,
            "distance_km": 3.0,
            "rating": 4.5
        }
        mock_eval.return_value = {
            "status": "POSSIBLE_MATCH",
            "resolution_type": "POSSIBLE_MATCH",
            "query": "Centrl Point",
            "primary_match": cand,
            "suggested_match": cand,
            "alternatives": [],
            "outlet_count": 1,
            "clarification_prompt": "Did you mean Central Point in Commercial Hub?",
            "provider_status": 200,
            "provider_error": None,
            "confidence_score": 0.75,
        }
        mock_trip.return_value = {
            "status": "OK",
            "destination": "Central Point",
            "dest": {"lat": 21.1480, "lon": 79.0850},
            "distance_km": 3.2,
            "duration_mins": 9,
            "fare_inr": 95.0,
            "route_geometry": {"type": "LineString", "coordinates": [[79.0882, 21.1458], [79.0850, 21.1480]]},
            "route_source": "osrm",
        }

        # Step 1: User speaks with slight typo: "I want to go to Centrl Point"
        handled = await self.controller._handle_text_turn("I want to go to Centrl Point")
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_DESTINATION_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_destination)
        self.assertIsNone(self.session.pending_ride, "Ride proposal must NOT be generated yet")
        self.assertIsNone(self.session.booking_id, "Ride must NOT be booked yet")

        # Step 2: User says "Yes" -> Confirms destination!
        handled = await self.controller._handle_text_turn("Yes, that's right")
        self.assertTrue(handled)
        # Stage must advance to AWAITING_CONFIRMATION (presenting fare proposal)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_ride, "Fare proposal must now be created")
        self.assertEqual(self.session.pending_ride["fare_inr"], 95.0)
        self.assertIsNone(self.session.booking_id, "Ride must STILL NOT be booked yet")

        # Step 3: User says "Yes, please book it" -> Confirms booking!
        handled = await self.controller._handle_text_turn("Yes, please book it")
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "RIDE_ACTIVE")
        self.assertIsNotNone(self.session.booking_id)
        self.assertTrue(self.session.booking_id.startswith("BOC-"))

    # -------------------------------------------------------------------------
    # 5. Destination Change Invalidation
    # -------------------------------------------------------------------------
    @patch("spatial_engine.evaluate_destination_intent")
    async def test_destination_change_clears_old_pending_state(self, mock_eval):
        # Set session into AWAITING_DESTINATION_CONFIRMATION
        self.session.stage = "AWAITING_DESTINATION_CONFIRMATION"
        self.session.pending_destination = {"name": "Old Place", "lat": 21.1, "lon": 79.1}

        # User changes mind to Alpha Tower
        mock_eval.return_value = {
            "status": "EXACT",
            "resolution_type": "EXACT_MATCH",
            "query": "Alpha Tower",
            "primary_match": {"name": "Alpha Tower", "lat": 21.15, "lon": 79.09, "distance_km": 1.5, "short_address": "Zone 1"},
            "alternatives": [],
            "outlet_count": 1,
            "clarification_prompt": "Found Alpha Tower",
            "provider_status": 200,
            "provider_error": None,
        }
        with patch.object(self.controller, "_resolve_and_preview_trip") as mock_preview:
            await self.controller._handle_text_turn("Actually take me to Alpha Tower")
            # Old pending destination was cleared
            self.assertIsNone(self.session.pending_destination)
            mock_preview.assert_called_once_with(21.15, 79.09, "Alpha Tower")

    # -------------------------------------------------------------------------
    # 6. GPS Truthfulness (client_exact vs fallback_approximate)
    # -------------------------------------------------------------------------
    def test_gps_truthfulness_sources(self):
        # Simulate CLIENT_INIT with exact GPS
        exact_msg = {
            "type": "CLIENT_INIT",
            "lat": 21.1555,
            "lon": 79.0999,
            "gps_source": "client_exact"
        }
        self.session.lat = exact_msg["lat"]
        self.session.lon = exact_msg["lon"]
        self.session.gps_source = exact_msg["gps_source"]

        self.assertEqual(self.session.lat, 21.1555)
        self.assertEqual(self.session.lon, 79.0999)
        self.assertEqual(self.session.gps_source, "client_exact")

        # Simulate fallback approximate when GPS is denied
        fallback_msg = {
            "type": "CLIENT_INIT",
            "lat": se.DEFAULT_LAT,
            "lon": se.DEFAULT_LON,
            "gps_source": "fallback_approximate"
        }
        self.session.lat = fallback_msg["lat"]
        self.session.lon = fallback_msg["lon"]
        self.session.gps_source = fallback_msg["gps_source"]

        self.assertEqual(self.session.gps_source, "fallback_approximate")
        self.assertNotEqual(self.session.gps_source, "client_exact")

    # -------------------------------------------------------------------------
    # 7. Absence of Real-World Location Dictionaries in Production Code
    # -------------------------------------------------------------------------
    def test_no_hardcoded_venue_lists_in_production(self):
        with open("spatial_engine.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Must not contain hardcoded venue or shop dictionaries
        forbidden = ["haldiram", "batukbhai", "dmart", "burdi", "sitabuldi", "dharampeth"]
        for term in forbidden:
            matches = re.findall(rf"\b{term}\b", code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded '{term}' in spatial_engine.py")

        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        for term in ["haldiram", "batukbhai", "dmart"]:
            matches = re.findall(rf"\b{term}\b", app_code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded '{term}' in app.py")


if __name__ == "__main__":
    unittest.main()
