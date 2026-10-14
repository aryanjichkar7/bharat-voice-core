"""
test_stage2e_conversation_and_lifecycle.py - Stage 2E Comprehensive Regression Suite
Verifies:
1. Exact match vs fuzzy typo/phonetic candidate (requiring destination confirmation, no silent normalization).
2. Natural composite sentence structures with intent + destination (e.g. "Book a cab to Central Point").
3. Strict two-phase confirmation separation (destination confirm != booking confirm).
4. Negative destination & proposal cancellations.
5. Stale candidate invalidation on destination change.
6. Backend text turn deduplication.
7. Zero hardcoded real-world venue dictionaries in production code.
"""

import asyncio
import re
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

import app
import spatial_engine as se
from app import Session, SessionController, is_confirmation_intent, is_cancellation_intent


class TestStage2EConversationAndLifecycle(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.session = Session(session_id="test-session-2e")
        self.session.lat = 21.1458
        self.session.lon = 79.0882
        self.session.gps_source = "client_exact"
        self.mock_push_frame = AsyncMock()
        self.controller = SessionController(self.session, self.mock_push_frame)

    # -------------------------------------------------------------------------
    # 1. Defect 1: Strict Exact Name Check (No Silent Normalization of Typos)
    # -------------------------------------------------------------------------
    def test_strict_exact_name_vs_typo(self):
        candidate = {
            "name": "Central Point",
            "short_address": "Commercial Hub",
            "lat": 21.1480,
            "lon": 79.0850,
            "distance_km": 2.5,
            "category": "commercial",
        }

        # Exact match (ignoring case & whitespace) -> EXACT_MATCH
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]):
            res_exact = se.evaluate_destination_intent("Central Point", self.session.lat, self.session.lon)
            self.assertEqual(res_exact["resolution_type"], "EXACT_MATCH")
            self.assertEqual(res_exact["status"], "EXACT")
            self.assertIn("Found Central Point", res_exact["clarification_prompt"])

        # Typo with high similarity ("Centrl Point") -> MUST NOT silently normalize to EXACT_MATCH
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]):
            res_typo = se.evaluate_destination_intent("Centrl Point", self.session.lat, self.session.lon)
            self.assertNotEqual(res_typo["resolution_type"], "EXACT_MATCH")
            self.assertEqual(res_typo["resolution_type"], "POSSIBLE_MATCH")
            self.assertEqual(res_typo["status"], "POSSIBLE_MATCH")
            self.assertIn("Did you mean Central Point", res_typo["clarification_prompt"])

        # Another spelling variation ("Central Poin") -> POSSIBLE_MATCH
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]):
            res_typo2 = se.evaluate_destination_intent("Central Poin", self.session.lat, self.session.lon)
            self.assertNotEqual(res_typo2["resolution_type"], "EXACT_MATCH")
            self.assertEqual(res_typo2["resolution_type"], "POSSIBLE_MATCH")
            self.assertIn("Did you mean Central Point", res_typo2["clarification_prompt"])

    # -------------------------------------------------------------------------
    # 2. Defect 2: Natural Composite Requests (Intent + Destination)
    # -------------------------------------------------------------------------
    def test_composite_intent_recognition_not_rejected(self):
        # Bare confirmation requests should be recognized as confirmation
        self.assertTrue(is_confirmation_intent("Book a cab"))
        self.assertTrue(is_confirmation_intent("yes"))
        self.assertTrue(is_confirmation_intent("please book"))
        self.assertTrue(is_confirmation_intent("book it"))

        # Composite requests containing destination prepositions MUST NOT be bare confirmation
        self.assertFalse(is_confirmation_intent("Book a cab to Central Point"))
        self.assertFalse(is_confirmation_intent("Please book a cab to Airport"))
        self.assertFalse(is_confirmation_intent("Need a ride towards North Point"))
        self.assertFalse(is_confirmation_intent("Book a taxi to Railway Station"))

        # Natural intent parser extracts destination cleanly
        p1 = se.parse_natural_intent_and_destination("Book a cab to Central Point")
        self.assertEqual(p1["intent"], "BOOK_CAB")
        self.assertEqual(p1["destination"], "Central Point")

        p2 = se.parse_natural_intent_and_destination("Please take me to Alpha Tower")
        self.assertEqual(p2["intent"], "DESTINATION_SEARCH")
        self.assertEqual(p2["destination"], "Alpha Tower")

        p3 = se.parse_natural_intent_and_destination("Need a ride for City Hospital")
        self.assertEqual(p3["destination"], "City Hospital")

    async def test_natural_request_reaches_destination_evaluation(self):
        # Ensure "Book a cab to Central Point" in GREETING stage evaluates destination and does not reject
        candidate = {
            "name": "Central Point",
            "short_address": "Zone 1",
            "lat": 21.1500,
            "lon": 79.0900,
            "distance_km": 1.5,
            "category": "commercial",
        }
        self.session.stage = "GREETING"
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]), \
             patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview:
            handled = await self.controller._handle_text_turn("Book a cab to Central Point")
            self.assertTrue(handled)
            mock_preview.assert_called_once_with(21.1500, 79.0900, "Central Point")

    # -------------------------------------------------------------------------
    # 3. Two-Phase Confirmation Separation
    # -------------------------------------------------------------------------
    async def test_two_phase_confirmation_full_flow(self):
        candidate = {
            "name": "Central Point",
            "short_address": "Zone 1",
            "lat": 21.1500,
            "lon": 79.0900,
            "distance_km": 1.5,
            "category": "commercial",
        }

        # Step 1: User gives typo destination "Centrl Point"
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]):
            await self.controller._handle_text_turn("Take me to Centrl Point")

        # Must be in AWAITING_DESTINATION_CONFIRMATION, not booked
        self.assertEqual(self.session.stage, "AWAITING_DESTINATION_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_destination)
        self.assertEqual(self.session.pending_destination["name"], "Central Point")
        self.assertIsNone(self.session.booking_id)

        # Step 2: User confirms destination ("Yes")
        with patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview:
            await self.controller._handle_text_turn("Yes, that's right")
            mock_preview.assert_called_once_with(21.1500, 79.0900, "Central Point")

        # Simulate proposal established
        self.session.stage = "AWAITING_CONFIRMATION"
        self.session.pending_ride = {
            "destination": "Central Point",
            "distance_km": 1.5,
            "duration_mins": 5,
            "fare_inr": 71.0,
        }

        # Step 3: User confirms booking ("Yes, please book it")
        with patch.object(self.controller, "_confirm_current_booking", new_callable=AsyncMock) as mock_confirm:
            await self.controller._handle_text_turn("Yes, please book it")
            mock_confirm.assert_called_once()

    async def test_negative_destination_confirmation_returns_to_awaiting_dest(self):
        self.session.stage = "AWAITING_DESTINATION_CONFIRMATION"
        self.session.pending_destination = {"name": "Central Point", "lat": 21.15, "lon": 79.09}

        handled = await self.controller._handle_text_turn("No, that's not where I want to go")
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_DESTINATION")
        self.assertIsNone(self.session.pending_destination)

    async def test_negative_booking_confirmation_cancels_proposal(self):
        self.session.stage = "AWAITING_CONFIRMATION"
        self.session.pending_ride = {"destination": "Central Point", "fare_inr": 75}

        handled = await self.controller._handle_text_turn("No, cancel the ride")
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_DESTINATION")
        self.assertIsNone(self.session.pending_ride)

    async def test_new_destination_invalidates_stale_candidate_or_proposal(self):
        # Case A: In AWAITING_DESTINATION_CONFIRMATION, user supplies new destination
        self.session.stage = "AWAITING_DESTINATION_CONFIRMATION"
        self.session.pending_destination = {"name": "Old Place", "lat": 21.1, "lon": 79.1}

        new_cand = {"name": "Metro Station", "lat": 21.16, "lon": 79.08, "distance_km": 2.0}
        with patch("spatial_engine.search_nearby_pois", return_value=[new_cand]), \
             patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview:
            await self.controller._handle_text_turn("Actually go to Metro Station")
            self.assertIsNone(self.session.pending_destination)
            mock_preview.assert_called_once_with(21.16, 79.08, "Metro Station")

        # Case B: In AWAITING_CONFIRMATION, user changes mind to another destination
        self.session.stage = "AWAITING_CONFIRMATION"
        self.session.pending_ride = {"destination": "Metro Station", "fare_inr": 80}
        with patch("spatial_engine.search_nearby_pois", return_value=[new_cand]), \
             patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview2:
            await self.controller._handle_text_turn("Change destination to Metro Station")
            mock_preview2.assert_called_once_with(21.16, 79.08, "Metro Station")

    # -------------------------------------------------------------------------
    # 4. Backend Turn Deduplication
    # -------------------------------------------------------------------------
    async def test_backend_duplicate_text_turn_suppression(self):
        processor = self.controller
        processor.push_frame = AsyncMock()

        from pipecat.frames.frames import InputTransportMessageFrame
        msg_frame = InputTransportMessageFrame(message={"type": "TEXT_INPUT", "text": "Book a cab to Central Point"})

        # First delivery -> Accepted
        await processor.process_frame(msg_frame, app.FrameDirection.DOWNSTREAM)
        self.assertEqual(self.session.turns, 1)

        # Immediate duplicate delivery within 100ms -> Suppressed
        await processor.process_frame(msg_frame, app.FrameDirection.DOWNSTREAM)
        self.assertEqual(self.session.turns, 1, "Duplicate TEXT_INPUT within 1.0s should be suppressed")

    # -------------------------------------------------------------------------
    # 5. Zero Hardcoded Real-World Dictionaries Rule Check
    # -------------------------------------------------------------------------
    def test_no_hardcoded_venue_lists(self):
        with open("spatial_engine.py", "r", encoding="utf-8") as f:
            code = f.read()
        for term in ["haldiram", "batukbhai", "dmart", "burdi", "sitabuldi", "dharampeth"]:
            matches = re.findall(rf"\b{term}\b", code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded '{term}' in spatial_engine.py")

        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()
        for term in ["haldiram", "batukbhai", "dmart"]:
            matches = re.findall(rf"\b{term}\b", app_code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded '{term}' in app.py")


if __name__ == "__main__":
    unittest.main()
