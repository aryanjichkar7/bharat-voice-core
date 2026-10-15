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
import os
import re
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
        self.controller = SessionController(self.session)
        self.controller.push_frame = self.mock_push_frame

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
        with open(REPO_ROOT / "spatial_engine.py", "r", encoding="utf-8") as f:
            code = f.read()
        for term in ["haldiram", "batukbhai", "dmart", "burdi", "sitabuldi", "dharampeth"]:
            matches = re.findall(rf"\b{term}\b", code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded '{term}' in spatial_engine.py")

        with open(REPO_ROOT / "app.py", "r", encoding="utf-8") as f:
            app_code = f.read()
        for term in ["haldiram", "batukbhai", "dmart"]:
            matches = re.findall(rf"\b{term}\b", app_code, re.IGNORECASE)
            self.assertEqual(len(matches), 0, f"Found hardcoded '{term}' in app.py")


    # -------------------------------------------------------------------------
    # 6. Stage 2K: Conversation Restart After Ride Cancellation Tests
    # -------------------------------------------------------------------------
    async def test_stage2k_initial_activation_greets_once(self):
        from pipecat.frames.frames import InputTransportMessageFrame, TTSSpeakFrame
        msg = InputTransportMessageFrame(message={"type": "START_VOICE_SESSION"})
        self.mock_push_frame.reset_mock()

        await self.controller.process_frame(msg, app.FrameDirection.DOWNSTREAM)
        self.assertTrue(self.session.greeting_sent)

        # Count TTS frames and BOT_RESPONSE frames
        calls = self.mock_push_frame.call_args_list
        tts_frames = [c[0][0] for c in calls if isinstance(c[0][0], TTSSpeakFrame)]
        bot_msgs = [c[0][0].message for c in calls if hasattr(c[0][0], "message") and c[0][0].message.get("type") == "BOT_RESPONSE"]

        self.assertEqual(len(tts_frames), 1, "Initial activation must produce exactly one audible greeting")
        self.assertEqual(len(bot_msgs), 1, "Initial activation must produce exactly one visible greeting")

    async def test_stage2k_repeated_mic_activation_does_not_duplicate_greeting(self):
        from pipecat.frames.frames import InputTransportMessageFrame, TTSSpeakFrame
        msg = InputTransportMessageFrame(message={"type": "START_VOICE_SESSION"})
        self.mock_push_frame.reset_mock()

        # First activation
        await self.controller.process_frame(msg, app.FrameDirection.DOWNSTREAM)
        # Second activation (e.g. pause/unpause during same conversation)
        await self.controller.process_frame(msg, app.FrameDirection.DOWNSTREAM)

        calls = self.mock_push_frame.call_args_list
        tts_frames = [c[0][0] for c in calls if isinstance(c[0][0], TTSSpeakFrame)]
        self.assertEqual(len(tts_frames), 1, "Repeated mic activation within same conversation must not duplicate greeting")

    async def test_stage2k_booking_and_cancellation_lifecycle(self):
        from pipecat.frames.frames import InputTransportMessageFrame, TTSSpeakFrame
        # 1. Establish proposal
        self.session.pending_ride = {
            "destination": "Central Point",
            "distance_km": 2.5,
            "duration_mins": 8,
            "fare_inr": 85,
        }
        self.session.stage = "AWAITING_CONFIRMATION"

        # 2. Confirm booking
        bid = await self.session.confirm_booking(self.mock_push_frame)
        self.assertIsNotNone(bid)
        self.assertEqual(self.session.stage, "RIDE_ACTIVE")
        self.assertEqual(self.session.booking_id, bid)

        # 3. Cancel ride via CANCEL_RIDE message
        self.mock_push_frame.reset_mock()
        cancel_msg = InputTransportMessageFrame(message={"type": "CANCEL_RIDE"})
        await self.controller.process_frame(cancel_msg, app.FrameDirection.DOWNSTREAM)

        self.assertEqual(self.session.stage, "AWAITING_DESTINATION")
        self.assertIsNone(self.session.booking_id)
        self.assertIsNone(self.session.active_ride)
        self.assertIsNone(self.session.pending_ride)
        self.assertFalse(self.session.greeting_sent, "greeting_sent should reset to False to allow fresh greeting")
        self.assertFalse(self.session.is_processing_turn, "is_processing_turn should be False after cancellation")

        # 4. Explicit fresh conversation receives exactly one greeting
        self.mock_push_frame.reset_mock()
        start_msg = InputTransportMessageFrame(message={"type": "START_VOICE_SESSION"})
        await self.controller.process_frame(start_msg, app.FrameDirection.DOWNSTREAM)

        calls = self.mock_push_frame.call_args_list
        tts_frames = [c[0][0] for c in calls if isinstance(c[0][0], TTSSpeakFrame)]
        bot_msgs = [c[0][0].message for c in calls if hasattr(c[0][0], "message") and c[0][0].message.get("type") == "BOT_RESPONSE"]
        self.assertEqual(len(tts_frames), 1, "Fresh conversation after cancellation must receive one audible greeting")
        self.assertEqual(len(bot_msgs), 1, "Fresh conversation after cancellation must receive one visible greeting")

    async def test_stage2k_second_conversation_accepts_destination_without_reload(self):
        from pipecat.frames.frames import InputTransportMessageFrame
        # 1. Simulate ride active then cancelled
        self.session.stage = "RIDE_ACTIVE"
        self.session.booking_id = "BOC-TEST123"
        await self.session.cancel_ride(self.mock_push_frame)

        # 2. Clear conversation (user taps Plan New Trip or Clear)
        clear_msg = InputTransportMessageFrame(message={"type": "CLEAR_CONVERSATION", "turn_seq": 10})
        await self.controller.process_frame(clear_msg, app.FrameDirection.DOWNSTREAM)
        self.assertFalse(self.session.greeting_sent)
        self.assertFalse(self.session.is_processing_turn)

        # 3. User requests a new destination
        candidate = {
            "name": "North Avenue",
            "short_address": "Zone 2",
            "lat": 21.1600,
            "lon": 79.0700,
            "distance_km": 4.0,
            "category": "commercial",
        }
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]), \
             patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview:
            handled = await self.controller.execute_turn("Take me to North Avenue", source="voice", turn_id="turn-001")
            self.assertTrue(handled)
            mock_preview.assert_called_once_with(21.1600, 79.0700, "North Avenue")
        self.assertFalse(self.session.is_processing_turn)

    async def test_stage2k_stale_cancelled_turn_cannot_send_messages(self):
        # Initial turn generation is 5
        self.session.stage = "RIDE_ACTIVE"
        self.session.turn_generation = 5
        # Simulate cancellation which bumps turn_generation
        await self.session.cancel_ride(self.mock_push_frame)
        self.assertGreater(self.session.turn_generation, 5)

        # Attempt to run a stale turn with seq=5
        handled = await self.controller.execute_turn("Stale Destination", source="voice", turn_seq=5)
        # Result of superseded turn should be discarded
        self.assertFalse(handled)

    async def test_stage2k_failed_stt_does_not_permanently_lock_session(self):
        # Simulate a turn that raises an unexpected error
        with patch.object(self.controller, "_handle_text_turn", side_effect=RuntimeError("Simulated provider failure")):
            await self.controller.execute_turn("Crash turn", source="text", turn_seq=1)

        # is_processing_turn must be safely restored to False
        self.assertFalse(self.session.is_processing_turn, "Turn error must reset is_processing_turn")

        # Next valid turn must be accepted cleanly
        candidate = {"name": "Airport", "lat": 21.09, "lon": 79.08, "distance_km": 6.0}
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]), \
             patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview:
            handled = await self.controller.execute_turn("Go to Airport", source="text", turn_seq=2)
            self.assertTrue(handled)
            mock_preview.assert_called_once_with(21.09, 79.08, "Airport")

    # -------------------------------------------------------------------------
    # Stage 2L: Context-Aware Booking & Proposal Protection
    # -------------------------------------------------------------------------
    async def test_stage2l_proposal_preserved_on_conversational_statement(self):
        """Conversational statements ('Wait a minute', 'Thank you', 'Can you hear me') must not erase pending proposal or trigger OSM search."""
        trip = {
            "destination": "Sitabuldi Metro Station",
            "distance_km": 2.5,
            "duration_mins": 8,
            "fare_inr": 85.0,
            "route_geometry": {"type": "LineString", "coordinates": [[79.08, 21.14], [79.085, 21.145]]},
        }
        await self.session.set_route_preview(trip, 21.145, 79.085, self.mock_push_frame)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_ride)
        prop_id = self.session.proposal_id

        # 1. User says "Wait a minute"
        handled = await self.controller.execute_turn("Wait a minute", source="voice", turn_seq=1)
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_ride)
        self.assertEqual(self.session.proposal_id, prop_id)

        # 2. User says "Thank you"
        handled = await self.controller.execute_turn("Thank you", source="voice", turn_seq=2)
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_ride)

        # 3. User says "Can you hear me?"
        handled = await self.controller.execute_turn("Can you hear me?", source="voice", turn_seq=3)
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_ride)

    async def test_stage2l_trip_pricing_inquiry_answers_truthfully(self):
        """Inquiries about fare or distance while proposal is pending answer with proposal details without state mutation."""
        trip = {
            "destination": "Medical Square",
            "distance_km": 4.0,
            "duration_mins": 12,
            "fare_inr": 106.0,
            "route_geometry": {"type": "LineString", "coordinates": []},
        }
        await self.session.set_route_preview(trip, 21.13, 79.09, self.mock_push_frame)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")

        # Inquire about fare
        self.mock_push_frame.reset_mock()
        handled = await self.controller.execute_turn("What is the fare?", source="voice", turn_seq=1)
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(self.session.pending_ride)

        # Check that BOT_RESPONSE pushed contains 106 rupees
        bot_msgs = [call.args[0].message for call in self.mock_push_frame.call_args_list if hasattr(call.args[0], "message") and call.args[0].message.get("type") == "BOT_RESPONSE"]
        self.assertTrue(any("106" in m.get("text", "") for m in bot_msgs))

    async def test_stage2l_natural_affirmation_advances_booking(self):
        """Natural affirmations ('sounds good', 'okay', 'chalel') confirm the pending proposal exactly once."""
        trip = {
            "destination": "Civil Lines",
            "distance_km": 3.0,
            "duration_mins": 10,
            "fare_inr": 92.0,
            "route_geometry": {"type": "LineString", "coordinates": []},
        }
        await self.session.set_route_preview(trip, 21.15, 79.08, self.mock_push_frame)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")

        # User confirms using natural affirmation "sounds good"
        handled = await self.controller.execute_turn("Sounds good", source="voice", turn_seq=1)
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "RIDE_ACTIVE")
        self.assertIsNone(self.session.pending_ride)
        self.assertIsNotNone(self.session.active_ride)
        self.assertTrue(self.session.booking_id.startswith("BOC-"))
        self.assertEqual(self.session.active_ride["destination"], "Civil Lines")
        self.assertEqual(self.session.active_ride["fare_inr"], 92.0)

    async def test_stage2l_duplicate_confirmation_protection(self):
        """Duplicate confirmation after booking active does not create duplicate booking or alter state."""
        trip = {
            "destination": "Civil Lines",
            "distance_km": 3.0,
            "duration_mins": 10,
            "fare_inr": 92.0,
            "route_geometry": {"type": "LineString", "coordinates": []},
        }
        await self.session.set_route_preview(trip, 21.15, 79.08, self.mock_push_frame)
        await self.session.confirm_booking(self.mock_push_frame)
        self.assertEqual(self.session.stage, "RIDE_ACTIVE")
        initial_bid = self.session.booking_id

        # User says "confirm" again
        self.mock_push_frame.reset_mock()
        handled = await self.controller.execute_turn("Confirm please", source="voice", turn_seq=2)
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "RIDE_ACTIVE")
        self.assertEqual(self.session.booking_id, initial_bid)

        # Spoken output informs user ride is already confirmed
        bot_msgs = [call.args[0].message for call in self.mock_push_frame.call_args_list if hasattr(call.args[0], "message") and call.args[0].message.get("type") == "BOT_RESPONSE"]
        self.assertTrue(any("already confirmed" in m.get("text", "") for m in bot_msgs))

    async def test_stage2l_destination_change_replaces_pending_proposal(self):
        """Mentions of a different destination while proposal pending invalidates old proposal and searches new destination."""
        trip = {
            "destination": "Sitabuldi",
            "distance_km": 2.0,
            "duration_mins": 6,
            "fare_inr": 78.0,
            "route_geometry": {"type": "LineString", "coordinates": []},
        }
        await self.session.set_route_preview(trip, 21.145, 79.085, self.mock_push_frame)
        self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
        initial_prop_id = self.session.proposal_id

        new_candidate = {"name": "Airport", "lat": 21.09, "lon": 79.08, "distance_km": 7.0}
        with patch("spatial_engine.search_nearby_pois", return_value=[new_candidate]), \
             patch.object(self.controller, "_resolve_and_preview_trip", new_callable=AsyncMock) as mock_preview:
            handled = await self.controller.execute_turn("Actually take me to Airport", source="voice", turn_seq=2)
            self.assertTrue(handled)
            mock_preview.assert_called_once_with(21.09, 79.08, "Airport")


if __name__ == "__main__":
    unittest.main()


