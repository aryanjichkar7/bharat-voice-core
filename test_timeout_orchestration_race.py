"""
test_timeout_orchestration_race.py - Verification for Spoken-Turn Timeout Race Repair

Verifies:
1. BharatUserTurnStopStrategy suppresses user-speech timeout triggering while backend is processing a turn.
2. Slow destination search hits bounded deadline (7.0s) and produces exactly ONE error and ONE bot response, with zero premature fallback responses.
3. A subsequent user turn works properly after timeout recovery.
4. Destination confirmation and booking confirmation separation remains intact.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app import (
    Session,
    SessionController,
    TranscriptTap,
    BharatUserTurnStopStrategy,
    LLMContext,
    TURN_SPEECH_TIMEOUT,
)
import spatial_engine as se


class TestTimeoutOrchestrationRace(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.session = Session(session_id="test-timeout-race")
        self.session.lat = 21.1461
        self.session.lon = 79.0617
        self.session.stage = "AWAITING_DESTINATION"
        self.pushed_frames = []

        async def fake_push(frame, direction=None):
            self.pushed_frames.append(frame)

        self.controller = SessionController(
            session=self.session,
            context=LLMContext(messages=[]),
        )
        self.controller.push_frame = fake_push

        self.user_stop_strategy = BharatUserTurnStopStrategy(
            session=self.session,
            user_speech_timeout=TURN_SPEECH_TIMEOUT,
        )
        self.transcript_tap = TranscriptTap(
            session=self.session,
            controller=self.controller,
            user_stop_strategy=self.user_stop_strategy,
        )
        self.transcript_tap.push_frame = fake_push

    async def test_stop_strategy_suppressed_during_in_flight_processing(self):
        """Verifies BharatUserTurnStopStrategy does not fire when is_processing_turn is True."""
        self.session.is_processing_turn = True
        triggered = False

        async def fake_trigger():
            nonlocal triggered
            triggered = True

        self.user_stop_strategy.trigger_user_turn_stopped = fake_trigger

        self.user_stop_strategy._user_speech_wait_done = True
        self.user_stop_strategy._stt_wait_done = True
        self.user_stop_strategy._text = "Some transcript"

        await self.user_stop_strategy._maybe_trigger_user_turn_stopped()
        self.assertFalse(triggered, "User turn stop strategy must NOT trigger while is_processing_turn is True")

        # Now release processing lock and verify it can trigger normally
        self.session.is_processing_turn = False
        await self.user_stop_strategy._maybe_trigger_user_turn_stopped()
        self.assertTrue(triggered, "User turn stop strategy should trigger once processing is complete")

    async def test_slow_search_produces_single_recoverable_error_on_timeout(self):
        """Simulates a slow OSM search exceeding deadline and ensures exactly one terminal error response."""
        async def mock_slow_search(*args, **kwargs):
            await asyncio.sleep(10.0)  # Exceeds 7.0s deadline
            return {"status": "RESOLVED"}

        with patch("spatial_engine.evaluate_destination_intent", side_effect=mock_slow_search):
            with patch("asyncio.to_thread", side_effect=mock_slow_search):
                # Run through controller text turn
                await self.controller._execute_text_turn("Slow Location", "turn-001", 1)

                # Check lock was released
                self.assertFalse(self.session.is_processing_turn)

                # Check emitted frames
                msg_types = [
                    f.message.get("type")
                    for f in self.pushed_frames
                    if hasattr(f, "message")
                ]
                self.assertIn("ERROR", msg_types)
                self.assertIn("BOT_RESPONSE", msg_types)

                # Exactly ONE BOT_RESPONSE
                bot_responses = [
                    f.message for f in self.pushed_frames
                    if hasattr(f, "message") and f.message.get("type") == "BOT_RESPONSE"
                ]
                self.assertEqual(len(bot_responses), 1)
                self.assertIn("search took too long", bot_responses[0]["text"].lower())

    async def test_subsequent_turn_works_after_timeout(self):
        """Verifies session can accept a new destination request cleanly after a previous timeout."""
        # 1. First turn times out
        async def mock_slow_search(*args, **kwargs):
            await asyncio.sleep(10.0)
            return {"status": "RESOLVED"}

        with patch("asyncio.to_thread", side_effect=mock_slow_search):
            await self.controller._execute_text_turn("Slow Location", "turn-001", 1)

        self.assertFalse(self.session.is_processing_turn)
        self.assertEqual(self.session.stage, "AWAITING_DESTINATION")

        # 2. Second turn succeeds
        fast_candidate = {
            "name": "Fast Location",
            "lat": 21.1400,
            "lon": 79.0700,
            "distance_km": 2.0,
            "short_address": "Nearby Area",
        }
        mock_res = {
            "status": "EXACT",
            "resolution_type": "EXACT_MATCH",
            "primary_match": fast_candidate,
            "alternatives": [],
        }

        with patch("spatial_engine.evaluate_destination_intent", return_value=mock_res):
            with patch("spatial_engine.calculate_trip_details", return_value={
                "status": "OK",
                "destination": "Fast Location",
                "distance_km": 2.0,
                "duration_mins": 6.0,
                "fare_inr": 75.0,
                "route_geometry": "mock_polyline",
                "route_coords": [[21.1461, 79.0617], [21.1400, 79.0700]],
            }):
                self.session.turn_generation += 1
                curr_seq = self.session.turn_generation
                await self.controller._execute_text_turn("Fast Location", "turn-002", curr_seq)

                self.assertFalse(self.session.is_processing_turn)
                self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
                self.assertIsNotNone(self.session.pending_ride)
                self.assertEqual(self.session.pending_ride["fare_inr"], 75.0)

    async def test_two_phase_confirmation_separation_preserved(self):
        """Verifies candidate clarification -> destination confirmation -> booking confirmation."""
        # Phase 1: Clarification needed (POSSIBLE_MATCH)
        candidate = {
            "name": "Target Location",
            "lat": 21.1400,
            "lon": 79.0700,
            "distance_km": 1.5,
            "short_address": "Test Locality",
        }
        clarify_res = {
            "status": "POSSIBLE_MATCH",
            "resolution_type": "POSSIBLE_MATCH",
            "primary_match": candidate,
            "suggested_match": candidate,
            "clarification_prompt": "Did you mean Target Location?",
        }

        with patch("spatial_engine.evaluate_destination_intent", return_value=clarify_res):
            handled = await self.controller._handle_text_turn("Target Locashun")
            self.assertTrue(handled)
            self.assertEqual(self.session.stage, "AWAITING_DESTINATION_CONFIRMATION")
            self.assertIsNone(self.session.pending_ride)

        # Phase 2: User confirms destination -> presents route & fare proposal (does NOT book yet)
        with patch("spatial_engine.calculate_trip_details", return_value={
            "status": "OK",
            "destination": "Target Location",
            "distance_km": 1.5,
            "duration_mins": 5.0,
            "fare_inr": 60.0,
            "route_geometry": "mock_polyline",
            "route_coords": [[21.1461, 79.0617], [21.1400, 79.0700]],
        }):
            handled = await self.controller._handle_text_turn("Yes")
            self.assertTrue(handled)
            self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")
            self.assertIsNotNone(self.session.pending_ride)
            self.assertIsNone(self.session.active_ride)

        # Phase 3: User confirms booking -> booking confirmed
        handled = await self.controller._handle_text_turn("Book the cab please")
        self.assertTrue(handled)
        self.assertEqual(self.session.stage, "RIDE_ACTIVE")
        self.assertIsNotNone(self.session.active_ride)


if __name__ == "__main__":
    unittest.main()
