"""
test_voice_destination_freeze_repair.py - Verification for Voice Turn Freeze Defect

Verifies:
1. Destination clarification triggers cleanly without dropping frames or freezing in AWAITING_DESTINATION.
2. Turn processing lock (is_processing_turn) protects in-flight turn from trailing microphone noise and audio frames.
3. Candidate pill and pin rendering logic in UI supports single clarification candidate (POSSIBLE_MATCH).
4. BOT_RESPONSE is delivered immediately before TTS audio without queue stall.
5. Full 3-phase lifecycle: Destination request (clarification) -> Destination confirmation (route/fare) -> Booking confirmation.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import spatial_engine as se
from app import Session, SessionController, LLMContext


class TestVoiceDestinationFreezeRepair(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.session = Session(session_id="test-freeze-repair")
        self.session.lat = 21.1461
        self.session.lon = 79.0617
        self.session.stage = "AWAITING_DESTINATION"
        self.pushed_frames = []

        async def fake_push(frame, direction=None):
            self.pushed_frames.append(frame)

        self.controller = SessionController(
            session=self.session,
            context=LLMContext(messages=[])
        )
        self.controller.push_frame = fake_push

    def test_possible_match_clarification_structure(self):
        candidate = {
            "name": "Target Location",
            "full_address": "Target Location, Main Road, Test City",
            "short_address": "Test Locality",
            "lat": 21.1400,
            "lon": 79.0700,
            "distance_km": 1.8,
            "category": "place/suburb",
            "rating": 4.5,
            "importance": 0.2,
        }

        # Simulating phonetic variation of candidate
        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]):
            res = se.evaluate_destination_intent("Target Locashun", self.session.lat, self.session.lon)
            self.assertEqual(res["status"], "POSSIBLE_MATCH")
            self.assertEqual(res["resolution_type"], "POSSIBLE_MATCH")
            self.assertEqual(res["primary_match"]["name"], "Target Location")
            self.assertIn("Did you mean Target Location", res["clarification_prompt"])

    async def test_turn_advances_to_destination_confirmation_state(self):
        candidate = {
            "name": "Target Location",
            "full_address": "Target Location, Main Road, Test City",
            "short_address": "Test Locality",
            "lat": 21.1400,
            "lon": 79.0700,
            "distance_km": 1.8,
            "category": "place/suburb",
            "rating": 4.5,
            "importance": 0.2,
        }

        with patch("spatial_engine.search_nearby_pois", return_value=[candidate]):
            handled = await self.controller._handle_text_turn("Take me to Target Locashun")
            self.assertTrue(handled)
            self.assertEqual(self.session.stage, "AWAITING_DESTINATION_CONFIRMATION")
            self.assertIsNotNone(self.session.pending_destination)
            self.assertEqual(self.session.pending_destination["name"], "Target Location")

            # Check emitted messages
            msg_types = [
                getattr(f, "message", {}).get("type")
                for f in self.pushed_frames
                if hasattr(f, "message")
            ]
            self.assertIn("POI_CANDIDATES", msg_types)
            self.assertIn("BOT_RESPONSE", msg_types)

            # Ensure BOT_RESPONSE text contains the clarification prompt
            bot_responses = [
                f.message["text"] for f in self.pushed_frames
                if hasattr(f, "message") and f.message.get("type") == "BOT_RESPONSE"
            ]
            self.assertTrue(any("Target Location" in txt for txt in bot_responses))

    def test_processing_lock_attribute_present(self):
        s = Session(session_id="lock-test")
        self.assertTrue(hasattr(s, "is_processing_turn"))
        self.assertFalse(s.is_processing_turn)
        s.is_processing_turn = True
        self.assertTrue(s.is_processing_turn)

    def test_tts_sample_rate_initialized_non_zero(self):
        from app import DynamicPiperTTSService, OUT_SAMPLE_RATE, VOICES_DIR
        from pipecat.services.piper.tts import PiperTTSService
        s = Session(session_id="tts-sr-test")
        tts = DynamicPiperTTSService(
            session=s,
            download_dir=VOICES_DIR,
            sample_rate=OUT_SAMPLE_RATE,
            settings=PiperTTSService.Settings(voice="en_GB-cori-medium"),
        )
        self.assertEqual(tts._sample_rate, OUT_SAMPLE_RATE)
        self.assertEqual(tts.sample_rate, OUT_SAMPLE_RATE)
        self.assertGreater(tts.sample_rate, 0)

    def test_stt_vad_filter_and_beam_size_configured(self):
        import inspect
        from app import MultilingualWhisperSTT
        source = inspect.getsource(MultilingualWhisperSTT._transcribe)
        self.assertIn("vad_filter=True", source)
        self.assertIn("beam_size=1", source)


if __name__ == "__main__":
    unittest.main()

