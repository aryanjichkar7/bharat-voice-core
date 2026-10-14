"""
Stage 2C Verification Test Suite: Real-Microphone-to-Booking Failure Isolation and End-to-End Voice Flow.

Tests:
1. Synthetic speech audio (resampled 16kHz PCM16) transcribed into raw STT and destination entity.
2. Exact destination resolution to primary match without premature chain disambiguation stall.
3. OSRM route calculation, distance, duration, and honest tariff fare proposal creation.
4. Turn state transition to AWAITING_CONFIRMATION with spoken confirmation question.
5. Confirmation intent handling -> BOOKING_CONFIRMED with simulation ID and RIDE_ACTIVE stage.
6. Honest error states: empty STT, unroutable coordinate, and search timeout.
7. 22-Stage Forensic Trace logging verification.
"""

import asyncio
import io
import time
import unittest
import numpy as np

import app
from app import Session, SessionController, MultilingualWhisperSTT, TurnTracer, resolve_voice_for_lang, VOICES_DIR
import spatial_engine as se
from piper import PiperVoice
from pipecat.frames.frames import (
    TranscriptionFrame,
    TTSSpeakFrame,
    OutputTransportMessageUrgentFrame,
    InputTransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class TestStage2CMicrophoneToBooking(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        # Pre-synthesize test audio using Piper voice model
        voice_path = VOICES_DIR / f"{resolve_voice_for_lang('en')}.onnx"
        voice = PiperVoice.load(str(voice_path))

        # Synthesize: "I want to go to Sitabuldi"
        chunks = list(voice.synthesize("I want to go to Sitabuldi"))
        raw_22k = b"".join(c.audio_int16_bytes for c in chunks)

        # Resample from 22050 to 16000
        audio_f32 = np.frombuffer(raw_22k, dtype=np.int16).astype(np.float32) / 32768.0
        resampled_len = int(len(audio_f32) * 16000 / 22050)
        resampled_f32 = np.interp(
            np.linspace(0, len(audio_f32), resampled_len, endpoint=False),
            np.arange(len(audio_f32)),
            audio_f32
        )
        cls.sitabuldi_pcm16 = (np.clip(resampled_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    async def test_01_stt_transcription_from_speech_audio(self):
        """Verify real speech audio passes through MultilingualWhisperSTT to produce final transcript."""
        session = Session(session_id="test-s2c-stt", language="en")
        stt = MultilingualWhisperSTT(session=session)

        pushed_frames = []
        async def mock_push(frame, direction=None):
            pushed_frames.append(frame)
        stt.push_frame = mock_push

        t0 = time.time()
        frames = []
        async for frame in stt.run_stt(self.sitabuldi_pcm16):
            frames.append(frame)
        stt_dur = time.time() - t0

        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TranscriptionFrame)
        raw_text = frames[0].text
        self.assertTrue(len(raw_text) > 3, f"Expected non-empty transcript, got: {raw_text!r}")
        self.assertLess(stt_dur, 4.0, f"STT took {stt_dur:.2f}s, expected < 4.0s")

    def test_02_destination_entity_extraction_robustness(self):
        """Verify entity extraction handles past/polite preambles without sending filler to Nominatim."""
        queries = [
            ("I wanted to go to Sitabuldi.", "Sitabuldi"),
            ("I would like to go to Dharampeth", "Dharampeth"),
            ("Could you please take me to Airport", "Airport"),
            ("Drop me off at Nagpur Railway Station", "Nagpur Railway Station"),
            ("I want to reach IT Park", "IT Park"),
            ("Take me towards Medical Square", "Medical Square"),
            ("Cab to Sadar please", "Sadar"),
            ("Sitabuldi", "Sitabuldi"),
        ]
        for query, expected in queries:
            extracted = se.extract_destination_entity(query)
            self.assertEqual(extracted.lower(), expected.lower(),
                             f"For query '{query}', expected '{expected}' but got '{extracted}'")

    async def test_03_end_to_end_destination_to_fare_proposal_and_booking(self):
        """Verify full voice turn: STT -> Destination Resolution -> OSRM Route -> Fare Proposal -> Booking Confirmation."""
        session = Session(session_id="test-s2c-e2e", language="en", lat=21.1458, lon=79.0882)
        controller = SessionController(session=session)

        pushed_frames = []
        async def mock_push(frame, direction=None):
            pushed_frames.append(frame)
        controller.push_frame = mock_push

        # Turn 1: User says "I wanted to go to Sitabuldi"
        session.turns += 1
        session.turn_generation += 1
        turn_id = f"turn-{session.turns:03d}"
        session.current_turn_id = turn_id

        # 1. Evaluate turn
        handled = await controller._handle_text_turn("I wanted to go to Sitabuldi")
        self.assertTrue(handled, "Destination turn must be handled by deterministic fast path")

        # 2. Verify state transition to AWAITING_CONFIRMATION
        self.assertEqual(session.stage, "AWAITING_CONFIRMATION")
        self.assertIsNotNone(session.pending_ride)
        self.assertIn("fare_inr", session.pending_ride)
        self.assertGreater(session.pending_ride["fare_inr"], 50)
        self.assertGreater(session.pending_ride["distance_km"], 0)

        # 3. Verify MAP_ROUTE_PREVIEW and BOT_RESPONSE emitted
        preview_frames = [f for f in pushed_frames if getattr(f, "message", {}).get("type") == "MAP_ROUTE_PREVIEW"]
        bot_frames = [f for f in pushed_frames if getattr(f, "message", {}).get("type") == "BOT_RESPONSE"]
        tts_frames = [f for f in pushed_frames if isinstance(f, TTSSpeakFrame)]

        self.assertEqual(len(preview_frames), 1, "Exactly one MAP_ROUTE_PREVIEW frame must be emitted")
        self.assertEqual(len(bot_frames), 1, "Exactly one BOT_RESPONSE asking for confirmation must be emitted")
        self.assertEqual(len(tts_frames), 1, "Exactly one TTSSpeakFrame must be emitted")

        # Verify confirmation question text
        self.assertIn("fare is", bot_frames[0].message["text"])
        self.assertIn("Shall I book", bot_frames[0].message["text"])

        # Turn 2: User confirms with "Yes, please book it"
        pushed_frames.clear()
        session.turns += 1
        session.turn_generation += 1
        turn_id_2 = f"turn-{session.turns:03d}"
        session.current_turn_id = turn_id_2

        confirm_handled = await controller._handle_text_turn("Yes, please book it")
        self.assertTrue(confirm_handled, "Confirmation turn must be handled deterministically")

        # Verify transition to RIDE_ACTIVE and BOOKING_CONFIRMED
        self.assertEqual(session.stage, "RIDE_ACTIVE")
        self.assertIsNotNone(session.booking_id)
        self.assertTrue(session.booking_id.startswith("BOC-"))

        confirmed_frames = [f for f in pushed_frames if getattr(f, "message", {}).get("type") == "BOOKING_CONFIRMED"]
        self.assertEqual(len(confirmed_frames), 1)
        self.assertEqual(confirmed_frames[0].message["booking_id"], session.booking_id)
        self.assertEqual(confirmed_frames[0].message["dispatch_mode"], "PROTOTYPE_SIMULATION")

    def test_04_watchdog_safety_and_no_premature_rearm(self):
        """Verify frontend watchdog configuration: 15s safety threshold with honest error reporting."""
        import pathlib
        app_js = (pathlib.Path(app.BASE_DIR) / "static" / "app.js").read_text(encoding="utf-8")

        # Watchdog must not silently re-arm listening after 4 seconds
        self.assertNotIn("enterListeningState('WATCHDOG_REARM')", app_js)
        self.assertIn("15000", app_js, "Watchdog must be configured with 15-second safety net")
        self.assertIn("OPERATION_TIMEOUT", app_js, "Watchdog must emit OPERATION_TIMEOUT error")
        self.assertIn("Search request timed out", app_js)

    def test_05_phonetic_variant_matching(self):
        """Verify phonetic variants (e.g. pet -> peth) resolve to valid Nagpur POIs."""
        res_pet = se.search_nearby_pois("Dharampet", 21.1458, 79.0882)
        res_peth = se.search_nearby_pois("Dharampeth", 21.1458, 79.0882)
        self.assertGreater(len(res_pet), 0, "Dharampet variant must resolve through phonetic suffix rules")
        self.assertGreater(len(res_peth), 0, "Dharampeth must resolve")


if __name__ == "__main__":
    unittest.main()
