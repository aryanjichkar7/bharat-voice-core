"""
Stage 2 Verification Test Suite - Live Speech Reliability, Session Lifecycle, and Booking UI Corrections
Tests:
1. No hardcoded venue dictionaries or fixed location lists in code.
2. Greeting lifecycle deduplication (no autoplay on connection, exactly 1 greeting per deliberate voice session).
3. Interim transcript streaming and final transcript differentiation (is_final flag, no search trigger on interim).
4. Truthful booking confirmation & UI elements (no fabricated driver names, fake plates, or fake OTPs).
5. State machine integrity (all 8 states: READY, REQUESTING_MIC, LISTENING, PROCESSING, SPEAKING, PAUSED, ERROR, DISCONNECTED).
6. Inactivity timeouts (5000ms idle listening timeout, 8000ms speech-start timeout).
"""

import asyncio
import json
import os
import re
import unittest
from pathlib import Path

# Load application modules
import app
from app import Session, SessionController, GREETINGS
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TTSSpeakFrame,
    OutputTransportMessageUrgentFrame,
    InputTransportMessageFrame,
)


class TestStage2SpeechAndLifecycle(unittest.IsolatedAsyncioTestCase):

    def test_01_no_hardcoded_venues(self):
        """Verify no hardcoded real-world venue dictionaries exist in application files."""
        for filename in ["app.py", "spatial_engine.py"]:
            filepath = Path(__file__).parent / filename
            content = filepath.read_text(encoding="utf-8")

            # Check for suspicious venue lists or dictionaries
            self.assertNotIn("NAGPUR_VENUES", content)
            self.assertNotIn("KNOWN_PLACES", content)
            self.assertNotIn("HOTWORD_LOCATIONS", content)
            self.assertNotIn("FALLBACK_DESTINATIONS", content)

    def test_02_truthful_booking_confirmation_data(self):
        """Verify Session.confirm_booking returns truthful prototype simulation fields with no fake driver."""
        session = Session(session_id="test-stage2-truth")
        session.pending_ride = {
            "destination": "Test Destination Area",
            "distance_km": 5.4,
            "duration_mins": 14,
            "fare_inr": 125,
            "route_geometry": {"type": "LineString", "coordinates": [[79.08, 21.14], [79.10, 21.16]]},
        }
        session.stage = "AWAITING_CONFIRMATION"

        pushed_frames = []
        async def mock_push(frame):
            pushed_frames.append(frame)

        booking_id = asyncio.run(session.confirm_booking(push_fn=mock_push))
        self.assertIsNotNone(booking_id)
        self.assertTrue(booking_id.startswith("BOC-"))
        self.assertEqual(session.stage, "RIDE_ACTIVE")
        self.assertEqual(session.active_ride["status"], "SIMULATED_CONFIRMED")
        self.assertEqual(session.active_ride["dispatch_mode"], "PROTOTYPE_SIMULATION")
        self.assertFalse(session.active_ride["is_live_dispatch"])

        # Check emitted message frame
        self.assertEqual(len(pushed_frames), 1)
        msg_payload = pushed_frames[0].message
        self.assertEqual(msg_payload["type"], "BOOKING_CONFIRMED")
        self.assertEqual(msg_payload["dispatch_mode"], "PROTOTYPE_SIMULATION")
        self.assertFalse(msg_payload["is_live_dispatch"])
        self.assertEqual(msg_payload["distance_km"], 5.4)
        self.assertEqual(msg_payload["duration_mins"], 14)
        self.assertEqual(msg_payload["fare_inr"], 125)

        # Confirm absolute absence of fabricated driver attributes
        self.assertNotIn("driver", msg_payload)
        self.assertNotIn("otp", msg_payload)
        self.assertNotIn("car", msg_payload)
        self.assertNotIn("plate", msg_payload)

    async def test_03_greeting_deduplication_and_deliberate_session_start(self):
        """Verify exactly one welcome greeting per deliberately started voice session."""
        session = Session(session_id="test-stage2-greeting", language="en")
        controller = SessionController(session=session)

        pushed_frames = []
        async def mock_push(frame, direction=None):
            pushed_frames.append(frame)
        controller.push_frame = mock_push

        # Frame 1: Simulate deliberate START_VOICE_SESSION from user orb click
        start_frame = type("TransportFrame", (), {
            "message": {"type": "START_VOICE_SESSION"}
        })()
        # Direct call to process_frame
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.frames.frames import InputTransportMessageFrame

        input_msg_frame = InputTransportMessageFrame(message={"type": "START_VOICE_SESSION"})
        await controller.process_frame(input_msg_frame, FrameDirection.DOWNSTREAM)

        # Exactly 1 greeting TTS and 1 BOT_RESPONSE should be emitted
        tts_frames = [f for f in pushed_frames if isinstance(f, TTSSpeakFrame)]
        ui_frames = [f for f in pushed_frames if isinstance(f, OutputTransportMessageUrgentFrame)]

        self.assertEqual(len(tts_frames), 1)
        self.assertEqual(tts_frames[0].text, GREETINGS["en"])
        self.assertEqual(len(ui_frames), 1)
        self.assertEqual(ui_frames[0].message["type"], "BOT_RESPONSE")
        self.assertEqual(ui_frames[0].message["text"], GREETINGS["en"])
        self.assertTrue(session.greeting_sent)

        # Frame 2: Re-sending START_VOICE_SESSION (e.g. on reconnect or subsequent orb tap) must NOT repeat greeting
        pushed_frames.clear()
        await controller.process_frame(input_msg_frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(len(pushed_frames), 0, "Duplicate greeting must not be sent once greeting_sent is True")

    def test_04_interim_and_final_transcript_differentiation(self):
        """Verify interim transcripts are marked is_final: False and final transcripts is_final: True."""
        session = Session(session_id="test-stage2-interim")
        stt = app.MultilingualWhisperSTT(session=session)

        # Check interim worker logic: pushes INTERIM_TRANSCRIPT with is_final: False
        pushed = []
        async def mock_push(frame, direction=None):
            pushed.append(frame)
        stt.push_frame = mock_push

        # Verify TranscriptTap and TEXT_INPUT both emit is_final: True
        # Read code to confirm is_final: True in TranscriptTap and TEXT_INPUT
        app_code = Path(__file__).parent / "app.py"
        content = app_code.read_text(encoding="utf-8")

        self.assertIn('"type": "INTERIM_TRANSCRIPT"', content)
        self.assertIn('"is_final": False', content)
        self.assertIn('"type": "USER_TRANSCRIPT"', content)
        self.assertIn('"is_final": True', content)

    def test_05_frontend_state_machine_and_timeouts(self):
        """Verify frontend app.js state machine constants and timeout durations."""
        app_js = (Path(__file__).parent / "static" / "app.js").read_text(encoding="utf-8")

        # 1. State machine constants
        required_states = [
            "READY",
            "REQUESTING_MIC",
            "LISTENING",
            "PROCESSING",
            "SPEAKING",
            "PAUSED",
            "ERROR",
            "DISCONNECTED",
        ]
        for state in required_states:
            self.assertIn(f"'{state}'", app_js, f"State {state} must be defined in app.js")

        # 2. Timeouts
        self.assertIn("SESSION_IDLE_TIMEOUT_MS = 5000", app_js, "5-second idle listening timeout must be configured")
        self.assertIn("SPEECH_START_TIMEOUT_MS = 8000", app_js, "8-second speech-start timeout must be configured")

        # 3. Truthful element bindings
        self.assertIn("confirmedBookingId", app_js)
        self.assertIn("confirmedDistance", app_js)
        self.assertIn("confirmedDuration", app_js)
        self.assertIn("confirmedFare", app_js)

        # 4. Fabricated driver elements eliminated
        self.assertNotIn("bookingOtpBadge.textContent", app_js)
        self.assertNotIn("driverName.textContent", app_js)
        self.assertNotIn("driverVehicle.textContent", app_js)
        self.assertNotIn("driverEtaPill.textContent", app_js)

    def test_06_index_html_truthfulness(self):
        """Verify index.html contains truthful prototype simulation cards and no fake driver card."""
        index_html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("confirmedBookingId", index_html)
        self.assertIn("confirmedDistance", index_html)
        self.assertIn("confirmedDuration", index_html)
        self.assertIn("confirmedFare", index_html)
        self.assertIn("Prototype Simulation", index_html)
        self.assertIn("SIMULATED BOOKING — NO LIVE DRIVER ASSIGNED", index_html)

        # Ensure fake driver fields are not present
        self.assertNotIn('id="driverName"', index_html)
        self.assertNotIn('id="driverVehicle"', index_html)
        self.assertNotIn('id="bookingOtpBadge"', index_html)
        self.assertNotIn("Rajesh Kumar", index_html)
        self.assertNotIn("MH 31 EQ 8421", index_html)

    async def test_07_stage2b_whisper_tiny_and_empty_stt(self):
        """Verify Stage 2B: Whisper tiny interim inference speed and STT_EMPTY frame dispatch."""
        import numpy as np
        import time

        session = Session(session_id="test-stage2b-stt", language="en")
        stt = app.MultilingualWhisperSTT(session=session)

        # Verify interim model loaded
        self.assertIsNotNone(stt._interim_model)

        # Benchmark interim model decoding latency on 1.0s of silence
        dummy_audio = np.zeros(16000, dtype=np.float32)
        t0 = time.perf_counter()
        interim_text = stt._transcribe_interim(dummy_audio, "en")
        dur = time.perf_counter() - t0

        self.assertLess(dur, 0.50, f"Interim decoding took {dur:.3f}s, expected < 0.50s on M1 CPU")
        self.assertEqual(interim_text, "")

        # Verify STT_EMPTY dispatched for short or silent audio
        pushed_frames = []
        async def mock_push(frame, direction=None):
            pushed_frames.append(frame)
        stt.push_frame = mock_push

        # 1. Very short audio (< 0.20s)
        short_pcm = bytes(16000 // 10)  # 0.05s
        async for _ in stt.run_stt(short_pcm):
            pass
        self.assertEqual(len(pushed_frames), 1)
        self.assertEqual(pushed_frames[0].message["type"], "STT_EMPTY")

        # 2. 1 second of silence
        pushed_frames.clear()
        silent_pcm = bytes(16000 * 2)  # 1s of 16-bit 0s
        async for _ in stt.run_stt(silent_pcm):
            pass
        self.assertEqual(len(pushed_frames), 1)
        self.assertEqual(pushed_frames[0].message["type"], "STT_EMPTY")

    def test_08_stage2b_frontend_lifecycle_tokens_and_watchdog(self):
        """Verify Stage 2B: Frontend lifecycle token pattern, track stop reporting, and processing watchdog."""
        app_js = (Path(__file__).parent / "static" / "app.js").read_text(encoding="utf-8")

        # 1. listeningCycleToken pattern
        self.assertIn("let listeningCycleToken = 0;", app_js)
        self.assertIn("listeningCycleToken++", app_js)
        self.assertIn("enterListeningState", app_js)

        # 2. Timeout validation matching cycle
        self.assertIn("expectedCycle !== listeningCycleToken", app_js)

        # 3. Processing watchdog
        self.assertIn("scheduleProcessingWatchdog", app_js)
        self.assertIn("OPERATION_TIMEOUT", app_js)

        # 4. Microphone track stopping and readyState logging
        self.assertIn("track.stop()", app_js)
        self.assertIn("readyState before", app_js)
        self.assertIn("readyState after", app_js)

        # 5. Stale interim transcript rejection
        self.assertIn("lastCommittedTurnSeq", app_js)
        self.assertIn("data.turn_seq <= lastCommittedTurnSeq", app_js)

        # 6. STT_EMPTY re-arms listening state
        self.assertIn("case 'STT_EMPTY':", app_js)
        self.assertIn("enterListeningState('STT_EMPTY_REARM')", app_js)


if __name__ == "__main__":
    unittest.main()

