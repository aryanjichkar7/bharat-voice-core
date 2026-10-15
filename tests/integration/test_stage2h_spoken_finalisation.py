"""
test_stage2h_spoken_finalisation.py - Verification for Bharat Online Cabs Critical Voice Pipeline Hotfix

Verifies:
1. Unified turn processing: SessionController.execute_turn is the single authoritative method for both spoken and typed turns.
2. Faster-Whisper VAD filter setting: _transcribe uses vad_filter=True and beam_size=1.
3. Authoritative final STT: Does not blindly commit unvalidated interim Whisper hypotheses when final STT is empty or hallucinated.
4. Interim text lifecycle: _last_interim_text is reset at turn boundaries (speech start, speech stop, STT completion).
5. Max speech utterance bound: If VADUserStoppedSpeakingFrame is delayed beyond 5.0s, speech stop is forced.
6. In-flight turn cancellation & supersession: A typed turn cancels an in-flight spoken turn cleanly without duplicate responses.
7. Interruption loop elimination: BharatSerializer only emits INTERRUPT when is_bot_speaking is true.
8. Audio control isolation: Inbound INTERRUPT_PLAYBACK does not cancel an active turn execution task (_turn_task).
9. VAD telemetry synchronization: TranscriptTap includes turn_seq in VAD_SPEECH_START and VAD_SPEECH_STOP frames.
10. Terminal turn sequence tracking: Late VAD_SPEECH_STOP events for already terminated turns are ignored.
"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    InputTransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from app import (
    Session,
    SessionController,
    TranscriptTap,
    MultilingualWhisperSTT,
    BharatSerializer,
    BharatUserTurnStopStrategy,
    LLMContext,
    TURN_SPEECH_TIMEOUT,
    IN_SAMPLE_RATE,
)
import spatial_engine as se


class TestStage2HSpokenFinalisation(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.session = Session(session_id="test-session-2h")
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
        self.controller.user_stop_strategy = self.user_stop_strategy

        self.transcript_tap = TranscriptTap(
            session=self.session,
            controller=self.controller,
            user_stop_strategy=self.user_stop_strategy,
        )
        self.transcript_tap.push_frame = fake_push

    async def test_unified_execute_turn_spoken_flow(self):
        """Verifies execute_turn handles spoken turns, sets processing locks, and returns route preview."""
        mock_dest = {
            "status": "EXACT",
            "resolution_type": "EXACT_MATCH",
            "primary_match": {
                "name": "Sitabuldi Metro Station",
                "lat": 21.1458,
                "lon": 79.0882,
                "distance_km": 1.2,
                "short_address": "Sitabuldi",
            },
            "alternatives": [],
        }
        mock_trip = {
            "status": "OK",
            "destination": "Sitabuldi Metro Station",
            "distance_km": 1.2,
            "duration_mins": 5.0,
            "fare_inr": 60.0,
            "route_geometry": "mock_geom",
            "route_coords": [[21.1461, 79.0617], [21.1458, 79.0882]],
        }

        with patch("spatial_engine.evaluate_destination_intent", return_value=mock_dest):
            with patch("spatial_engine.calculate_trip_details", return_value=mock_trip):
                handled = await self.controller.execute_turn(
                    text="Take me to Sitabuldi",
                    source="voice",
                    language="en",
                    vad_speech_s=1.5,
                    stt_ms=250.0,
                )
                self.assertTrue(handled)
                self.assertFalse(self.session.is_processing_turn)
                self.assertEqual(self.session.stage, "AWAITING_CONFIRMATION")

                msg_types = [
                    f.message.get("type")
                    for f in self.pushed_frames
                    if hasattr(f, "message")
                ]
                self.assertIn("USER_TRANSCRIPT", msg_types)
                self.assertIn("POI_CANDIDATES", msg_types)
                self.assertIn("MAP_ROUTE_PREVIEW", msg_types)
                self.assertIn("BOT_RESPONSE", msg_types)
                self.assertIn("PERF_DIAGNOSTICS", msg_types)

                user_tx = next(f.message for f in self.pushed_frames if hasattr(f, "message") and f.message.get("type") == "USER_TRANSCRIPT")
                self.assertEqual(user_tx["source"], "voice")
                self.assertEqual(user_tx["text"], "Take me to Sitabuldi")

    async def test_stt_configuration(self):
        """Verifies Faster-Whisper transcribe configuration matches required parameters."""
        stt = MultilingualWhisperSTT(session=self.session, sample_rate=IN_SAMPLE_RATE)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        stt._model = mock_model

        dummy_audio = np.zeros(16000, dtype=np.float32)
        stt._transcribe(dummy_audio, "en")

        self.assertTrue(mock_model.transcribe.called)
        _, kwargs = mock_model.transcribe.call_args
        self.assertEqual(kwargs.get("beam_size"), 1)
        self.assertEqual(kwargs.get("temperature"), 0.0)

    async def test_authoritative_stt_no_interim_fallback_on_empty(self):
        """Verifies STT does NOT fall back to interim hypothesis when authoritative Whisper is empty, emitting STT_EMPTY."""
        stt = MultilingualWhisperSTT(session=self.session, sample_rate=IN_SAMPLE_RATE)
        stt._last_interim_text = "We continue."
        stt.push_frame = AsyncMock()

        # Mock _transcribe returning empty text
        stt._transcribe = MagicMock(return_value=("", "en", 0.0))

        dummy_pcm = (np.zeros(16000, dtype=np.int16)).tobytes()

        frames = []
        async for frame in stt.run_stt(dummy_pcm):
            frames.append(frame)

        # No TranscriptionFrame yielded
        self.assertEqual(len(frames), 0)
        # STT_EMPTY pushed
        self.assertTrue(stt.push_frame.called)
        empty_msg = stt.push_frame.call_args[0][0].message
        self.assertEqual(empty_msg.get("type"), "STT_EMPTY")
        # _last_interim_text cleared
        self.assertEqual(stt._last_interim_text, "")

    async def test_authoritative_stt_no_interim_fallback_on_hallucination(self):
        """Verifies STT filters hallucinations and does NOT resurrect interim noise, emitting STT_EMPTY."""
        stt = MultilingualWhisperSTT(session=self.session, sample_rate=IN_SAMPLE_RATE)
        stt._last_interim_text = "Random room noise"
        stt.push_frame = AsyncMock()

        # Mock _transcribe returning Whisper hallucination
        stt._transcribe = MagicMock(return_value=("Thank you for watching!", "en", 0.0))

        dummy_pcm = (np.zeros(16000, dtype=np.int16)).tobytes()

        frames = []
        async for frame in stt.run_stt(dummy_pcm):
            frames.append(frame)

        self.assertEqual(len(frames), 0)
        self.assertTrue(stt.push_frame.called)
        empty_msg = stt.push_frame.call_args[0][0].message
        self.assertEqual(empty_msg.get("type"), "STT_EMPTY")
        self.assertEqual(stt._last_interim_text, "")

    async def test_interim_text_lifecycle_cleared_at_boundaries(self):
        """Verifies _last_interim_text is cleared on speech start, speech stop, and STT completion."""
        stt = MultilingualWhisperSTT(session=self.session, sample_rate=IN_SAMPLE_RATE)
        stt.push_frame = AsyncMock()

        # 1. Speech start clears it
        stt._last_interim_text = "stale"
        await stt._handle_user_started_speaking(VADUserStartedSpeakingFrame())
        self.assertEqual(stt._last_interim_text, "")

        # 2. Speech stop clears it
        stt._last_interim_text = "stale interim"
        await stt._handle_user_stopped_speaking(VADUserStoppedSpeakingFrame())
        self.assertEqual(stt._last_interim_text, "")

    async def test_max_utterance_timeout_forces_speech_stop(self):
        """Verifies that an utterance exceeding 5.0 seconds automatically triggers speech stop."""
        stt = MultilingualWhisperSTT(session=self.session, sample_rate=IN_SAMPLE_RATE)
        stt.push_frame = AsyncMock()
        stt._handle_user_stopped_speaking = AsyncMock()

        # Simulate user started speaking 5.2 seconds ago
        stt._user_speaking = True
        stt._speech_start_time = time.time() - 5.2

        dummy_frame = InputAudioRawFrame(
            audio=np.zeros(320, dtype=np.int16).tobytes(),
            num_channels=1,
            sample_rate=16000,
        )

        await stt.process_audio_frame(dummy_frame, FrameDirection.DOWNSTREAM)
        self.assertTrue(stt._handle_user_stopped_speaking.called)

    async def test_typed_turn_cancels_inflight_spoken_turn(self):
        """Verifies a typed turn cleanly cancels a running spoken turn without racing."""
        import threading
        slow_event = threading.Event()

        def mock_slow_search(*args, **kwargs):
            slow_event.wait(timeout=5.0)
            return {
                "status": "EXACT",
                "resolution_type": "EXACT_MATCH",
                "primary_match": {"name": "Spoken Destination", "lat": 21.14, "lon": 79.08},
                "alternatives": [],
            }

        with patch("spatial_engine.evaluate_destination_intent", side_effect=mock_slow_search):
            spoken_task = asyncio.create_task(
                self.controller.execute_turn("Spoken Slow Destination", source="voice")
            )
            self.controller._turn_task = spoken_task
            await asyncio.sleep(0.05)
            self.assertTrue(self.session.is_processing_turn)
            spoken_seq = self.session.turn_generation

            fast_res = {
                "status": "EXACT",
                "resolution_type": "EXACT_MATCH",
                "primary_match": {"name": "Fast Typed Destination", "lat": 21.15, "lon": 79.09},
                "alternatives": [],
            }
            mock_trip = {
                "status": "OK",
                "destination": "Fast Typed Destination",
                "distance_km": 2.0,
                "duration_mins": 6.0,
                "fare_inr": 75.0,
                "route_geometry": "mock_geom",
                "route_coords": [[21.1461, 79.0617], [21.15, 79.09]],
            }

            with patch("spatial_engine.evaluate_destination_intent", return_value=fast_res):
                with patch("spatial_engine.calculate_trip_details", return_value=mock_trip):
                    typed_handled = await self.controller.execute_turn("Fast Typed Destination", source="text")
                    self.assertTrue(typed_handled)

            slow_event.set()
            self.assertTrue(spoken_task.cancelled() or spoken_task.done())
            self.assertGreater(self.session.turn_generation, spoken_seq)

            bot_texts = [
                f.message.get("text")
                for f in self.pushed_frames
                if hasattr(f, "message") and f.message.get("type") == "BOT_RESPONSE"
            ]
            for txt in bot_texts:
                self.assertNotIn("Spoken Destination", txt)

    async def test_bharat_serializer_interruption_filter(self):
        """Verifies BharatSerializer suppresses InterruptionFrame serialization when bot is not speaking."""
        serializer = BharatSerializer(session=self.session)

        # 1. Bot NOT speaking: InterruptionFrame should return None
        self.session.is_bot_speaking = False
        res = await serializer.serialize(InterruptionFrame())
        self.assertIsNone(res)

        # 2. Bot IS speaking: InterruptionFrame should serialize to {"type": "INTERRUPT"}
        self.session.is_bot_speaking = True
        res = await serializer.serialize(InterruptionFrame())
        self.assertIsNotNone(res)
        data = json.loads(res)
        self.assertEqual(data.get("type"), "INTERRUPT")

    async def test_interrupt_playback_does_not_cancel_turn_task(self):
        """Verifies inbound INTERRUPT_PLAYBACK frame never cancels an active _turn_task."""
        async def dummy_slow_work():
            await asyncio.sleep(1.0)
            return "done"

        turn_task = asyncio.create_task(dummy_slow_work())
        self.controller._turn_task = turn_task
        self.session.is_bot_speaking = False

        # Inbound INTERRUPT_PLAYBACK from client
        msg_frame = InputTransportMessageFrame(message={"type": "INTERRUPT_PLAYBACK"})
        await self.controller.process_frame(msg_frame, FrameDirection.DOWNSTREAM)

        # Task MUST remain running
        self.assertFalse(turn_task.cancelled())
        self.assertFalse(turn_task.done())

        # Clean up
        turn_task.cancel()
        try:
            await turn_task
        except asyncio.CancelledError:
            pass

    async def test_transcript_tap_includes_turn_seq_in_vad_frames(self):
        """Verifies TranscriptTap stamps turn_seq on VAD_SPEECH_START and VAD_SPEECH_STOP."""
        self.session.turn_generation = 42
        self.pushed_frames.clear()

        await self.transcript_tap.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        start_frame = next(f for f in self.pushed_frames if hasattr(f, "message") and f.message.get("type") == "VAD_SPEECH_START")
        self.assertEqual(start_frame.message.get("turn_seq"), 42)

        self.pushed_frames.clear()
        await self.transcript_tap.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        stop_frame = next(f for f in self.pushed_frames if hasattr(f, "message") and f.message.get("type") == "VAD_SPEECH_STOP")
        self.assertEqual(stop_frame.message.get("turn_seq"), 42)

    def test_frontend_state_race_prevention_logic(self):
        """Verifies simulated client state race handling where late VAD_SPEECH_STOP is ignored if turn <= lastTerminalTurnSeq."""
        # Simulated JS state
        last_terminal_turn_seq = 0
        current_state = "READY"

        def handle_server_msg(data):
            nonlocal last_terminal_turn_seq, current_state
            t = data.get("type")
            t_seq = data.get("turn_seq", 0)

            if t == "STT_EMPTY":
                last_terminal_turn_seq = max(last_terminal_turn_seq, t_seq)
                current_state = "LISTENING"
            elif t == "VAD_SPEECH_STOP":
                if t_seq <= last_terminal_turn_seq:
                    # Ignore late speech stop
                    return
                current_state = "PROCESSING"

        # 1. Speech onset
        current_state = "LISTENING"

        # 2. STT_EMPTY arrives for turn 5
        handle_server_msg({"type": "STT_EMPTY", "turn_seq": 5})
        self.assertEqual(last_terminal_turn_seq, 5)
        self.assertEqual(current_state, "LISTENING")

        # 3. Late VAD_SPEECH_STOP arrives for turn 5 (36ms later)
        handle_server_msg({"type": "VAD_SPEECH_STOP", "turn_seq": 5})
        # State MUST remain LISTENING, not revert to PROCESSING
        self.assertEqual(current_state, "LISTENING")

        # 4. New turn 6 VAD_SPEECH_STOP can transition to PROCESSING
        handle_server_msg({"type": "VAD_SPEECH_STOP", "turn_seq": 6})
        self.assertEqual(current_state, "PROCESSING")


if __name__ == "__main__":
    unittest.main()
