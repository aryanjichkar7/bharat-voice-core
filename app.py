"""
Bharat Online Cabs - Enterprise Full-Duplex Voice Server & Mobility Dispatch.

Architecture:
- Deterministic State Machine for Routing, Route Rendering, and Ride Confirmation.
- Persistent Post-Booking Lifecycle (RIDE_ACTIVE) with Driver Tracking, OTP Support & Cancellation.
- Acoustic Echo Muting & Whisper Silence Hallucination Filtering.
- Faster-Whisper (small, int8) with Language Biasing (en, hi, mr).
- Dynamic Piper Neural TTS (en_GB-cori-medium, hi_IN-priyamvada-medium).
- WebSocket Protocol with Immediate CONNECTION_STATUS Handshake and Telemetry Sync.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper import tts as piper_tts_module
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

import spatial_engine as se

# Load .env
load_dotenv()

# ----------------------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("BOC_LLM_MODEL", "qwen2.5:3b")
WHISPER_MODEL = os.getenv("BOC_WHISPER_MODEL", "small")
MAX_SESSIONS = int(os.getenv("BOC_MAX_SESSIONS", "8"))

IN_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 22050          # Piper medium voices are native 22.05 kHz -> no resampling
VAD_STOP_SECS = 0.9               # Natural conversational pause (0.8-1.2s) before speech segment closes
TURN_SPEECH_TIMEOUT = 0.15        # Turn-end buffer
ALLOWED_LANGS = ("en", "hi", "mr")
STT_HOTWORDS = "Bharat Online Cabs cab booking ride pickup destination OTP driver Nagpur"

VOICE_MAP = {
    "en": "en_GB-cori-medium",
    "hi": "hi_IN-priyamvada-medium",
    "mr": "mr_IN-google-medium",
}

GREETINGS = {
    "en": "Hello! Welcome to Bharat Online Cabs. Where in Nagpur would you like to go?",
    "hi": "नमस्ते! भारत ऑनलाइन कैब्स में आपका स्वागत है। आप नागपुर में कहाँ जाना चाहेंगे?",
    "mr": "नमस्कार! भारत ऑनलाईन कॅब्समध्ये आपले स्वागत आहे. आपण नागपूरमध्ये कुठे जाऊ इच्छिता?",
}

CONFIRMATION_KEYWORDS = (
    "yes", "confirm", "book", "ok", "okay", "sure", "yep", "yeah",
    "haan", "ha", "ho", "chalo", "theek", "sahi", "kar do", "book it", "please", "done"
)


def is_cancellation_intent(text: str) -> bool:
    """Strictly matches cancellation intent while rejecting confirmation tokens."""
    t = text.lower().strip()
    if any(p in t for p in ("रद्द", "कॅन्सल", "कन्सल", "कैंसिल", "नको", "मत करो")):
        return True
    words = set(re.findall(r"\b[a-zA-Z']+\b", t))
    if words & {"cancel", "cancellation", "radd", "abort", "stop", "band", "nako"}:
        return True
    return any(p in t for p in (
        "cancel ride", "cancel the ride", "cancel my ride", "cancel booking",
        "cancel the booking", "radd kara", "band karo", "stop booking",
        "don't book", "dont book", "do not book", "never mind", "mat karo",
        "nahi chahiye", "nako aahe"
    ))


def is_confirmation_intent(text: str) -> bool:
    """Matches affirmative booking/proposal confirmation, explicitly rejecting negation, cancellation, and destination utterances."""
    t = text.lower().strip()
    if is_cancellation_intent(t):
        return False
    if any(neg in t for neg in ("don't", "dont", "no ", "nahi", "nako", "not now", "wait", "नको", "नाही", "मत")):
        return False
    # If the text carries a destination preposition or destination target, it is a ride request, NOT a bare confirmation
    if re.search(r"\b(?:to|towards|reach|drop\s+(?:me\s+)?at|heading\s+for)\s+[a-zA-Z0-9\u0900-\u097f]{2,}", t):
        return False
    if any(p in t for p in ("हो", "होय", "नक्की", "करा", "बुक करा", "हाँ", "हा", "कर दो", "ठीक आहे", "ठीक है", "पुष्टी")):
        return True
    words = set(re.findall(r"\b[a-zA-Z']+\b", t))
    if words & {"yes", "confirm", "sure", "yep", "yeah", "haan", "ha", "ho", "done", "correct", "right", "proceed"}:
        return True
    if any(p in t for p in ("book it", "book this", "please book", "book cab", "book a cab", "book the cab", "confirm booking", "confirm ride", "kar do", "theek hai", "chalo", "sahi hai", "that's right", "thats right", "go ahead")):
        return True
    return bool("book" in words and "cancel" not in words and len(words) <= 5)


WHISPER_HALLUCINATIONS = {
    "rejection", "thank you", "subtitles", "watching", "subscribe",
    "thank you for watching", "thank you for watching.", "thanks for watching",
    "thank you for watching!", "subtitles by", "please subscribe",
    "thank you.", "everything", "you", "bye", "bye.", "bye bye",
    "transcription by", "translated by", "watching", "rejection."
}


def resolve_voice_for_lang(lang: str) -> str:
    target = VOICE_MAP.get(lang, "en_GB-cori-medium")
    if (VOICES_DIR / f"{target}.onnx").exists():
        return target
    available = sorted(p.stem for p in VOICES_DIR.glob("*.onnx"))
    return available[0] if available else target


def get_system_prompt(lang: str = "en") -> str:
    lang_rules = {
        "en": "LANGUAGE: Indian English. Speak concisely in fluent English.",
        "hi": "LANGUAGE: Hindi. Speak in simple, clear Hindi written in Devanagari script.",
        "mr": "LANGUAGE: Marathi. Speak in polite, clear Marathi written in Devanagari script.",
    }.get(lang, "LANGUAGE: English.")

    return f"""You are Disha, the elite voice concierge of Bharat Online Cabs in {se.DEFAULT_CITY}, India.
You are on a live voice call. Everything you write is spoken aloud by a text-to-speech engine.

{lang_rules}

STYLE & CONSTRAINTS
- Reply in 1 or 2 concise, spoken sentences. Never use bullet points, lists, markdown, asterisks, emojis or symbols.
- Speak money as "rupees" (or "रुपये"), never write the ₹ symbol.
- The customer's GPS location is already known; never ask for it.
- Never invent places, distances, fares, driver details, or booking outcomes. Rely exclusively on verified tool outputs and session state.
- If asked unrelated questions, politely guide the customer back to cab booking in Nagpur.

BOOKING WORKFLOW & TOOLS
1. When the customer mentions any destination or category (e.g. jewelry shop, restaurant, landmark), call search_nearby_pois with that query.
2. If search_nearby_pois returns DISAMBIGUATION_NEEDED:
   Concierge Rule: Announce ONLY the single closest branch and its distance using the next_step instruction (e.g., "The closest option is [Name] in [Area], about [X] meters away. Would you like to go there, or check the other options on your map?").
3. When trip details are calculated, announce: "The trip is [X] km and the estimated fare is [Y] rupees. Shall I book this ride?"
4. When the customer confirms, confirm the booking in one short sentence."""


# ----------------------------------------------------------------------------- disambiguation matching
def match_candidate_choice(text: str, options: list[dict]) -> dict | None:
    if not options:
        return None
    t = text.lower().strip()

    # 1. Direct ordinals or "closest / nearest / first"
    if any(k in t for k in ("first", "closest", "nearest", "paas", "pehla", "pehli", "option 1", "number 1", "choice 1", "first option", "1st", "1", "one")):
        return options[0]
    if any(k in t for k in ("second", "dusra", "dusri", "option 2", "number 2", "choice 2", "second option", "2nd", "2", "two")) and len(options) > 1:
        return options[1]
    if any(k in t for k in ("third", "teesra", "teesri", "option 3", "number 3", "choice 3", "third option", "3rd", "3", "three")) and len(options) > 2:
        return options[2]

    # 2. Token overlap with address or candidate name
    best_match = None
    max_score = 0
    for opt in options:
        haystack = f"{opt.get('name', '')} {opt.get('short_address', '')} {opt.get('full_address', '')}".lower()
        clean_words = re.findall(r'[a-zA-Z]{4,}', t)
        matches = sum(1 for w in clean_words if w in haystack)
        if matches > max_score:
            max_score = matches
            best_match = opt

    if best_match and max_score > 0:
        return best_match

    # 3. Simple affirmative agreement to concierge's closest recommendation
    affirmative = (
        "yes", "ha", "haan", "sure", "ok", "okay", "chalo", "theek", "kar do",
        "take me there", "go there", "book it", "book this", "please", "done",
        "confirm", "let's go", "yep", "yeah", "ho", "sahi", "fine", "take me",
        "closest", "nearest", "first"
    )
    if is_confirmation_intent(t) or any(k == t or t.startswith(k + " ") or t.endswith(" " + k) or (" " + k + " ") in t for k in affirmative):
        return options[0]

    return None


class TurnTracer:
    """
    22-Stage Forensic Trace Pipeline for Bharat Online Cabs.
    Tracks each voice turn across acoustic, phonetic, intent, spatial, and speech synthesis lifecycles.
    """
    @staticmethod
    def trace(stage_num: int, name: str, session_id: str, turn_id: str, **kwargs):
        ts = time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"[{ts}][STAGE {stage_num:02d}][{name}] [{session_id}][{turn_id}] {details}")


# ----------------------------------------------------------------------------- shared models
_whisper_model = None
_interim_whisper_model = None
_piper_cache: dict[str, object] = {}


def get_whisper_model():
    """Load faster-whisper authoritative model once per process (int8 on CPU is fast & memory-efficient on M1)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        t0 = time.time()
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
        logger.info(f"Authoritative Whisper '{WHISPER_MODEL}' loaded in {time.time() - t0:.1f}s")
    return _whisper_model


def get_interim_whisper_model():
    """Load streaming Faster-Whisper model for ultra-low-latency live interim hypotheses (200-300ms)."""
    global _interim_whisper_model
    if _interim_whisper_model is None:
        from faster_whisper import WhisperModel

        t0 = time.time()
        interim_model = os.getenv("WHISPER_INTERIM_MODEL", "tiny")
        _interim_whisper_model = WhisperModel(interim_model, device="cpu", compute_type="int8", cpu_threads=2)
        logger.info(f"Streaming Interim Whisper '{interim_model}' loaded in {time.time() - t0:.1f}s")
    return _interim_whisper_model


class _CachedPiperVoice:
    """Drop-in for piper.PiperVoice used by PiperTTSService: memoizes loaded ONNX voices."""

    @staticmethod
    def load(model_path, use_cuda: bool = False, **kwargs):
        from piper import PiperVoice

        key = str(model_path)
        if key not in _piper_cache:
            _piper_cache[key] = PiperVoice.load(model_path, use_cuda=use_cuda, **kwargs)
            logger.info(f"Piper voice loaded: {Path(key).name}")
        return _piper_cache[key]


piper_tts_module.PiperVoice = _CachedPiperVoice  # share voices across sessions


# ----------------------------------------------------------------------------- session state
@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turn_generation: int = 0
    current_turn_id: str = "turn-000"
    lat: float = se.DEFAULT_LAT
    lon: float = se.DEFAULT_LON
    gps_source: str = "default"
    language: str = "en"
    stage: str = "GREETING"  # GREETING > AWAITING_DESTINATION > AWAITING_DESTINATION_CONFIRMATION > DISAMBIGUATING > AWAITING_CONFIRMATION > RIDE_ACTIVE
    is_bot_speaking: bool = False
    is_processing_turn: bool = False
    last_search: dict | None = None
    pending_destination: dict | None = None
    pending_ride: dict | None = None
    active_ride: dict | None = None
    booking_id: str | None = None
    dispatch_mode: str = "PROTOTYPE_SIMULATION"
    is_live_dispatch: bool = False
    turns: int = 0
    greeting_sent: bool = False
    created_at: float = field(default_factory=time.time)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def public(self) -> dict:
        d = asdict(self)
        if d["pending_ride"]:
            d["pending_ride"] = {k: v for k, v in d["pending_ride"].items() if k != "route_geometry"}
        d.pop("last_search", None)
        d.pop("pending_destination", None)
        d.pop("session_lock", None)
        return d

    def clear_conversation(self) -> bool:
        """Invalidates conversational turn, resets dialogue context while preserving an active ride."""
        self.turn_generation += 1
        self.last_search = None
        self.pending_destination = None
        if self.stage in ("GREETING", "DISAMBIGUATION", "DISAMBIGUATING", "AWAITING_DESTINATION_CONFIRMATION", "AWAITING_CONFIRMATION", "BOOKING_IN_PROGRESS"):
            self.stage = "AWAITING_DESTINATION"
            self.pending_ride = None
        return self.stage == "RIDE_ACTIVE"

    async def set_route_preview(self, trip: dict, dest_lat: float, dest_lon: float, push_fn=None):
        self.pending_ride = {**trip, "dest_lat": dest_lat, "dest_lon": dest_lon}
        self.stage = "AWAITING_CONFIRMATION"
        if push_fn:
            await push_fn(ui_message({
                "type": "MAP_ROUTE_PREVIEW",
                "destination": trip["destination"],
                "distance_km": trip["distance_km"],
                "duration_mins": trip["duration_mins"],
                "fare_inr": trip["fare_inr"],
                "route_geometry": trip["route_geometry"],
                "origin": {"lat": self.lat, "lon": self.lon},
                "dest": {"lat": dest_lat, "lon": dest_lon},
                "route_source": trip.get("route_source", "osrm"),
                "turn_id": self.current_turn_id,
                "turn_seq": self.turn_generation,
            }))

    async def confirm_booking(self, push_fn=None) -> str | None:
        async with self.session_lock:
            if not self.pending_ride:
                return None
            if self.stage == "RIDE_ACTIVE" and self.booking_id:
                return self.booking_id
            if not self.booking_id:
                self.booking_id = f"BOC-{uuid.uuid4().hex[:6].upper()}"
            self.stage = "RIDE_ACTIVE"
            self.active_ride = {
                **self.pending_ride,
                "booking_id": self.booking_id,
                "status": "SIMULATED_CONFIRMED",
                "dispatch_mode": "PROTOTYPE_SIMULATION",
                "is_live_dispatch": False,
                "confirmed_at": time.time(),
            }
            if push_fn:
                ride = self.pending_ride
                await push_fn(ui_message({
                    "type": "BOOKING_CONFIRMED",
                    "booking_id": self.booking_id,
                    "destination": ride["destination"],
                    "fare_inr": ride["fare_inr"],
                    "distance_km": ride["distance_km"],
                    "duration_mins": ride["duration_mins"],
                    "dispatch_mode": "PROTOTYPE_SIMULATION",
                    "is_live_dispatch": False,
                    "turn_id": self.current_turn_id,
                    "turn_seq": self.turn_generation,
                }))
            return self.booking_id

    async def cancel_ride(self, push_fn=None) -> bool:
        async with self.session_lock:
            if self.stage != "RIDE_ACTIVE":
                return False
            self.stage = "AWAITING_DESTINATION"
            self.active_ride = None
            self.pending_ride = None
            self.booking_id = None
            if push_fn:
                await push_fn(ui_message({
                    "type": "RIDE_CANCELLED",
                    "reason": "ACTIVE_RIDE_CANCELLED",
                    "turn_id": self.current_turn_id,
                    "turn_seq": self.turn_generation,
                }))
            return True


SESSIONS: dict[str, Session] = {}


def ui_message(payload: dict) -> OutputTransportMessageUrgentFrame:
    if "msg_id" not in payload:
        payload["msg_id"] = f"msg-{uuid.uuid4().hex[:8]}"
    return OutputTransportMessageUrgentFrame(message=payload)


# ----------------------------------------------------------------------------- Dynamic Piper TTS
class DynamicPiperTTSService(PiperTTSService):
    """Dynamically switches Piper voice model according to session.language before synthesis."""

    def __init__(self, session: Session, **kwargs):
        self.session = session
        sr = kwargs.get("sample_rate", OUT_SAMPLE_RATE)
        kwargs["sample_rate"] = sr
        super().__init__(**kwargs)
        # Pipecat's TTSService initializes self._sample_rate = 0 until setup() runs.
        # Ensure self._sample_rate is non-zero so soxr resampler never sees 0.
        self._sample_rate = sr

    async def run_tts(self, text: str, context_id: str):
        voice_stem = resolve_voice_for_lang(self.session.language)
        voice_path = VOICES_DIR / f"{voice_stem}.onnx"
        if voice_path.exists():
            self._voice = _CachedPiperVoice.load(voice_path)

        captured_seq = self.session.turn_generation
        t0 = time.time()
        first_frame = True
        self.session.is_bot_speaking = True
        total_audio_bytes = 0
        try:
            async for frame in super().run_tts(text, context_id):
                if captured_seq != self.session.turn_generation:
                    logger.info(f"[{self.session.session_id}] TTS interrupted, discarding frames for obsolete turn {captured_seq}")
                    break
                if first_frame:
                    first_frame = False
                    logger.info(f"[{self.session.session_id}] Piper TTS TTFB: {time.time() - t0:.3f}s for {text[:35]!r}")
                if isinstance(frame, OutputAudioRawFrame):
                    total_audio_bytes += len(frame.audio)
                yield frame
        finally:
            play_dur = (total_audio_bytes / 2) / OUT_SAMPLE_RATE if total_audio_bytes else 0.35
            await asyncio.sleep(min(play_dur, 2.5))
            self.session.is_bot_speaking = False


# ----------------------------------------------------------------------------- Multilingual Whisper STT
class MultilingualWhisperSTT(SegmentedSTTService):
    """Whisper STT with fast live streaming interim hypotheses and authoritative final turn settlement."""

    @property
    def wants_wav_segments(self) -> bool:
        return False

    def __init__(self, session: Session, **kwargs):
        # 300ms trailing silence gives natural end-of-speech detection without dead-air stall
        super().__init__(trailing_silence_secs=0.30, **kwargs)
        self.session = session
        self._model = get_whisper_model()
        self._interim_model = get_interim_whisper_model()
        self._interim_task: asyncio.Task | None = None
        self._last_interim_time: float = 0.0
        self._last_interim_text: str = ""
        self._active_utterance_seq: int = 0

    def _transcribe_interim(self, audio_f32: np.ndarray, lang: str) -> str:
        """Fast greedy decoding for live provisional updates while user is speaking (200-300ms)."""
        try:
            segments, _ = self._interim_model.transcribe(
                audio_f32,
                language=lang,
                beam_size=1,
                temperature=0.0,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.75).strip()
        except Exception as e:
            logger.debug(f"Interim transcription error: {e}")
            return ""

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        if getattr(self.session, "is_processing_turn", False):
            logger.debug(f"[{self.session.session_id}] Discarding speech start during active turn processing")
            return
        await super()._handle_user_started_speaking(frame)
        self._last_interim_time = time.time()
        self._last_interim_text = ""
        self._active_utterance_seq = self.session.turn_generation + 1
        logger.debug(f"[{self.session.session_id}] Speech started, active utterance seq={self._active_utterance_seq}")

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._active_utterance_seq = 0
        if self._interim_task and not self._interim_task.done():
            self._interim_task.cancel()
            self._interim_task = None
        await super()._handle_user_stopped_speaking(frame)

    async def _run_interim_worker(self, audio_bytes: bytes, target_seq: int):
        if not self._user_speaking or self.session.is_bot_speaking or getattr(self.session, "is_processing_turn", False) or self._active_utterance_seq != target_seq:
            return
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_f32.size < IN_SAMPLE_RATE * 0.35:
            return

        # Keep bounded to the most recent 4 seconds of speech
        if audio_f32.size > IN_SAMPLE_RATE * 4:
            audio_f32 = audio_f32[-IN_SAMPLE_RATE * 4:]

        lang = self.session.language if self.session.language in ALLOWED_LANGS else "en"
        t0 = time.time()
        try:
            text = await asyncio.to_thread(self._transcribe_interim, audio_f32, lang)
        except Exception as e:
            logger.debug(f"Interim transcription exception: {e}")
            return

        inf_dur = time.time() - t0

        if not self._user_speaking or self._active_utterance_seq != target_seq or self.session.is_bot_speaking or getattr(self.session, "is_processing_turn", False):
            return

        if text and text != self._last_interim_text:
            self._last_interim_text = text
            logger.info(f"[{self.session.session_id}] [INTERIM {inf_dur:.3f}s][seq={target_seq}] {text!r}")
            await self.push_frame(ui_message({
                "type": "INTERIM_TRANSCRIPT",
                "text": text,
                "turn_seq": target_seq,
                "is_final": False,
            }))

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        await super().process_audio_frame(frame, direction)

        if self._user_speaking and not self.session.is_bot_speaking and not getattr(self.session, "is_processing_turn", False):
            now = time.time()
            if (now - self._last_interim_time >= 0.35) and (self._interim_task is None or self._interim_task.done()):
                self._last_interim_time = now
                curr_buffer = bytes(self._audio_buffer)
                self._interim_task = asyncio.create_task(self._run_interim_worker(curr_buffer, self._active_utterance_seq))

    def _transcribe(self, audio: np.ndarray, biased_lang: str) -> tuple[str, str, float]:
        lang = biased_lang if biased_lang in ALLOWED_LANGS else "en"
        initial_prompt = (
            "Bharat Online Cabs ride booking in Nagpur, Maharashtra. "
            "Taking customer to destination, landmark, area, shop, hotel, hospital, or railway station."
        )
        segments, _ = self._model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            temperature=0.0,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            hotwords=STT_HOTWORDS,
            vad_filter=True,
        )
        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list if s.no_speech_prob < 0.8).strip()
        avg_prob = 1.0 - (sum(s.no_speech_prob for s in seg_list) / len(seg_list)) if seg_list else 0.0
        return text, lang, avg_prob

    async def run_stt(self, audio: bytes):
        if self.session.is_bot_speaking or getattr(self.session, "is_processing_turn", False):
            logger.debug(f"[{self.session.session_id}] Discarding microphone audio during active turn processing or bot playback")
            return

        audio_f32 = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_f32.size < IN_SAMPLE_RATE * 0.20:
            await self.push_frame(ui_message({
                "type": "STT_EMPTY",
                "turn_id": self.session.current_turn_id or "turn-init",
                "turn_seq": self.session.turn_generation,
            }))
            return

        turn_ref = self.session.current_turn_id or "turn-init"
        # Stage 2: Audio capture & chunking
        TurnTracer.trace(2, "AUDIO_CAPTURE", self.session.session_id, turn_ref,
                         sample_rate=IN_SAMPLE_RATE, channels=1, bytes_len=len(audio))

        # Stage 3: Audio metrics
        rms = float(np.sqrt(np.mean(audio_f32**2))) if audio_f32.size else 0.0
        audio_dur = float(audio_f32.size / IN_SAMPLE_RATE)
        TurnTracer.trace(3, "AUDIO_METRICS", self.session.session_id, turn_ref,
                         duration_s=round(audio_dur, 2), rms=round(rms, 4), speech_detected=True)

        # Stage 5: STT decoding settings
        TurnTracer.trace(5, "STT_SETTINGS", self.session.session_id, turn_ref,
                         beam_size=2, initial_prompt_active=True, hotwords=STT_HOTWORDS[:35])

        t0 = time.time()
        biased = self.session.language
        text, lang, prob = await asyncio.to_thread(self._transcribe, audio_f32, biased)
        stt_dur = time.time() - t0

        clean_lower = text.lower().strip() if text else ""
        is_hallucination = (len(clean_lower) <= 2 or clean_lower in WHISPER_HALLUCINATIONS or any(
            h in clean_lower for h in ("thank you for watching", "subtitles by", "please subscribe", "transcription by")
        ))

        if not text or is_hallucination:
            if is_hallucination:
                logger.info(f"Filtered Whisper ghost hallucination: {text!r}")
            else:
                logger.debug(f"[{self.session.session_id}] STT returned empty transcript")
            await self.push_frame(ui_message({
                "type": "STT_EMPTY",
                "turn_id": turn_ref,
                "turn_seq": self.session.turn_generation,
            }))
            return

        # Stage 4: Raw STT output
        TurnTracer.trace(4, "RAW_STT", self.session.session_id, turn_ref,
                         text=text, lang=lang, confidence=round(prob, 3), latency_s=round(stt_dur, 3))

        logger.info(f"[{self.session.session_id}] STT [{lang}] ({stt_dur:.3f}s): {text}")
        try:
            language = Language(lang)
        except ValueError:
            language = None
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)


# ----------------------------------------------------------------------------- serializer
class BharatSerializer(FrameSerializer):
    """Raw PCM binary frames for audio, JSON text frames for control/telemetry."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message, ensure_ascii=False)
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "INTERRUPT"})
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            if len(data) < 2:
                return None
            if len(data) % 2:
                data = data[:-1]
            return InputAudioRawFrame(audio=bytes(data), sample_rate=IN_SAMPLE_RATE, num_channels=1)
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Ignoring non-JSON text message: {data[:80]!r}")
            return None
        return InputTransportMessageFrame(message=msg) if isinstance(msg, dict) else None


# ----------------------------------------------------------------------------- Session Controller (State Machine)
class SessionController(FrameProcessor):
    """Deterministic State Machine for Routing, Route Rendering, and Ride Confirmation."""

    def __init__(self, session: Session, context: LLMContext | None = None, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.context = context
        self._turn_task: asyncio.Task | None = None

    async def _resolve_and_preview_trip(self, lat: float, lon: float, name: str):
        """Deterministically calculate trip details, push MAP_ROUTE_PREVIEW, and speak confirmation."""
        captured_seq = self.session.turn_generation
        logger.info(f"[{self.session.session_id}] Deterministic trip resolution -> {name} ({lat}, {lon})")
        trip = await asyncio.to_thread(se.calculate_trip_details, self.session.lat, self.session.lon, lat, lon, name)
        if captured_seq != self.session.turn_generation:
            logger.info(f"[{self.session.session_id}] Discarding trip resolution: turn superseded/cleared")
            return

        if trip.get("status") == "ERROR" or trip.get("fare_inr") is None:
            logger.warning(f"[{self.session.session_id}] Routing failed: {trip.get('error_message')}")
            lang = self.session.language
            if lang == "hi":
                spoken = f"मुझे {name} के लिए सड़क मार्ग खोजने में समस्या आ रही है। क्या आप कोई अन्य मुख्य सड़क या नजदीकी स्थान बता सकते हैं?"
            elif lang == "mr":
                spoken = f"मला {name} साठी रस्त्याचा मार्ग शोधण्यात अडचण येत आहे. आपण दुसरा जवळचा रस्ता किंवा ठिकाण सांगू शकता का?"
            else:
                spoken = f"I'm unable to calculate the road driving route to {name} right now. Could you specify a nearby main road or landmark?"
            await self.push_frame(TTSSpeakFrame(spoken))
            await self.push_frame(ui_message({
                "type": "BOT_RESPONSE",
                "text": spoken,
                "turn_id": self.session.current_turn_id,
                "turn_seq": self.session.turn_generation,
            }))
            return

        await self.session.set_route_preview(trip, lat, lon, self.push_frame)

        fare = round(trip["fare_inr"])
        dist = trip["distance_km"]
        dur = trip["duration_mins"]
        dest = trip["destination"]

        lang = self.session.language
        if lang == "hi":
            spoken = f"{dest} के लिए ट्रिप {dist} किलोमीटर है और अनुमानित किराया {fare} रुपये है। क्या मैं यह राइड बुक कर दूँ?"
        elif lang == "mr":
            spoken = f"{dest} साठी ट्रिप {dist} किलोमीटर असून अंदाजे भाडे {fare} रुपये आहे. मी ही राईड बुक करू का?"
        else:
            spoken = f"The distance to {dest} is {dist} km and the fare is {fare} rupees. Shall I book this ride for you?"

        # Stage 12: Route & fare calculation, TTS spoken output
        TurnTracer.trace(12, "ROUTE_FARE_TTS", self.session.session_id, self.session.current_turn_id or "turn-000",
                         dest=dest, distance_km=dist, fare_inr=fare, duration_mins=dur, spoken_len=len(spoken))

        await self.push_frame(TTSSpeakFrame(spoken))
        await self.push_frame(ui_message({
            "type": "BOT_RESPONSE",
            "text": spoken,
            "turn_id": self.session.current_turn_id,
            "turn_seq": self.session.turn_generation,
        }))
        if self.context and self.context.messages:
            self.context.messages.append({"role": "assistant", "content": spoken})

    async def _confirm_current_booking(self):
        """Deterministically confirm booking, emit BOOKING_CONFIRMED, and speak confirmation."""
        bid = await self.session.confirm_booking(self.push_frame)
        if not bid or not self.session.pending_ride:
            return
        dest = self.session.pending_ride["destination"]
        logger.info(f"[{self.session.session_id}] Deterministic booking confirmed -> {bid} ({dest})")

        lang = self.session.language
        if lang == "hi":
            spoken = f"{dest} के लिए आपकी डेमो बुकिंग संदर्भ {bid} के साथ दर्ज हो गई है। कृपया ध्यान दें कि यह एक सिमुलेशन प्रोटोटाइप है, कोई वास्तविक ड्राइवर असाइन नहीं है।"
        elif lang == "mr":
            spoken = f"{dest} साठी आपली डेमो बुकिंग संदर्भ {bid} सह नोंदवली गेली आहे. कृपया लक्षात घ्या की हे केवळ सिम्युलेशन प्रोटोटाइप आहे, प्रत्यक्ष ड्रायव्हर उपलब्ध नाही."
        else:
            spoken = f"Your demo booking to {dest} is recorded with reference {bid}. Please note this is a simulation prototype and no live driver has been dispatched."

        await self.push_frame(TTSSpeakFrame(spoken))
        await self.push_frame(ui_message({
            "type": "BOT_RESPONSE",
            "text": spoken,
            "turn_id": self.session.current_turn_id,
            "turn_seq": self.session.turn_generation,
        }))
        if self.context and self.context.messages:
            self.context.messages.append({"role": "assistant", "content": spoken})

    async def _handle_text_turn(self, text: str) -> bool:
        """Evaluates customer input against state machine rules."""
        t = text.lower().strip()

        # Stage 6: Intent classification
        parsed_intent = se.parse_natural_intent_and_destination(text)
        intent_type = "UNKNOWN"
        if is_cancellation_intent(text):
            intent_type = "CANCELLATION"
        elif is_confirmation_intent(text):
            intent_type = "CONFIRMATION"
        elif parsed_intent.get("intent") in ("BOOK_CAB", "DESTINATION_SEARCH") and parsed_intent.get("destination"):
            intent_type = "DESTINATION_SEARCH"
        elif any(t.startswith(w) for w in ("hello", "hi", "hey", "namaste", "namaskar", "good morning", "good evening")):
            intent_type = "GREETING"
        elif any(k in t for k in ("where", "driver", "when", "kahan", "kitna time", "arrival", "cab", "otp", "pin")):
            intent_type = "RIDE_STATUS_QUERY"
        else:
            intent_type = "DESTINATION_SEARCH"
        TurnTracer.trace(6, "INTENT_CLASSIFICATION", self.session.session_id, self.session.current_turn_id or "turn-000",
                         intent=intent_type, raw_text=text, stage=self.session.stage)

        # Conversational greetings
        words = t.strip().split()
        if (t in ("hello", "hi", "hey", "namaste", "namaskar", "good morning", "good evening") or
            (len(words) <= 2 and words and words[0] in ("hello", "hi", "hey", "namaste", "namaskar"))):
            lang = self.session.language
            greeting = GREETINGS.get(lang, GREETINGS["en"])
            await self.push_frame(TTSSpeakFrame(greeting))
            await self.push_frame(ui_message({
                "type": "BOT_RESPONSE",
                "text": greeting,
                "turn_id": self.session.current_turn_id,
                "turn_seq": self.session.turn_generation,
            }))
            if self.context and self.context.messages:
                self.context.messages.append({"role": "assistant", "content": greeting})
            return True

        # Handle active ride queries
        if self.session.stage == "RIDE_ACTIVE":
            if is_cancellation_intent(text):
                ok = await self.session.cancel_ride(self.push_frame)
                if ok:
                    lang = self.session.language
                    if lang == "hi":
                        spoken = "आपकी सिमुलेटेड राइड कैंसिल कर दी गई है। आप कहाँ जाना चाहेंगे?"
                    elif lang == "mr":
                        spoken = "आपली सिम्युलेटेड राईड रद्द करण्यात आली आहे. आपण कुठे जाऊ इच्छिता?"
                    else:
                        spoken = "Your simulated ride has been cancelled. Where would you like to go instead?"
                    await self.push_frame(TTSSpeakFrame(spoken))
                    await self.push_frame(ui_message({
                        "type": "BOT_RESPONSE",
                        "text": spoken,
                        "turn_id": self.session.current_turn_id,
                        "turn_seq": self.session.turn_generation,
                    }))
                return True

            if any(w in t for w in ("where", "driver", "when", "kahan", "kitna time", "arrival", "cab", "gaadi")):
                lang = self.session.language
                if lang == "hi":
                    spoken = "यह बुकिंग सिमुलेशन प्रोटोटाइप मोड में है। लाइव ड्राइवर डिस्पैच सक्रिय नहीं है, इसलिए सड़क पर कोई वास्तविक कैब नहीं है।"
                elif lang == "mr":
                    spoken = "ही बुकिंग सिम्युलेशन प्रोटोटाइप मोडमध्ये आहे. प्रत्यक्ष ड्रायव्हर डिस्पॅच सक्रिय नाही, त्यामुळे रस्त्यावर कोणतीही प्रत्यक्ष कॅब नाही."
                else:
                    spoken = "This booking is in prototype simulation mode. Live driver dispatch is not connected, so no real vehicle is on the road."
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                return True

            if any(w in t for w in ("otp", "pin", "booking id", "number", "code")):
                bid = self.session.booking_id
                lang = self.session.language
                if lang == "hi":
                    spoken = f"आपकी डेमो बुकिंग संदर्भ संख्या {bid} है। सिमुलेशन मोड में किसी ओटीपी की आवश्यकता नहीं है।"
                elif lang == "mr":
                    spoken = f"आपला डेमो बुकिंग संदर्भ क्रमांक {bid} आहे. सिम्युलेशन मोडमध्ये कोणत्याही ओटीपीची आवश्यकता नाही."
                else:
                    spoken = f"Your demo booking reference is {bid}. No OTP is needed as this is an automated prototype simulation."
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                return True

            if is_confirmation_intent(text):
                dest = (self.session.active_ride or {}).get("destination", "your destination")
                spoken = f"Your ride to {dest} is already confirmed with booking ID {self.session.booking_id}."
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                return True

            return False

        # Guard: Confirmation intent when no trip proposal or destination confirmation exists
        if self.session.stage in ("GREETING", "AWAITING_DESTINATION") and is_confirmation_intent(text):
            lang = self.session.language
            if lang == "hi":
                spoken = "आपने अभी तक कोई गंतव्य नहीं चुना है। आप कहाँ जाना चाहेंगे?"
            elif lang == "mr":
                spoken = "आपण अद्याप कोणतेही ठिकाण निवडलेले नाही. आपण कुठे जाऊ इच्छिता?"
            else:
                spoken = f"You haven't selected a destination yet. Where in {se.DEFAULT_CITY_NAME} would you like to go?"
            await self.push_frame(TTSSpeakFrame(spoken))
            await self.push_frame(ui_message({
                "type": "BOT_RESPONSE",
                "text": spoken,
                "turn_id": self.session.current_turn_id,
                "turn_seq": self.session.turn_generation,
            }))
            if self.context and self.context.messages:
                self.context.messages.append({"role": "assistant", "content": spoken})
            return True

        # Guard: Cancellation intent when no proposal or active ride exists
        if self.session.stage in ("GREETING", "AWAITING_DESTINATION") and is_cancellation_intent(text):
            lang = self.session.language
            if lang == "hi":
                spoken = "अभी कोई ऐक्टिव राइड या बुकिंग नहीं है। आप कहाँ जाना चाहेंगे?"
            elif lang == "mr":
                spoken = "सध्या कोणतीही सक्रिय राईड किंवा बुकिंग नाही. आपण कुठे जाऊ इच्छिता?"
            else:
                spoken = "You don't have an active ride or booking to cancel. Where would you like to go?"
            await self.push_frame(TTSSpeakFrame(spoken))
            await self.push_frame(ui_message({
                "type": "BOT_RESPONSE",
                "text": spoken,
                "turn_id": self.session.current_turn_id,
                "turn_seq": self.session.turn_generation,
            }))
            if self.context and self.context.messages:
                self.context.messages.append({"role": "assistant", "content": spoken})
            return True

        # 1. Candidate confirmation during destination clarification (Stage 1 of Two-Phase Confirmation)
        if self.session.stage == "AWAITING_DESTINATION_CONFIRMATION" and self.session.pending_destination:
            if is_cancellation_intent(text):
                self.session.pending_destination = None
                self.session.stage = "AWAITING_DESTINATION"
                lang = self.session.language
                if lang == "hi":
                    spoken = "ठीक है, आप कहाँ जाना चाहेंगे?"
                elif lang == "mr":
                    spoken = "ठीक आहे, आपण कुठे जाऊ इच्छिता?"
                else:
                    spoken = "Alright, where would you like to go instead?"
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                return True

            if is_confirmation_intent(text):
                # User confirmed the suggested destination!
                cand = self.session.pending_destination
                self.session.pending_destination = None
                logger.info(f"[{self.session.session_id}] Destination confirmed by user -> {cand['name']}")
                # Calculates route & fare, transitions to AWAITING_CONFIRMATION, and presents fare proposal (does NOT book yet)
                await self._resolve_and_preview_trip(cand["lat"], cand["lon"], cand["name"])
                return True

            # If user spoke a new destination instead of affirming/denying, clear and fall through
            self.session.pending_destination = None
            self.session.stage = "AWAITING_DESTINATION"

        # 2. Booking confirmation or cancellation in proposal stage (Stage 2 of Two-Phase Confirmation)
        if self.session.stage == "AWAITING_CONFIRMATION":
            if is_cancellation_intent(text):
                self.session.stage = "AWAITING_DESTINATION"
                self.session.pending_ride = None
                lang = self.session.language
                if lang == "hi":
                    spoken = "ट्रिप कैंसिल कर दी गई है। आप कहाँ जाना चाहेंगे?"
                elif lang == "mr":
                    spoken = "ट्रिप रद्द करण्यात आली आहे. आपण कुठे जाऊ इच्छिता?"
                else:
                    spoken = "Trip proposal cancelled. Where would you like to go instead?"
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({
                    "type": "RIDE_CANCELLED",
                    "reason": "PROPOSAL_CANCELLED",
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                return True

            if is_confirmation_intent(text):
                await self._confirm_current_booking()
                return True

        # 3. Candidate resolution during disambiguation
        if self.session.stage == "DISAMBIGUATING" and self.session.last_search:
            if is_cancellation_intent(text):
                self.session.stage = "AWAITING_DESTINATION"
                self.session.last_search = None
                spoken = "Where would you like to go instead?"
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                return True

            options = ([self.session.last_search["primary_match"]] if self.session.last_search.get("primary_match") else []) + self.session.last_search.get("alternatives", [])
            cand = match_candidate_choice(text, options)
            if cand:
                await self._resolve_and_preview_trip(cand["lat"], cand["lon"], cand["name"])
                return True

        # 4. Fast Deterministic Destination Evaluation (Sub-Second Routing & Disambiguation)
        # Note: also evaluated when in AWAITING_CONFIRMATION or AWAITING_DESTINATION_CONFIRMATION to seamlessly change destination
        if self.session.stage in ("GREETING", "AWAITING_DESTINATION", "AWAITING_DESTINATION_CONFIRMATION", "DISAMBIGUATING", "AWAITING_CONFIRMATION"):
            captured_seq = self.session.turn_generation
            turn_ref = self.session.current_turn_id or "turn-000"

            clean_dest = se.extract_destination_entity(text)
            is_cat = clean_dest.lower() in se.COMMON_CATEGORIES or any(c in clean_dest.lower() for c in se.COMMON_CATEGORIES)
            # Stage 7: Entity/Category extraction
            TurnTracer.trace(7, "ENTITY_EXTRACTION", self.session.session_id, turn_ref,
                             extracted=clean_dest, is_category=is_cat)

            # Stage 8: Search mode
            search_engine = "OpenStreetMap Nominatim Engine"
            TurnTracer.trace(8, "SEARCH_MODE", self.session.session_id, turn_ref,
                             mode=search_engine)

            # Stage 9: Search query parameters
            sanitized_q = se.sanitize_search_query(clean_dest)
            TurnTracer.trace(9, "SEARCH_PARAMS", self.session.session_id, turn_ref,
                             sanitized=sanitized_q, lat=self.session.lat, lon=self.session.lon, radius_km=15.0)

            # Invalidate old state when a new destination search starts
            self.session.pending_destination = None
            res = await asyncio.to_thread(se.evaluate_destination_intent, text, self.session.lat, self.session.lon)
            if captured_seq != self.session.turn_generation:
                logger.info(f"[{self.session.session_id}] Discarding destination search result: turn superseded/cleared")
                return True

            # Stage 10: Candidate results count & status
            status_val = res.get("status") if res else "NONE"
            res_type = res.get("resolution_type", status_val) if res else "NONE"
            TurnTracer.trace(10, "CANDIDATE_RESULTS", self.session.session_id, turn_ref,
                             status=status_val, resolution_type=res_type, count=res.get("outlet_count", 0) if res else 0,
                             primary=(res.get("primary_match") or {}).get("name") if res else None)

            # Stage 11: Candidate selection or rejection reason
            decision = ("exact_match" if status_val in ("EXACT", "RESOLVED")
                        else ("possible_match_clarification" if status_val == "POSSIBLE_MATCH"
                              else ("disambiguation_required" if status_val in ("DISAMBIGUATION_NEEDED", "MULTIPLE_CANDIDATES")
                                    else ("provider_quota_exhausted" if status_val == "PROVIDER_ERROR"
                                          else "zero_results")))) if res else "empty_response"
            TurnTracer.trace(11, "CANDIDATE_DECISION", self.session.session_id, turn_ref,
                             decision=decision, provider_status=res.get("provider_status") if res else None)

            # Outcome 1: EXACT MATCH -> Route calculation & fare proposal immediately
            if res and (res.get("status") in ("EXACT", "RESOLVED") or res.get("resolution_type") in ("EXACT_MATCH", "RESOLVED")):
                self.session.last_search = res
                options = ([res["primary_match"]] if res.get("primary_match") else []) + res.get("alternatives", [])
                await self.push_frame(ui_message({
                    "type": "POI_CANDIDATES",
                    "query": text,
                    "status": "EXACT",
                    "resolution_type": "EXACT_MATCH",
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                    "candidates": [{k: c.get(k) for k in ("name", "short_address", "lat", "lon", "distance_km", "rating")}
                                   for c in options[:10]]
                }))
                pm = res["primary_match"]
                await self._resolve_and_preview_trip(pm["lat"], pm["lon"], pm["name"])
                return True

            # Outcome 2: POSSIBLE MATCH (Typo/Phonetic Recovery) -> Ask for destination confirmation (does NOT book)
            if res and (res.get("status") == "POSSIBLE_MATCH" or res.get("resolution_type") == "POSSIBLE_MATCH"):
                pm = res.get("suggested_match") or res.get("primary_match")
                self.session.pending_destination = pm
                self.session.stage = "AWAITING_DESTINATION_CONFIRMATION"
                self.session.last_search = res

                await self.push_frame(ui_message({
                    "type": "POI_CANDIDATES",
                    "query": text,
                    "status": "POSSIBLE_MATCH",
                    "resolution_type": "POSSIBLE_MATCH",
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                    "candidates": [{k: pm.get(k) for k in ("name", "short_address", "lat", "lon", "distance_km", "rating")}]
                }))

                spoken = res.get("clarification_prompt") or f"Did you mean {pm['name']}?"
                lang = self.session.language
                loc = f" {pm.get('short_address', '')} में" if pm.get("short_address") else ""
                loc_mr = f" {pm.get('short_address', '')} मधील" if pm.get("short_address") else ""
                if lang == "hi":
                    spoken = f"क्या आपका मतलब{loc} {pm['name']} से है?"
                elif lang == "mr":
                    spoken = f"आपल्याला{loc_mr} {pm['name']} येथे जायचे आहे का?"

                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                await self.push_frame(TTSSpeakFrame(spoken))
                if self.context and self.context.messages:
                    self.context.messages.append({"role": "assistant", "content": spoken})
                return True

            # Outcome 3: MULTIPLE CANDIDATES (Chain disambiguation or multiple options)
            if res and (res.get("status") in ("DISAMBIGUATION_NEEDED", "MULTIPLE_CANDIDATES") or res.get("resolution_type") in ("MULTIPLE_CANDIDATES", "NEEDS_CLARIFICATION")):
                self.session.stage = "DISAMBIGUATING"
                self.session.last_search = res
                options = ([res["primary_match"]] if res.get("primary_match") else []) + res.get("alternatives", [])

                await self.push_frame(ui_message({
                    "type": "POI_CANDIDATES",
                    "query": text,
                    "status": "DISAMBIGUATION_NEEDED",
                    "resolution_type": "MULTIPLE_CANDIDATES",
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                    "candidates": [{k: c.get(k) for k in ("name", "short_address", "lat", "lon", "distance_km", "rating")}
                                   for c in options[:10]]
                }))

                closest = options[0] if options else None
                hint = res.get("clarification_prompt") or (
                    f"The closest option is {closest['name']} in {closest['short_address']}, about {closest['distance_km']} km away. Would you like to go there, or pick another from the map?"
                )
                lang = self.session.language
                if lang == "hi" and closest:
                    spoken = f"आपके सबसे पास {closest['short_address']} में {closest['name']} है, लगभग {closest['distance_km']} किलोमीटर। क्या आप यहाँ जाना चाहेंगे या दूसरा विकल्प चुनेंगे?"
                elif lang == "mr" and closest:
                    spoken = f"तुमच्या सर्वात जवळ {closest['short_address']} मध्ये {closest['name']} आहे, सुमारे {closest['distance_km']} किलोमीटर. आपण तिथे जाऊ इच्छिता का?"
                else:
                    spoken = hint

                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
                await self.push_frame(TTSSpeakFrame(spoken))
                if self.context and self.context.messages:
                    self.context.messages.append({"role": "assistant", "content": spoken})
                return True

            # Outcome 4 & 5: NO RESULTS or PROVIDER ERROR
            if res and (res.get("status") in ("NOT_FOUND", "PROVIDER_ERROR", "FILTERED_ALL") or res.get("resolution_type") in ("NO_RESULTS", "PROVIDER_ERROR", "FILTERED_ALL")):
                is_general_qa = any(t.startswith(w) for w in ("what", "who", "why", "how", "tell me about"))
                if not is_general_qa:
                    prompt = res.get("clarification_prompt") or f"I couldn't find '{text}'. Could you tell me a nearby landmark or major road?"
                    lang = self.session.language
                    if lang == "hi":
                        spoken = "हमारा मैप सर्च सेवा अभी अत्यधिक ट्रैफिक का सामना कर रहा है। क्या आप कोई नजदीकी मुख्य सड़क या लैंडमार्क बता सकते हैं?" if res.get("status") == "PROVIDER_ERROR" else f"मुझे '{text}' नहीं मिला। क्या आप कोई नजदीकी लैंडमार्क या मुख्य सड़क बता सकते हैं?"
                    elif lang == "mr":
                        spoken = "आमची नकाशा शोध सेवा सध्या व्यस्त आहे. आपण जवळचा मुख्य रस्ता किंवा खूण सांगू शकता का?" if res.get("status") == "PROVIDER_ERROR" else f"मला '{text}' सापडले नाही. आपण जवळची एखादी खूण किंवा परिसराचे नाव सांगू शकता का?"
                    else:
                        spoken = prompt
                    if res.get("status") == "PROVIDER_ERROR":
                        await self.push_frame(ui_message({
                            "type": "ERROR",
                            "error": res.get("provider_error") or "Map search provider timeout",
                            "turn_id": self.session.current_turn_id,
                            "turn_seq": self.session.turn_generation,
                        }))
                    await self.push_frame(ui_message({
                        "type": "BOT_RESPONSE",
                        "text": spoken,
                        "status": res.get("status"),
                        "resolution_type": res.get("resolution_type"),
                        "turn_id": self.session.current_turn_id,
                        "turn_seq": self.session.turn_generation,
                    }))
                    await self.push_frame(TTSSpeakFrame(spoken))
                    if self.context and self.context.messages:
                        self.context.messages.append({"role": "assistant", "content": spoken})
                    return True

        return False

    async def _execute_text_turn(self, text: str, turn_id: str, curr_seq: int):
        self.session.is_processing_turn = True
        try:
            t_intent_start = time.time()
            handled = False
            try:
                handled = await asyncio.wait_for(
                    self._handle_text_turn(text),
                    timeout=7.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.session.session_id}][{turn_id}] Text operation timed out after 7.0s")
                lang = self.session.language
                if lang == "hi":
                    spoken = "सर्च में अधिक समय लग रहा है। कृपया दोबारा बताएं या कोई नजदीकी मुख्य स्थान बताएं।"
                elif lang == "mr":
                    spoken = "शोधण्यात जास्त वेळ लागत आहे. कृपया पुन्हा सांगा किंवा जवळचा मुख्य रस्ता सांगा."
                else:
                    spoken = "The destination search took too long. Please try again or specify a nearby landmark."
                await self.push_frame(ui_message({
                    "type": "ERROR",
                    "error": "Destination search timed out",
                    "turn_id": turn_id,
                    "turn_seq": curr_seq,
                }))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": turn_id,
                    "turn_seq": curr_seq,
                }))
                await self.push_frame(TTSSpeakFrame(spoken))
                handled = True

            intent_dur = time.time() - t_intent_start

            if curr_seq != self.session.turn_generation:
                logger.info(f"[{self.session.session_id}][{turn_id}] Discarding superseded text turn result")
                return

            await self.push_frame(ui_message({
                "type": "PERF_DIAGNOSTICS",
                "turn_id": turn_id,
                "turn_seq": curr_seq,
                "timings": {
                    "intent_ms": round(intent_dur * 1000),
                    "deterministic_fast_path": handled,
                }
            }))
            if not handled:
                await self.push_frame(
                    LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True))
        except asyncio.CancelledError:
            logger.info(f"[{self.session.session_id}][{turn_id}] Text turn task cancelled")
        except Exception as e:
            logger.exception(f"[{self.session.session_id}][{turn_id}] Text turn error: {e}")
        finally:
            self.session.is_processing_turn = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self.session.is_bot_speaking = False

        if isinstance(frame, InputTransportMessageFrame) and direction == FrameDirection.DOWNSTREAM:
            msg = frame.message or {}
            mtype = msg.get("type")

            if mtype == "CLEAR_CONVERSATION":
                self._last_text_input = ("", 0.0)
                if self._turn_task and not self._turn_task.done():
                    self._turn_task.cancel()
                client_seq = msg.get("turn_seq")
                was_active = self.session.clear_conversation()
                if client_seq is not None:
                    try:
                        self.session.turn_generation = max(self.session.turn_generation, int(client_seq) + 1)
                    except (ValueError, TypeError):
                        pass
                logger.info(f"[{self.session.session_id}] CLEAR_CONVERSATION (active_ride_preserved={was_active}, turn_gen={self.session.turn_generation})")
                if self.context:
                    self.context.set_messages([{"role": "system", "content": get_system_prompt(self.session.language)}])
                await self.push_frame(InterruptionFrame())
                await self.push_frame(ui_message({
                    "type": "CONVERSATION_CLEARED",
                    "turn_seq": self.session.turn_generation,
                    "active_ride_preserved": was_active,
                    "booking_id": self.session.booking_id if was_active else None,
                }))
                return

            if mtype == "INTERRUPT_PLAYBACK":
                if self._turn_task and not self._turn_task.done():
                    self._turn_task.cancel()
                self.session.is_bot_speaking = False
                await self.push_frame(InterruptionFrame())
                return

            if mtype == "SET_LANGUAGE":
                lang = str(msg.get("lang", "")).strip().lower()
                if lang in ALLOWED_LANGS:
                    old_lang = self.session.language
                    self.session.language = lang
                    logger.info(f"[{self.session.session_id}] Language switched to: {lang}")
                    if self.context and self.context.messages:
                        self.context.messages[0] = {"role": "system", "content": get_system_prompt(lang)}
                    await self.push_frame(ui_message({"type": "LANGUAGE_CHANGED", "language": lang}))
                    if self.session.turns == 0 and old_lang != lang:
                        greeting = GREETINGS.get(lang, GREETINGS["en"])
                        await self.push_frame(TTSSpeakFrame(greeting))
                        await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": greeting}))
                return

            if mtype == "CLIENT_INIT":
                if "lang" in msg and str(msg["lang"]).lower() in ALLOWED_LANGS:
                    self.session.language = str(msg["lang"]).lower()
                    if self.context and self.context.messages:
                        self.context.messages[0] = {"role": "system", "content": get_system_prompt(self.session.language)}
                try:
                    lat, lon = float(msg["lat"]), float(msg["lon"])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError("out of range")
                    gps_src = str(msg.get("gps_source", "client_exact"))
                    self.session.lat, self.session.lon, self.session.gps_source = lat, lon, gps_src
                except (KeyError, TypeError, ValueError) as e:
                    await self.push_frame(ui_message({"type": "ERROR", "error": f"Invalid CLIENT_INIT: {e}"}))
                logger.info(f"[{self.session.session_id}] GPS {self.session.lat},{self.session.lon} "
                            f"({self.session.gps_source}) lang={self.session.language}")
                await self.push_frame(ui_message({"type": "SESSION_READY", **self.session.public()}))
                return

            if mtype == "SELECT_POI":
                try:
                    lat, lon = float(msg["lat"]), float(msg["lon"])
                    name = str(msg.get("name", "Destination")).strip()
                    await self._resolve_and_preview_trip(lat, lon, name)
                except Exception as e:
                    logger.error(f"Error handling SELECT_POI: {e}")
                return

            if mtype in ("CONFIRM_BOOKING", "CONFIRM_RIDE"):
                if self.session.pending_ride:
                    await self._confirm_current_booking()
                return

            if mtype == "CANCEL_RIDE":
                if self.session.stage == "RIDE_ACTIVE":
                    await self.session.cancel_ride(self.push_frame)
                    lang = self.session.language
                    spoken = "Your ride has been cancelled. Where would you like to go instead?"
                    await self.push_frame(TTSSpeakFrame(spoken))
                    await self.push_frame(ui_message({
                        "type": "BOT_RESPONSE",
                        "text": spoken,
                        "turn_id": self.session.current_turn_id,
                        "turn_seq": self.session.turn_generation,
                    }))
                elif self.session.stage == "AWAITING_CONFIRMATION":
                    self.session.stage = "AWAITING_DESTINATION"
                    self.session.pending_ride = None
                    await self.push_frame(ui_message({
                        "type": "RIDE_CANCELLED",
                        "reason": "PROPOSAL_CANCELLED",
                        "turn_id": self.session.current_turn_id,
                        "turn_seq": self.session.turn_generation,
                    }))
                return

            if mtype == "START_VOICE_SESSION":
                if not self.session.greeting_sent and self.session.turns == 0:
                    self.session.greeting_sent = True
                    greeting = GREETINGS.get(self.session.language, GREETINGS["en"])
                    await self.push_frame(TTSSpeakFrame(greeting))
                    await self.push_frame(ui_message({
                        "type": "BOT_RESPONSE",
                        "text": greeting,
                        "turn_id": "turn-000",
                        "turn_seq": self.session.turn_generation,
                    }))
                return

            if mtype == "TEXT_INPUT" and str(msg.get("text", "")).strip():
                text = str(msg["text"]).strip()
                now_t = time.time()
                last_txt, last_t = getattr(self, "_last_text_input", ("", 0.0))
                if text == last_txt and (now_t - last_t) < 1.0:
                    logger.info(f"[{self.session.session_id}] Discarding duplicate TEXT_INPUT within 1.0s: {text}")
                    return
                self._last_text_input = (text, now_t)

                self.session.greeting_sent = True
                self.session.turn_generation += 1
                self.session.turns += 1
                turn_id = f"turn-{self.session.turns:03d}"
                self.session.current_turn_id = turn_id
                curr_seq = self.session.turn_generation

                await self.push_frame(ui_message({
                    "type": "USER_TRANSCRIPT",
                    "text": text,
                    "source": "text",
                    "is_final": True,
                    "turn_id": turn_id,
                    "turn_seq": curr_seq,
                }))

                if self._turn_task and not self._turn_task.done():
                    self._turn_task.cancel()
                self._turn_task = asyncio.create_task(self._execute_text_turn(text, turn_id, curr_seq))
                return
            return
        await self.push_frame(frame, direction)


class BharatUserTurnStopStrategy(SpeechTimeoutUserTurnStopStrategy):
    """Guards against premature LLM user-turn stop triggering while backend is actively processing."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    async def _maybe_trigger_user_turn_stopped(self):
        # Prevent user-speech timeout logic from generating another assistant response
        # for that same in-flight turn while the backend is resolving the transcript.
        if getattr(self.session, "is_processing_turn", False):
            logger.debug(f"[{self.session.session_id}] Suppressed premature user-speech timeout: turn is currently in-flight")
            return
        await super()._maybe_trigger_user_turn_stopped()


class TranscriptTap(FrameProcessor):
    """Mirrors final user transcripts, tracks stage timings, and manages voice states."""

    def __init__(self, session: Session, controller: SessionController, user_stop_strategy: SpeechTimeoutUserTurnStopStrategy | None = None, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.controller = controller
        self.user_stop_strategy = user_stop_strategy
        self._vad_start_time = 0.0
        self._vad_stop_time = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_start_time = time.time()
            logger.info(f"[{self.session.session_id}] [VAD] Speech onset detected")
            await self.push_frame(ui_message({"type": "VAD_SPEECH_START"}))
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stop_time = time.time()
            vad_dur = self._vad_stop_time - self._vad_start_time if self._vad_start_time else 0.0
            logger.info(f"[{self.session.session_id}] [VAD] Speech stopped (speech duration: {vad_dur:.2f}s)")
            await self.push_frame(ui_message({"type": "VAD_SPEECH_STOP"}))
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return
            if getattr(self.session, "is_processing_turn", False):
                logger.info(f"[{self.session.session_id}] Discarding trailing transcription while turn is processing: {text!r}")
                return
            t_stt = time.time()
            vad_dur = self._vad_stop_time - self._vad_start_time if self._vad_start_time else 0.0
            stt_dur = t_stt - self._vad_stop_time if self._vad_stop_time else 0.0
            self.session.turn_generation += 1
            self.session.turns += 1
            turn_id = f"turn-{self.session.turns:03d}"
            self.session.current_turn_id = turn_id
            curr_seq = self.session.turn_generation

            # Stage 1: Turn ID & Session ID
            TurnTracer.trace(1, "TURN_START", self.session.session_id, turn_id,
                             turn_seq=curr_seq, source="voice", stage=self.session.stage)

            logger.info(f"[{self.session.session_id}][{turn_id}][seq={curr_seq}] [TIMING] VAD: {vad_dur:.2f}s | STT: {stt_dur:.3f}s | Transcript: {text!r}")
            self.session.greeting_sent = True
            await self.push_frame(ui_message({
                "type": "USER_TRANSCRIPT",
                "text": text,
                "source": "voice",
                "is_final": True,
                "turn_id": turn_id,
                "turn_seq": curr_seq,
                "language": str(frame.language) if frame.language else None,
            }))

            # Lock turn as processing and cancel any pending user-speech timers
            self.session.is_processing_turn = True
            if self.user_stop_strategy:
                await self.user_stop_strategy._discard_pending_end_of_turn()

            t_intent_start = time.time()
            handled = False
            try:
                handled = await asyncio.wait_for(
                    self.controller._handle_text_turn(text),
                    timeout=7.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.session.session_id}][{turn_id}] Spoken destination operation timed out after 7.0s")
                lang = self.session.language
                if lang == "hi":
                    spoken = "सर्च में अधिक समय लग रहा है। कृपया दोबारा बताएं या कोई नजदीकी मुख्य स्थान बताएं।"
                elif lang == "mr":
                    spoken = "शोधण्यात जास्त वेळ लागत आहे. कृपया पुन्हा सांगा किंवा जवळचा मुख्य रस्ता सांगा."
                else:
                    spoken = "The destination search took too long. Please try again or specify a nearby landmark."

                await self.push_frame(ui_message({
                    "type": "ERROR",
                    "error": "Destination search timed out",
                    "turn_id": turn_id,
                    "turn_seq": curr_seq,
                }))
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": spoken,
                    "turn_id": turn_id,
                    "turn_seq": curr_seq,
                }))
                await self.push_frame(TTSSpeakFrame(spoken))
                handled = True
            except Exception as e:
                logger.exception(f"[{self.session.session_id}][{turn_id}] Error in _handle_text_turn: {e}")
                handled = False
            finally:
                self.session.is_processing_turn = False

            intent_dur = time.time() - t_intent_start
            total_turn_dur = time.time() - self._vad_stop_time if self._vad_stop_time else intent_dur

            if curr_seq != self.session.turn_generation:
                logger.info(f"[{self.session.session_id}][{turn_id}] Discarding superseded voice turn result")
                return

            logger.info(f"[{self.session.session_id}][{turn_id}] [TIMING] Intent: {intent_dur:.3f}s | E2E Turn: {total_turn_dur:.3f}s (handled={handled})")

            await self.push_frame(ui_message({
                "type": "PERF_DIAGNOSTICS",
                "turn_id": turn_id,
                "turn_seq": curr_seq,
                "timings": {
                    "vad_speech_s": round(vad_dur, 2),
                    "stt_ms": round(stt_dur * 1000),
                    "intent_ms": round(intent_dur * 1000),
                    "e2e_turn_ms": round(total_turn_dur * 1000),
                    "deterministic_fast_path": handled,
                }
            }))

            if not handled:
                await self.push_frame(
                    LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
                )
            return

        await self.push_frame(frame, direction)


class BotTextTap(FrameProcessor):
    """Aggregates streamed LLM tokens and sends each full reply to the UI."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self._buf: list[str] = []
        self._captured_turn_seq: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            if not self._buf:
                self._captured_turn_seq = self.session.turn_generation
            self._buf.append(frame.text)
        elif isinstance(frame, InterruptionFrame):
            self._buf.clear()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._buf:
            text = "".join(self._buf).strip()
            self._buf.clear()
            if text and self._captured_turn_seq == self.session.turn_generation and not getattr(self.session, "is_processing_turn", False):
                await self.push_frame(ui_message({
                    "type": "BOT_RESPONSE",
                    "text": text,
                    "turn_id": self.session.current_turn_id,
                    "turn_seq": self.session.turn_generation,
                }))
        await self.push_frame(frame, direction)


# ----------------------------------------------------------------------------- tools
TOOLS = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="search_nearby_pois",
        description="Find a destination near the customer's GPS location. Returns EXACT, "
                    "DISAMBIGUATION_NEEDED (several outlets) or NOT_FOUND with options sorted nearest first.",
        properties={"query": {"type": "string",
                              "description": "Destination place name only, e.g. 'City Center' or 'jewelry shop near me'"}},
        required=["query"],
    ),
    FunctionSchema(
        name="calculate_trip_details",
        description="Calculate road distance, travel time and fare from the customer to the chosen destination.",
        properties={
            "dest_lat": {"type": "number", "description": "Destination latitude from search results"},
            "dest_lon": {"type": "number", "description": "Destination longitude from search results"},
            "destination_name": {"type": "string", "description": "Destination name"},
        },
        required=["dest_lat", "dest_lon", "destination_name"],
    ),
    FunctionSchema(
        name="confirm_booking",
        description="Book the cab for the last calculated trip. Call only after the customer agrees.",
        properties={},
        required=[],
    ),
])


def register_tools(llm: OLLamaLLMService, session: Session) -> None:
    sid = session.session_id

    async def search_nearby_pois(params: FunctionCallParams):
        query = str(params.arguments.get("query", "")).strip()
        logger.info(f"[{sid}] tool search_nearby_pois({query!r})")
        res = await asyncio.to_thread(se.evaluate_destination_intent, query, session.lat, session.lon)
        session.last_search = res
        options = ([res["primary_match"]] if res["primary_match"] else []) + res["alternatives"]
        session.stage = {"EXACT": "AWAITING_CONFIRMATION", "DISAMBIGUATION_NEEDED": "DISAMBIGUATING"}.get(
            res["status"], "AWAITING_DESTINATION")
        await params.llm.push_frame(ui_message({
            "type": "POI_CANDIDATES", "query": query, "status": res["status"],
            "candidates": [{k: c[k] for k in ("name", "short_address", "lat", "lon", "distance_km", "rating")}
                           for c in options[:10]]}))
        if res["status"] == "EXACT":
            pm = res["primary_match"]
            trip = await asyncio.to_thread(se.calculate_trip_details, session.lat, session.lon, pm["lat"], pm["lon"], pm["name"])
            await session.set_route_preview(trip, pm["lat"], pm["lon"], params.llm.push_frame)
            await params.result_callback({
                "status": "EXACT",
                "destination": trip["destination"],
                "distance_km": trip["distance_km"],
                "duration_mins": trip["duration_mins"],
                "fare_rupees": round(trip["fare_inr"]),
                "next_step": f"Tell the customer: Destination {trip['destination']}, distance {trip['distance_km']} km, time {trip['duration_mins']} mins, and fare {round(trip['fare_inr'])} rupees. Ask if they want to book now."
            })
            return

        closest = options[0] if options else None
        hint = res.get("clarification_prompt") or (
            f"Announce ONLY the closest option ({closest['name']} in {closest['short_address']}, {closest['distance_km']} km away). "
            "Ask if they would like to go there or pick another branch from the map."
        )
        await params.result_callback({
            "status": res["status"],
            "total_outlets": res["outlet_count"],
            "closest_branch": {
                "name": closest["name"],
                "area": closest["short_address"],
                "distance_km": closest["distance_km"],
            } if closest else None,
            "next_step": hint,
        })

    async def calculate_trip_details(params: FunctionCallParams):
        args = params.arguments
        name = str(args.get("destination_name") or "Destination").strip()
        try:
            lat, lon = float(args["dest_lat"]), float(args["dest_lon"])
        except (KeyError, TypeError, ValueError):
            lat = lon = None
        cands = []
        if session.last_search and session.last_search.get("primary_match"):
            cands = [session.last_search["primary_match"]] + session.last_search["alternatives"]
        if lat is None or se.calculate_distance(session.lat, session.lon, lat, lon) > 60:
            named = [c for c in cands if name.lower() in c["name"].lower() or c["name"].lower() in name.lower()]
            if not named:
                await params.result_callback({"error": "Invalid destination coordinates. Search the place first."})
                return
            lat, lon = named[0]["lat"], named[0]["lon"]
        logger.info(f"[{sid}] tool calculate_trip_details({lat}, {lon}, {name!r})")
        trip = await asyncio.to_thread(se.calculate_trip_details, session.lat, session.lon, lat, lon, name)
        await session.set_route_preview(trip, lat, lon, params.llm.push_frame)
        await params.result_callback({
            "destination": trip["destination"], "distance_km": trip["distance_km"],
            "duration_mins": trip["duration_mins"], "fare_rupees": round(trip["fare_inr"]),
            "next_step": f"Tell the customer: The trip is {trip['distance_km']} km and the estimated fare is {round(trip['fare_inr'])} rupees. Shall I book this ride?",
        })

    async def confirm_booking(params: FunctionCallParams):
        bid = await session.confirm_booking(params.llm.push_frame)
        if not bid:
            await params.result_callback({"error": "No trip calculated yet. Ask for the destination."})
            return
        ride = session.pending_ride
        logger.info(f"[{sid}] tool confirm_booking -> {bid}")
        await params.result_callback({
            "booking_id": bid, "destination": ride["destination"],
            "fare_rupees": round(ride["fare_inr"]), "status": "CONFIRMED",
            "next_step": f"Confirm the booking to the customer in one short sentence mentioning booking ID {bid}.",
        })

    llm.register_function("search_nearby_pois", search_nearby_pois)
    llm.register_function("calculate_trip_details", calculate_trip_details)
    llm.register_function("confirm_booking", confirm_booking)


# ----------------------------------------------------------------------------- app
def _ollama_warmup() -> bool:
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", timeout=60, json={
            "model": LLM_MODEL, "prompt": "hi", "stream": False, "keep_alive": "30m",
            "options": {"num_predict": 1}})
        return r.ok
    except requests.RequestException:
        return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    t0 = time.time()
    await asyncio.to_thread(get_whisper_model)
    for voice_name in set(VOICE_MAP.values()):
        vp = VOICES_DIR / f"{voice_name}.onnx"
        if vp.exists():
            await asyncio.to_thread(_CachedPiperVoice.load, vp)
    ok = await asyncio.to_thread(_ollama_warmup)
    logger.info(f"Startup in {time.time() - t0:.1f}s | voices={list(set(VOICE_MAP.values()))} | "
                f"whisper={WHISPER_MODEL} | ollama={'ok' if ok else 'UNREACHABLE'} | "
                f"RSS={psutil.Process().memory_info().rss / 2**20:.0f} MB")
    yield


app = FastAPI(title="Bharat Online Cabs Voice Core", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/config")
async def get_config():
    """Return frontend configuration safely without leaking server-side secrets."""
    return {
        "map_provider": se.MAP_PROVIDER,
        "map_tile_url": os.getenv("MAP_TILE_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"),
        "osm_routing_url": se.OSRM_URL,
        "osm_geocoder_url": se.NOMINATIM_URL,
        "osm_reverse_geocoder_url": se.NOMINATIM_REVERSE_URL,
        "default_lat": se.DEFAULT_LAT,
        "default_lon": se.DEFAULT_LON,
        "default_city": se.DEFAULT_CITY,
    }


@app.get("/health")
async def health():
    vm = psutil.virtual_memory()
    return {
        "status": "ok",
        "active_sessions": len(SESSIONS),
        "max_sessions": MAX_SESSIONS,
        "models": {
            "whisper": WHISPER_MODEL,
            "piper_voices": [Path(k).stem for k in _piper_cache],
            "llm": LLM_MODEL
        },
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        "system_available_mb": round(vm.available / 2**20),
    }


@app.get("/sessions")
async def sessions():
    return [s.public() for s in SESSIONS.values()]


def build_worker(websocket: WebSocket, session: Session) -> PipelineWorker:
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=BharatSerializer(),
            session_timeout=30 * 60,
        ),
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(
        sample_rate=IN_SAMPLE_RATE, params=VADParams(stop_secs=VAD_STOP_SECS, start_secs=0.2)))
    stt = MultilingualWhisperSTT(session=session, sample_rate=IN_SAMPLE_RATE)
    llm = OLLamaLLMService(
        base_url=f"{OLLAMA_HOST}/v1",
        settings=OLLamaLLMService.Settings(model=LLM_MODEL, temperature=0.2, max_tokens=150),
    )
    register_tools(llm, session)
    initial_voice = resolve_voice_for_lang(session.language)
    tts = DynamicPiperTTSService(
        session=session,
        download_dir=VOICES_DIR,
        sample_rate=OUT_SAMPLE_RATE,
        settings=PiperTTSService.Settings(voice=initial_voice),
    )

    context = LLMContext(messages=[{"role": "system", "content": get_system_prompt(session.language)}], tools=TOOLS)
    controller = SessionController(session=session, context=context)
    user_stop_strategy = BharatUserTurnStopStrategy(session=session, user_speech_timeout=TURN_SPEECH_TIMEOUT)
    transcript_tap = TranscriptTap(session=session, controller=controller, user_stop_strategy=user_stop_strategy)

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(stop=[user_stop_strategy]),
            empty_user_turn=None,
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        controller,
        vad,
        stt,
        transcript_tap,
        aggregators.user(),
        llm,
        BotTextTap(session=session),
        tts,
        transport.output(),
        aggregators.assistant(),
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=IN_SAMPLE_RATE, audio_out_sample_rate=OUT_SAMPLE_RATE),
        idle_timeout_secs=600,
        enable_rtvi=False,
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _ws):
        logger.info(f"[{session.session_id}] client connected (lang={session.language})")
        session.stage = "AWAITING_DESTINATION"
        await worker.queue_frames([ui_message({"type": "CONNECTION_STATUS", "status": "CONNECTED"})])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _ws):
        logger.info(f"[{session.session_id}] client disconnected")
        await worker.cancel()

    @transport.event_handler("on_session_timeout")
    async def on_timeout(_transport, _ws):
        await worker.cancel()

    return worker


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    # Immediate status handshake frame
    await websocket.send_text(json.dumps({"type": "CONNECTION_STATUS", "status": "CONNECTED"}))

    if len(SESSIONS) >= MAX_SESSIONS:
        await websocket.send_text(json.dumps({"type": "ERROR", "error": "Server busy, try again shortly"}))
        await websocket.close(code=1013)
        return

    session = Session()
    q = websocket.query_params
    if "lat" in q and "lon" in q:
        try:
            session.lat, session.lon, session.gps_source = float(q["lat"]), float(q["lon"]), "query"
        except ValueError:
            pass
    if "lang" in q and q["lang"].lower() in ALLOWED_LANGS:
        session.language = q["lang"].lower()

    SESSIONS[session.session_id] = session
    logger.info(f"[{session.session_id}] session opened ({len(SESSIONS)}/{MAX_SESSIONS}) lang={session.language}")
    try:
        worker = build_worker(websocket, session)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await runner.run()
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[{session.session_id}] pipeline error: {e}")
    finally:
        SESSIONS.pop(session.session_id, None)
        logger.info(f"[{session.session_id}] session closed (stage={session.stage}, turns={session.turns})")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
