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
MAX_SESSIONS = int(os.getenv("BOC_MAX_SESSIONS", "4"))

IN_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 22050          # Piper medium voices are native 22.05 kHz -> no resampling
VAD_STOP_SECS = 0.3              # VAD silence before speech segment closes
TURN_SPEECH_TIMEOUT = 0.2        # extra wait for the user to resume -> ~500 ms turn end
ALLOWED_LANGS = ("en", "hi", "mr")
STT_HOTWORDS = "Nagpur Sitabuldi Dharampeth Civil Lines Ramdaspeth Sadar Haldiram DMart Londe Jewellers Bharat Online Cabs"

VOICE_MAP = {
    "en": "en_GB-cori-medium",
    "hi": "hi_IN-priyamvada-medium",
    "mr": "hi_IN-priyamvada-medium",
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
- Never invent places, distances, or fares. Rely exclusively on tool outputs.

BOOKING WORKFLOW & TOOLS
1. When the customer mentions any destination or category (e.g. jewelry shop, restaurant, landmark), call search_nearby_pois with that query.
2. If search_nearby_pois returns DISAMBIGUATION_NEEDED:
   Concierge Rule: Announce ONLY the single closest branch and its distance using the next_step instruction (e.g., "The closest jeweler near you in Dharampeth is [Name], about [X] meters away. Would you like to go there, or check the other options on your map?").
3. When trip details are calculated, announce: "The trip is [X] km and the estimated fare is [Y] rupees. Shall I book this ride?"
4. When the customer confirms, confirm the booking in one short sentence."""


# ----------------------------------------------------------------------------- disambiguation matching
def match_candidate_choice(text: str, options: list[dict]) -> dict | None:
    if not options:
        return None
    t = text.lower().strip()

    # 1. Direct ordinals or "closest / nearest / first"
    if any(k in t for k in ("first", "closest", "nearest", "paas", "pehla", "pehli", "option 1", "number 1", "1st")):
        return options[0]
    if any(k in t for k in ("second", "dusra", "dusri", "option 2", "number 2", "2nd")) and len(options) > 1:
        return options[1]
    if any(k in t for k in ("third", "teesra", "teesri", "option 3", "number 3", "3rd")) and len(options) > 2:
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
    affirmative = ("yes", "ha", "haan", "sure", "ok", "okay", "chalo", "theek", "kar do", "take me there")
    if any(k == t or t.startswith(k + " ") or t.endswith(" " + k) for k in affirmative):
        return options[0]

    return None


# ----------------------------------------------------------------------------- shared models
_whisper_model = None
_piper_cache: dict[str, object] = {}


def get_whisper_model():
    """Load faster-whisper once per process (int8 on CPU is fast & memory-efficient on M1)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        t0 = time.time()
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
        logger.info(f"Whisper '{WHISPER_MODEL}' loaded in {time.time() - t0:.1f}s")
    return _whisper_model


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
    lat: float = se.DEFAULT_LAT
    lon: float = se.DEFAULT_LON
    gps_source: str = "default"
    language: str = "en"
    stage: str = "GREETING"  # GREETING > AWAITING_DESTINATION > DISAMBIGUATING > AWAITING_CONFIRMATION > RIDE_ACTIVE
    is_bot_speaking: bool = False
    last_search: dict | None = None
    pending_ride: dict | None = None
    booking_id: str | None = None
    otp: str = "4821"
    driver_info: dict = field(default_factory=lambda: {
        "name": "Rajesh Kumar",
        "car": "White Swift Dzire",
        "plate": "MH 31 EQ 8421",
        "eta_mins": 3,
        "rating": 4.9,
        "phone": "+91 98220 12345"
    })
    turns: int = 0
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        if d["pending_ride"]:
            d["pending_ride"] = {k: v for k, v in d["pending_ride"].items() if k != "route_geometry"}
        d.pop("last_search", None)
        return d

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
            }))

    async def confirm_booking(self, push_fn=None) -> str | None:
        if not self.pending_ride:
            return None
        if not self.booking_id:
            self.booking_id = f"BOC-{uuid.uuid4().hex[:6].upper()}"
        self.stage = "RIDE_ACTIVE"
        if push_fn:
            ride = self.pending_ride
            await push_fn(ui_message({
                "type": "BOOKING_CONFIRMED",
                "booking_id": self.booking_id,
                "otp": self.otp,
                "driver": self.driver_info,
                "destination": ride["destination"],
                "fare_inr": ride["fare_inr"],
                "distance_km": ride["distance_km"],
                "duration_mins": ride["duration_mins"]
            }))
        return self.booking_id

    async def cancel_ride(self, push_fn=None):
        self.stage = "AWAITING_DESTINATION"
        self.pending_ride = None
        self.booking_id = None
        if push_fn:
            await push_fn(ui_message({"type": "RIDE_CANCELLED"}))


SESSIONS: dict[str, Session] = {}


def ui_message(payload: dict) -> OutputTransportMessageUrgentFrame:
    return OutputTransportMessageUrgentFrame(message=payload)


# ----------------------------------------------------------------------------- Dynamic Piper TTS
class DynamicPiperTTSService(PiperTTSService):
    """Dynamically switches Piper voice model according to session.language before synthesis."""

    def __init__(self, session: Session, **kwargs):
        self.session = session
        super().__init__(**kwargs)

    async def run_tts(self, text: str, context_id: str):
        voice_stem = resolve_voice_for_lang(self.session.language)
        voice_path = VOICES_DIR / f"{voice_stem}.onnx"
        if voice_path.exists():
            self._voice = _CachedPiperVoice.load(voice_path)

        self.session.is_bot_speaking = True
        try:
            async for frame in super().run_tts(text, context_id):
                yield frame
        finally:
            await asyncio.sleep(0.35)
            self.session.is_bot_speaking = False


# ----------------------------------------------------------------------------- Multilingual Whisper STT
class MultilingualWhisperSTT(SegmentedSTTService):
    """Whisper STT with language biasing and acoustic echo/hallucination filter."""

    @property
    def wants_wav_segments(self) -> bool:
        return False

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self._model = get_whisper_model()

    def _transcribe(self, audio: np.ndarray, biased_lang: str) -> tuple[str, str]:
        lang = biased_lang if biased_lang in ALLOWED_LANGS else "en"
        segments, _ = self._model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            condition_on_previous_text=False,
            hotwords=STT_HOTWORDS,
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
        return text, lang

    async def run_stt(self, audio: bytes):
        if self.session.is_bot_speaking:
            return

        audio_f32 = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_f32.size < IN_SAMPLE_RATE * 0.25:
            return

        t0 = time.time()
        biased = self.session.language
        text, lang = await asyncio.to_thread(self._transcribe, audio_f32, biased)
        if not text:
            return

        clean_lower = text.lower().strip()
        if len(clean_lower) <= 2 or clean_lower in WHISPER_HALLUCINATIONS or any(
            h in clean_lower for h in ("thank you for watching", "subtitles by", "please subscribe", "transcription by")
        ):
            logger.info(f"Filtered Whisper ghost hallucination: {text!r}")
            return

        logger.info(f"STT [{lang}] ({time.time() - t0:.2f}s): {text}")
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

    async def _resolve_and_preview_trip(self, lat: float, lon: float, name: str):
        """Deterministically calculate trip details, push MAP_ROUTE_PREVIEW, and speak confirmation."""
        logger.info(f"[{self.session.session_id}] Deterministic trip resolution -> {name} ({lat}, {lon})")
        trip = await asyncio.to_thread(se.calculate_trip_details, self.session.lat, self.session.lon, lat, lon, name)
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

        await self.push_frame(TTSSpeakFrame(spoken))
        await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": spoken}))
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
            spoken = f"आपकी {dest} के लिए कैब बुक हो चुकी है! बुकिंग आईडी {bid} है, और ड्राइवर 3 मिनट में पहुँचेगा।"
        elif lang == "mr":
            spoken = f"आपली {dest} साठी कॅब बुक झाली आहे! बुकिंग आयडी {bid} असून ड्रायव्हर 3 मिनिटांत पोहोचेल।"
        else:
            spoken = f"Your ride to {dest} is confirmed! Your booking ID is {bid}, and your driver will arrive in 3 minutes."

        await self.push_frame(TTSSpeakFrame(spoken))
        await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": spoken}))
        if self.context and self.context.messages:
            self.context.messages.append({"role": "assistant", "content": spoken})

    async def _handle_text_turn(self, text: str) -> bool:
        """Evaluates customer input against state machine rules."""
        t = text.lower().strip()

        # Handle active ride queries
        if self.session.stage == "RIDE_ACTIVE":
            if any(w in t for w in ("cancel", "radd", "band karo")):
                await self.session.cancel_ride(self.push_frame)
                lang = self.session.language
                if lang == "hi":
                    spoken = "आपकी राइड कैंसिल कर दी गई है। आप कहाँ जाना चाहेंगे?"
                elif lang == "mr":
                    spoken = "आपली राईड रद्द करण्यात आली आहे. आपण कुठे जाऊ इच्छिता?"
                else:
                    spoken = "Your ride has been cancelled. Where would you like to go instead?"
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": spoken}))
                return True

            if any(w in t for w in ("where", "driver", "when", "kahan", "kitna time", "arrival", "cab", "gaadi")):
                d = self.session.driver_info
                lang = self.session.language
                if lang == "hi":
                    spoken = f"आपके ड्राइवर {d['name']} {d['car']} में लगभग {d['eta_mins']} मिनट में पहुँच रहे हैं।"
                elif lang == "mr":
                    spoken = f"तुमचे ड्रायव्हर {d['name']} {d['car']} मध्ये सुमारे {d['eta_mins']} मिनिटांत पोहोचत आहेत."
                else:
                    spoken = f"Your driver {d['name']} in the {d['car']} is about {d['eta_mins']} minutes away."
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": spoken}))
                return True

            if any(w in t for w in ("otp", "pin", "booking id", "number", "code")):
                otp = self.session.otp
                bid = self.session.booking_id
                lang = self.session.language
                if lang == "hi":
                    spoken = f"आपकी बुकिंग आईडी {bid} है, और राइड ओटीपी {otp} है।"
                elif lang == "mr":
                    spoken = f"तुमचा बुकिंग आयडी {bid} असून राईड ओटीपी {otp} आहे."
                else:
                    spoken = f"Your booking ID is {bid}, and your ride OTP is {otp}."
                await self.push_frame(TTSSpeakFrame(spoken))
                await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": spoken}))
                return True

            return False

        # 1. Booking confirmation
        if self.session.stage == "AWAITING_CONFIRMATION" and any(w in t for w in CONFIRMATION_KEYWORDS):
            await self._confirm_current_booking()
            return True

        # 2. Candidate resolution during disambiguation
        if self.session.stage == "DISAMBIGUATING" and self.session.last_search:
            options = ([self.session.last_search["primary_match"]] if self.session.last_search.get("primary_match") else []) + self.session.last_search.get("alternatives", [])
            cand = match_candidate_choice(text, options)
            if cand:
                await self._resolve_and_preview_trip(cand["lat"], cand["lon"], cand["name"])
                return True

        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self.session.is_bot_speaking = False

        if isinstance(frame, InputTransportMessageFrame) and direction == FrameDirection.DOWNSTREAM:
            msg = frame.message or {}
            mtype = msg.get("type")

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
                    self.session.lat, self.session.lon, self.session.gps_source = lat, lon, "client"
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
                await self.session.cancel_ride(self.push_frame)
                return

            if mtype == "TEXT_INPUT" and str(msg.get("text", "")).strip():
                text = str(msg["text"]).strip()
                self.session.turns += 1
                await self.push_frame(ui_message({"type": "USER_TRANSCRIPT", "text": text, "source": "text"}))
                handled = await self._handle_text_turn(text)
                if not handled:
                    await self.push_frame(
                        LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True))
                return
            return
        await self.push_frame(frame, direction)


class TranscriptTap(FrameProcessor):
    """Mirrors final user transcripts to the UI and triggers state machine handling."""

    def __init__(self, session: Session, controller: SessionController, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.controller = controller

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return
            self.session.turns += 1
            await self.push_frame(ui_message({
                "type": "USER_TRANSCRIPT", "text": text, "source": "voice",
                "language": str(frame.language) if frame.language else None}))
            handled = await self.controller._handle_text_turn(text)
            if handled:
                return
        await self.push_frame(frame, direction)


class BotTextTap(FrameProcessor):
    """Aggregates streamed LLM tokens and sends each full reply to the UI."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buf: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._buf.append(frame.text)
        elif isinstance(frame, InterruptionFrame):
            self._buf.clear()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._buf:
            text = "".join(self._buf).strip()
            self._buf.clear()
            if text:
                await self.push_frame(ui_message({"type": "BOT_RESPONSE", "text": text}))
        await self.push_frame(frame, direction)


# ----------------------------------------------------------------------------- tools
TOOLS = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="search_nearby_pois",
        description="Find a destination near the customer's GPS location. Returns EXACT, "
                    "DISAMBIGUATION_NEEDED (several outlets) or NOT_FOUND with options sorted nearest first.",
        properties={"query": {"type": "string",
                              "description": "Destination place name only, e.g. 'Haldiram' or 'jewelry shop near me'"}},
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
        "google_maps_browser_key": os.getenv("GOOGLE_MAPS_BROWSER_API_KEY", "").strip(),
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
        settings=PiperTTSService.Settings(voice=initial_voice),
    )

    context = LLMContext(messages=[{"role": "system", "content": get_system_prompt(session.language)}], tools=TOOLS)
    controller = SessionController(session=session, context=context)
    transcript_tap = TranscriptTap(session=session, controller=controller)

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TURN_SPEECH_TIMEOUT)]),
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
        BotTextTap(),
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
        greeting = GREETINGS.get(session.language, GREETINGS["en"])
        await worker.queue_frames([ui_message({"type": "CONNECTION_STATUS", "status": "CONNECTED"})])
        await worker.queue_frames([TTSSpeakFrame(greeting)])
        await worker.queue_frames([ui_message({"type": "BOT_RESPONSE", "text": greeting})])

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
