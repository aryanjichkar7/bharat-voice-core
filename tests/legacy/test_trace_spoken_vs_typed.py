"""
Script to compare spoken vs typed destination request on live WebSocket server.
Traces:
1. final transcript
2. processing handler
3. destination search
4. route/fare
5. BOT_RESPONSE
6. TTS audio output
"""
import asyncio
import json
import time
import numpy as np
import soxr
import websockets
from app import VOICES_DIR, _CachedPiperVoice, IN_SAMPLE_RATE

async def run_trace():
    url = "ws://127.0.0.1:8000/ws/voice?lat=21.1458&lon=79.0882&lang=en"
    phrase = "I would like to go to Ramdas Pet."
    
    # Pre-generate 16kHz mono audio for the spoken phrase
    voice = _CachedPiperVoice.load(VOICES_DIR / "en_GB-cori-medium.onnx")
    raw_pcm = b""
    for chunk in voice.synthesize(phrase):
        raw_pcm += chunk.audio_int16_bytes

    audio_i16 = np.frombuffer(raw_pcm, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    resampled_f32 = soxr.resample(audio_f32, voice.config.sample_rate, IN_SAMPLE_RATE)
    resampled_i16 = (np.clip(resampled_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    audio_16k = resampled_i16.tobytes()
    print(f"Generated {len(audio_16k)} bytes of 16kHz PCM audio ({len(audio_16k)/(2*IN_SAMPLE_RATE):.2f}s)")

    # ==========================
    # 1. TYPED PATH
    # ==========================
    print("\n--- [START] TRACING TYPED PATH ---")
    typed_log = []
    async with websockets.connect(url, open_timeout=30) as ws:
        # Start session
        await ws.send(json.dumps({"type": "START_VOICE_SESSION"}))
        # Drain greeting
        while True:
            msg = await ws.recv()
            if isinstance(msg, str) and json.loads(msg).get("type") == "BOT_RESPONSE":
                break
        await asyncio.sleep(0.5)

        print(f"Typed: Sending TEXT_INPUT: {phrase!r}")
        t0 = time.time()
        await ws.send(json.dumps({"type": "TEXT_INPUT", "text": phrase}))

        while time.time() - t0 < 20.0:
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=10.0)
                if isinstance(frame, str):
                    d = json.loads(frame)
                    t = d.get("type")
                    typed_log.append((time.time() - t0, "TEXT", t, d))
                    print(f"  [TYPED +{time.time()-t0:.2f}s] {t}: {d.get('text', d.get('status', ''))}")
                    if t == "BOT_RESPONSE":
                        print(f"  [TYPED] Spoken text: {d.get('text')}")
                elif isinstance(frame, bytes):
                    typed_log.append((time.time() - t0, "AUDIO", len(frame), None))
            except asyncio.TimeoutError:
                break

    # ==========================
    # 2. SPOKEN PATH
    # ==========================
    print("\n--- [START] TRACING SPOKEN PATH ---")
    spoken_log = []
    async with websockets.connect(url, open_timeout=30) as ws:
        # Start session
        await ws.send(json.dumps({"type": "START_VOICE_SESSION"}))
        # Drain greeting
        while True:
            msg = await ws.recv()
            if isinstance(msg, str) and json.loads(msg).get("type") == "BOT_RESPONSE":
                break
        await asyncio.sleep(0.5)

        print(f"Spoken: Streaming {len(audio_16k)} bytes of microphone audio in 20ms chunks...")
        t0 = time.time()
        chunk_size = 640  # 20ms at 16kHz
        for i in range(0, len(audio_16k), chunk_size):
            await ws.send(audio_16k[i:i+chunk_size])
            await asyncio.sleep(0.018)

        silence = b"\x00" * 640
        for _ in range(30):
            await ws.send(silence)
            await asyncio.sleep(0.018)

        print("Spoken: Finished streaming audio and silence. Waiting for server pipeline events...")
        while time.time() - t0 < 25.0:
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=10.0)
                if isinstance(frame, str):
                    d = json.loads(frame)
                    t = d.get("type")
                    spoken_log.append((time.time() - t0, "TEXT", t, d))
                    print(f"  [SPOKEN +{time.time()-t0:.2f}s] {t}: {d.get('text', d.get('status', ''))}")
                    if t == "BOT_RESPONSE":
                        print(f"  [SPOKEN] Spoken text: {d.get('text')}")
                elif isinstance(frame, bytes):
                    spoken_log.append((time.time() - t0, "AUDIO", len(frame), None))
            except asyncio.TimeoutError:
                break

    # ==========================
    # COMPARISON REPORT
    # ==========================
    print("\n==========================================")
    print("STAGE COMPARISON REPORT")
    print("==========================================")
    stages = [
        "1. final transcript (USER_TRANSCRIPT)",
        "2. processing handler / candidates (POI_CANDIDATES)",
        "3. destination search / diagnostics (PERF_DIAGNOSTICS)",
        "4. route/fare preview (MAP_ROUTE_PREVIEW)",
        "5. bot response (BOT_RESPONSE)",
        "6. tts audio output (AUDIO_RAW frames)",
    ]
    typed_types = [x[2] for x in typed_log if x[1] == "TEXT"]
    typed_audio_count = sum(1 for x in typed_log if x[1] == "AUDIO")
    spoken_types = [x[2] for x in spoken_log if x[1] == "TEXT"]
    spoken_audio_count = sum(1 for x in spoken_log if x[1] == "AUDIO")

    print(f"Typed text events: {typed_types}")
    print(f"Typed audio frame count: {typed_audio_count}")
    print(f"Spoken text events: {spoken_types}")
    print(f"Spoken audio frame count: {spoken_audio_count}")

if __name__ == "__main__":
    asyncio.run(run_trace())
