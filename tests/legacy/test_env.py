"""Phase 1 smoke test: Silero VAD, Faster-Whisper (base), Piper voices, Ollama qwen2.5:3b."""
import glob
import json
import os
import time

import numpy as np
import psutil
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b"
proc = psutil.Process(os.getpid())
results = []  # (component, status, detail, seconds, rss_mb)


def rss_mb() -> float:
    return proc.memory_info().rss / 1024 / 1024


def check(name):
    def deco(fn):
        t0 = time.time()
        try:
            detail = fn()
            ok = True
        except Exception as e:  # noqa: BLE001
            detail, ok = f"{type(e).__name__}: {e}"[:60], False
        results.append((name, ok, detail, time.time() - t0, rss_mb()))
        return fn
    return deco


@check("Silero VAD")
def _vad():
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    vad = SileroVADAnalyzer(sample_rate=16000)
    vad.set_sample_rate(16000)
    conf = vad.voice_confidence(np.zeros(vad.num_frames_required(), dtype=np.int16).tobytes())
    return f"loaded, silence conf={float(np.asarray(conf).ravel()[0]):.2f}"


@check("Faster-Whisper base")
def _whisper():
    from faster_whisper import WhisperModel

    m = WhisperModel("base", device="cpu", compute_type="int8")
    segs, info = m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
    list(segs)
    return f"int8 CPU, lang-detect ok ({info.language})"


@check("Piper voices")
def _piper():
    onnx = sorted(glob.glob("voices/*.onnx"))
    assert onnx, "no .onnx files in ./voices"
    names = []
    for f in onnx:
        cfg = json.load(open(f + ".json"))
        names.append(f"{os.path.basename(f)[:-5]}@{cfg['audio']['sample_rate']}Hz")
    try:
        from piper import PiperVoice

        v = PiperVoice.load(onnx[0])
        del v
        names.append("load ok")
    except ImportError:
        names.append("piper module missing")
    return ", ".join(names)


@check(f"Ollama {MODEL}")
def _ollama():
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}},
        timeout=60,
    )
    r.raise_for_status()
    return f"reply={r.json().get('response', '')!r}"


vm = psutil.virtual_memory()
w = (22, 6, 62, 7, 8)
line = "+" + "+".join("-" * (x + 2) for x in w) + "+"
print("\n" + line)
print("| " + " | ".join(h.ljust(x) for h, x in zip(("Component", "Status", "Detail", "Time s", "RSS MB"), w)) + " |")
print(line)
for n, ok, d, t, m in results:
    row = (n, "PASS" if ok else "FAIL", str(d)[: w[2]], f"{t:.1f}", f"{m:.0f}")
    print("| " + " | ".join(c.ljust(x) for c, x in zip(row, w)) + " |")
print(line)
print(f"System RAM: {vm.total/2**30:.1f} GB total | {vm.available/2**30:.1f} GB available | "
      f"{vm.percent:.0f}% used | swap used {psutil.swap_memory().used/2**30:.2f} GB")
ready = all(r[1] for r in results)
print(("✅ READY for Phase 2" if ready else "❌ NOT READY - fix failures above") + "\n")
raise SystemExit(0 if ready else 1)
