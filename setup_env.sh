#!/usr/bin/env bash
# Phase 1 environment setup for Bharat Online Cabs voice core.
set -euo pipefail
cd "$(dirname "$0")"

HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
OLLAMA_URL="http://localhost:11434"

# ---- 1. Pick a Python 3.10-3.12 interpreter (3.13+/3.14 lack wheels for some deps)
PY=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "❌ Python 3.10-3.12 required"; exit 1; }
echo "🐍 Using $($PY --version)"

# ---- 2. venv + deps
[ -d venv ] || "$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools -q
pip install -r requirements.txt

# ---- 3. Piper voices
mkdir -p voices
# Format: "requested_path|fallback_path"  (requested names don't exist upstream; fallbacks are verified)
VOICES=(
  "hi/hi_IN/patricia/medium/hi_IN-patricia-medium|hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium"
  "en/en_IN/cora/medium/en_IN-cora-medium|en/en_GB/cori/medium/en_GB-cori-medium"
  "mr/mr_IN/google/medium/mr_IN-google-medium|mr/mr_IN/google/medium/mr_IN-google-medium"
)

download_voice() {
  local path="$1" name; name="$(basename "$path")"
  if [ -s "voices/$name.onnx" ] && [ -s "voices/$name.onnx.json" ]; then
    echo "✅ $name already present"; return 0
  fi
  if curl -fsSL -o "voices/$name.onnx" "$HF_BASE/$path.onnx" &&
     curl -fsSL -o "voices/$name.onnx.json" "$HF_BASE/$path.onnx.json"; then
    local size; size=$(stat -f%z "voices/$name.onnx")
    if [ "$size" -lt 1000000 ] || ! python -c "import json,sys;json.load(open(sys.argv[1]))" "voices/$name.onnx.json"; then
      echo "❌ $name failed validation"; rm -f "voices/$name".onnx*; return 1
    fi
    echo "✅ $name downloaded ($((size/1024/1024)) MB)"; return 0
  fi
  rm -f "voices/$name".onnx*; return 1
}

for entry in "${VOICES[@]}"; do
  req="${entry%%|*}"; fb="${entry##*|}"
  download_voice "$req" 2>/dev/null || {
    echo "⚠️  $(basename "$req") not found upstream, using $(basename "$fb")"
    download_voice "$fb" || { echo "❌ Voice download failed"; exit 1; }
  }
done

# ---- 4. Ollama check
if curl -fs "$OLLAMA_URL/api/tags" >/dev/null; then
  echo "✅ Ollama reachable at $OLLAMA_URL"
  curl -fs "$OLLAMA_URL/api/tags" | grep -q 'qwen2.5:3b' && echo "✅ qwen2.5:3b available" || echo "⚠️  qwen2.5:3b not pulled (run: ollama pull qwen2.5:3b)"
else
  echo "⚠️  Ollama NOT reachable at $OLLAMA_URL (run: ollama serve)"
fi

echo "🎉 Setup complete. Run: source venv/bin/activate && python test_env.py"
