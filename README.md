# Bharat Online Cabs (`bharat-voice-core`)

An intelligent, voice-first cab booking assistant tailored for India, featuring local speech recognition, multilingual neural text-to-speech, deterministic conversation flow, and high-performance local OpenStreetMap spatial indexing.

## Quick Start

### 1. Environment Setup
```bash
# Set up Python virtual environment and dependencies
./setup_env.sh
source venv/bin/activate
```

### 2. Start the Application Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at `http://localhost:8000` to interact with Disha, the voice assistant.

### 3. Verify System Health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/config
```

---

## Repository Structure

```
bharat-voice-core/
├── app.py                      # FastAPI server, WebSocket handler, and Pipecat voice pipeline
├── spatial_engine.py           # Local OSM search, fuzzy matching, OSRM routing, tariffs
├── osm_indexer.py              # SQLite FTS5 + R-Tree indexing engine
├── data/
│   └── places.db               # Local Nagpur OpenStreetMap database (3,862 POIs)
├── static/                     # Web UI (HTML, CSS, JS, AudioWorklet, Leaflet)
├── voices/                     # Piper ONNX neural voices (English, Hindi, Marathi)
├── scripts/                    # Management and verification CLI scripts
│   ├── import_osm_places.py    # Idempotent OSM extract importer CLI
│   └── verify_web.py           # Web and WebSocket integration verification
├── tests/                      # Automated test suites
│   ├── unit/                   # Algorithmic and isolated component tests (50 tests)
│   ├── integration/            # Multi-component and voice pipeline tests (43 tests)
│   └── legacy/                 # Historical regression suites archive
├── docs/                       # Architecture and design documentation
│   └── ARCHITECTURE.md         # Detailed system architecture
├── setup_env.sh                # Environment setup and voice downloader
├── requirements.txt            # Python dependencies
└── .env.example                # Configuration parameters template
```

---

## Running Automated Tests

The test suite contains 93 active automated tests covering unit components and end-to-end integration flows.

### Run Unit Tests
```bash
python -m unittest discover -s tests/unit -p "test_*.py"
```

### Run Integration Tests
```bash
python -m unittest discover -s tests/integration -p "test_*.py"
```

### Run a Specific Test
```bash
python -m unittest tests/unit/test_dynamic_destination_understanding.py
python -m unittest tests/integration/test_stage2e_conversation_and_lifecycle.py
```

See [tests/README.md](tests/README.md) for details on test coverage.

---

## Key Features

- **Multilingual Voice Pipeline**: Offline STT via Faster-Whisper with VAD filtering, coupled with high-speed neural TTS via Piper.
- **Local Spatial Search**: SQLite FTS5 full-text search and R-Tree spatial indexing querying 3,862 POIs with sub-50ms latency.
- **Zero Hardcoding**: Dynamically resolves destinations and handles phonetically similar mispronunciations without static venue dictionaries.
- **Two-Phase Confirmation**: Transparent clarification for low-confidence matches before committing to route calculation.
- **Deterministic Booking & Fares**: Minimum ₹70, ₹50 base fare, and ₹14/km calculated via OSRM road geometry.
