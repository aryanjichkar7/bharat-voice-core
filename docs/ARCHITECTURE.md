# Bharat Online Cabs — Architecture & System Overview

**Bharat Online Cabs** (`bharat-voice-core`) is an enterprise-grade multilingual voice assistant for cab booking operating in the Nagpur metropolitan area.

## Key Capabilities

1. **Multilingual Speech-to-Text (STT)**: Faster-Whisper with VAD filtering and hotword prompting (English, Hindi, Marathi).
2. **Local Piper Text-to-Speech (TTS)**: Offline ONNX neural voices (`en_GB-cori-medium`, `hi_IN-priyamvada-medium`, `mr_IN-google-medium`).
3. **Local OpenStreetMap Spatial Engine**:
   - High-performance SQLite FTS5 (full-text search) and R-Tree (spatial bounding box) index (`data/places.db`).
   - Sub-50ms POI retrieval across 3,800+ real places.
   - Dynamic fuzzy matching, Levenshtein distance, and Double Metaphone phonetic indexing without hardcoded location dictionaries.
   - Public Nominatim fallback with strict 2.5s timeouts and 504 circuit breaker.
   - Road network routing and driving distance/duration calculations via OSRM.
4. **Deterministic Conversation State Machine**:
   - Two-phase ride confirmation: `AWAITING_DESTINATION_CONFIRMATION` (for fuzzy disambiguation) and `AWAITING_CONFIRMATION` (trip route & fare).
   - Clean ride cancellation and seamless conversation restart.
5. **Real-Time Web Frontend**:
   - Responsive Leaflet.js map display with road polylines.
   - WebSocket streaming on `/ws/voice`.
   - AudioWorklet for low-latency PCM16 microphone streaming.

## Repository Layout

```
bharat-voice-core/
├── app.py                     # Primary FastAPI & Pipecat application entry point
├── spatial_engine.py          # Spatial search, routing, and tariff calculation engine
├── osm_indexer.py             # SQLite FTS5 + R-Tree indexing engine for OSM data
├── data/
│   └── places.db              # Indexed local OSM database (3,862 POIs in Nagpur)
├── static/                    # Frontend web application assets
│   ├── index.html             # Single-page interface with Leaflet map & voice orb
│   ├── app.js                 # Frontend state machine, WebSocket, and audio controller
│   ├── pcm-recorder-worklet.js# Low-latency AudioWorklet for 16kHz PCM capture
│   ├── style.css              # Styling and responsive design
│   └── vendor/                # Vendored Leaflet CSS and JS
├── voices/                    # Piper ONNX neural voice models & configs
├── scripts/                   # Administrative and verification utilities
│   ├── import_osm_places.py   # CLI tool to index OSM extracts into places.db
│   ├── verify_web.py          # HTTP & WebSocket smoke-test script
│   └── README.md              # Utility scripts documentation
├── tests/                     # Organized automated test suites
│   ├── unit/                  # Fast, isolated unit tests (50 tests)
│   ├── integration/           # Pipeline and lifecycle integration tests (43 tests)
│   ├── legacy/                # Historical stage tests archive
│   └── README.md              # Test documentation & execution guide
├── docs/                      # Architectural documentation
│   └── ARCHITECTURE.md        # System architecture overview
├── setup_env.sh               # Environment setup script
├── requirements.txt           # Python package dependencies
├── .env.example               # Example configuration template
└── README.md                  # Project root README
```

## Startup Commands

Start the development server with live reload:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Health check and runtime config endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Application configuration & service area bounding box
curl http://localhost:8000/api/config
```
