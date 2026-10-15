# Bharat Online Cabs — Test Suites

The test suite is structured into two active test categories and one historical archive:

```
tests/
├── unit/           # Fast, isolated algorithmic, indexing, and parameter tests
├── integration/    # State machine, speech pipeline, audio lifecycle, and async race tests
└── legacy/         # Historical stage tests preserved for provenance and regression history
```

## Running Active Test Suites

Run all active unit tests:
```bash
python -m unittest discover -s tests/unit -p "test_*.py"
```

Run all active integration tests:
```bash
python -m unittest discover -s tests/integration -p "test_*.py"
```

Run a specific test suite:
```bash
# Dynamic destination understanding
python -m unittest tests/unit/test_dynamic_destination_understanding.py

# Conversation restart & cancellation lifecycle
python -m unittest tests/integration/test_stage2e_conversation_and_lifecycle.py

# Spoken turn finalisation & audio pipeline
python -m unittest tests/integration/test_stage2h_spoken_finalisation.py
```

## Summary of Active Suites

| Category | Suite | Test Count | Description |
|---|---|---|---|
| **Unit** | `test_dynamic_destination_understanding.py` | 10 | Multilingual destination extraction, zero hardcoded venues, parameter validation. |
| **Unit** | `test_stage2d_destination_recovery.py` | 16 | Typo recovery, phonetic matching, candidate ranking, candidate scoring. |
| **Unit** | `test_stage2g_local_osm_search.py` | 10 | SQLite FTS5 + R-Tree spatial indexing, bounding box enforcement, fast sub-50ms POI queries. |
| **Unit** | `test_osm_timeout_handling.py` | 4 | Nominatim timeout bounding, 504 provider handling, and graceful fallback. |
| **Unit** | `test_stage1_osm_migration.py` | 10 | OpenStreetMap / OSRM migration verification, Leaflet vendor assets, tariff formulas. |
| **Integration** | `test_stage2e_conversation_and_lifecycle.py` | 15 | Conversation state transitions, two-phase confirmation, ride cancellation, restart flow. |
| **Integration** | `test_stage2h_spoken_finalisation.py` | 11 | Spoken turn finalisation, VAD boundaries, audio worklet frame processing, in-flight cancellation. |
| **Integration** | `test_timeout_orchestration_race.py` | 4 | Async timeout races, watchdog timers, concurrent request serialization. |
| **Integration** | `test_voice_destination_freeze_repair.py` | 5 | Voice freeze repair, pipeline recovery, state transitions on destination resolution. |
| **Integration** | `test_stage2_speech_and_lifecycle.py` | 8 | End-to-end speech processing, Whisper STT integration, Piper TTS frames, booking confirmation data. |
| **Total** | **10 Active Test Files** | **93 Tests** | **All Passing** |
