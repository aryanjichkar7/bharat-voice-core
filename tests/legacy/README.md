# Legacy Test Suite Archive

This directory archives historical stage-specific regression and verification tests from earlier development phases of **Bharat Online Cabs** (`bharat-voice-core`).

Per repository preservation safeguards, these suites are retained in full for provenance, forensic tracing, and regression history.

## Archived Test Modules

| Test Module | Historical Origin | Superseded By | Reason Archived |
|---|---|---|---|
| `test_google_places_and_lifecycle.py` | Stage 0 / Initial Architecture | `tests/unit/test_stage1_osm_migration.py` | Built for legacy Google Places API (superseded by OpenStreetMap migration). |
| `test_spatial.py` | Phase 2 / Spatial Engine Prototype | `tests/unit/test_stage2g_local_osm_search.py` | Early spatial engine tests preceding local FTS5/RTree SQLite indexing. |
| `test_pipeline.py` | Phase 3 / Pipecat Pipeline Draft | `tests/integration/test_stage2_speech_and_lifecycle.py` | Initial pipeline frame processor draft before unified turn lifecycle. |
| `test_phase3_booking_and_lifecycle.py` | Phase 3 / Booking Flow | `tests/integration/test_stage2e_conversation_and_lifecycle.py` | Superseded by Stage 2E two-phase confirmation and cancellation state machine. |
| `test_enterprise_suite.py` | Phase 4 / Enterprise Polish | `tests/integration/test_stage2_speech_and_lifecycle.py` | Pre-OSM enterprise verification tests with early timeout assumptions. |
| `test_enterprise_polish.py` | Phase 4 / Enterprise Polish | `tests/integration/test_stage2_speech_and_lifecycle.py` | Pre-OSM UI assert checks. |
| `test_env.py` | Phase 1 / Environment Sanity | Standard `.env.example` audit | Early voice file presence check. |
| `test_p0_voice_session_regression.py` | P0 Voice Regression | `tests/integration/test_stage2h_spoken_finalisation.py` | Early regression script without standalone test discovery classes. |
| `test_trace_spoken_vs_typed.py` | Stage 2B Diagnostics | `tests/integration/test_stage2h_spoken_finalisation.py` | Early turn trace assertion runner. |
| `test_stage2c_microphone_to_booking.py` | Stage 2C Audio Turn Timeout | `tests/integration/test_stage2h_spoken_finalisation.py` | Contains hardcoded assertions for older 15,000ms idle timeout (updated to 5,000ms in Stage 2). |
| `test_voice_failure_isolation_and_repair.py` | Stage 1 / Early Voice Repair | `tests/unit/test_stage2d_destination_recovery.py` | Relies on legacy static phonetic variants replaced by dynamic fuzzy matching. |

## Running Active Tests

To run the active, fully verified test suites, execute:

```bash
# Run unit test suite (50 tests)
python -m unittest discover -s tests/unit -p "test_*.py"

# Run integration test suite (43 tests)
python -m unittest discover -s tests/integration -p "test_*.py"
```
