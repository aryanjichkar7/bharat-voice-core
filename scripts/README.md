# Bharat Online Cabs — Utility Scripts

Utility scripts for data indexing, ingestion, and runtime verification.

## Available Scripts

### 1. `import_osm_places.py`
Idempotent OpenStreetMap Extract Importer CLI. Parses regional `.osm.pbf` or `.osm` XML extracts and populates `data/places.db` with SQLite FTS5 full-text search and R-Tree spatial indexing.

Usage:
```bash
# Check status of local OSM database
python scripts/import_osm_places.py --info

# Import places from an OSM extract
python scripts/import_osm_places.py --input data/nagpur.osm.pbf
```

### 2. `verify_web.py`
Phase 4 Frontend & Web Integration Verification script. Launches Uvicorn on a test port, validates static asset delivery, HTTP `/health` and `/api/config` endpoints, and live WebSocket handshake on `/ws/voice`.

Usage:
```bash
python scripts/verify_web.py
```
