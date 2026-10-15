"""
Bharat Online Cabs — Local OpenStreetMap Place Index & Search Engine.

Autonomous local spatial index for the Nagpur metropolitan mobility area.
Consumes OpenStreetMap extracts (.osm.pbf or .osm XML) and indexes named cab
destinations in SQLite with FTS5 full-text search and RTree spatial indexing.

Preserves OpenStreetMap attribution (ODbL 1.0).
Zero hardcoded place dictionaries, venue lists, or spelling hacks.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# Service area configuration: configurable via environment
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "21.1458"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "79.0882"))
DEFAULT_CITY_NAME = os.getenv("DEFAULT_CITY_NAME", "Nagpur")
NAGPUR_DEFAULT_LAT = DEFAULT_LAT
NAGPUR_DEFAULT_LON = DEFAULT_LON
DEFAULT_SERVICE_BBOX = (
    float(os.getenv("SERVICE_BBOX_MIN_LAT", "20.92")),
    float(os.getenv("SERVICE_BBOX_MIN_LON", "78.85")),
    float(os.getenv("SERVICE_BBOX_MAX_LAT", "21.37")),
    float(os.getenv("SERVICE_BBOX_MAX_LON", "79.33")),
)  # min_lat, min_lon, max_lat, max_lon

OSM_ATTRIBUTION = "Data © OpenStreetMap contributors, ODbL 1.0"
OSM_LICENSE_URL = "https://www.openstreetmap.org/copyright"

DEFAULT_DB_PATH = os.getenv("OSM_LOCAL_DB_PATH", "data/places.db")


def normalize_token(text: str) -> str:
    """Normalize query or place name for indexed retrieval."""
    if not text:
        return ""
    # Lowercase, strip accents/diacritics, simplify spaces
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates spherical distance in km using the haversine formula."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _phonetic_skeleton(s: str) -> str:
    """Generates phonetic consonant skeleton for transliterated toponyms."""
    s = s.lower().strip()
    s = re.sub(r"c(?=[ei])", "s", s)
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"w", "v", s)
    s = re.sub(r"(?:tion|sion|shun)\b", "shn", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    tokens = re.findall(r"[a-z0-9]+", s)
    res_tokens = []
    for tok in tokens:
        if not tok:
            continue
        first = tok[0]
        rest = re.sub(r"[aeiou]", "", tok[1:])
        res_tokens.append(first + rest)
    return " ".join(res_tokens)


def init_database(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initializes the SQLite schema with places table, FTS5 virtual table, and RTree."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Places base table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS places (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            osm_id INTEGER NOT NULL,
            osm_type TEXT NOT NULL,
            name TEXT NOT NULL,
            name_normalized TEXT NOT NULL,
            name_compacted TEXT,
            phonetic TEXT,
            name_alt TEXT,
            category TEXT,
            feature_class TEXT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            short_address TEXT,
            raw_tags TEXT,
            provenance TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_places_name_norm ON places(name_normalized);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_places_name_comp ON places(name_compacted);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_places_phonetic ON places(phonetic);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_places_category ON places(category);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_places_feature ON places(feature_class);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_places_osm ON places(osm_type, osm_id);")

    # FTS5 Full-Text Search virtual table
    try:
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS places_fts USING fts5(
                name,
                name_alt,
                short_address,
                content='places',
                content_rowid='id'
            )
        """)
    except sqlite3.OperationalError as e:
        logger.warning(f"FTS5 virtual table initialization notice: {e}")

    # RTree Spatial Index
    try:
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS places_rtree USING rtree(
                id,
                min_lat, max_lat,
                min_lon, max_lon
            )
        """)
    except sqlite3.OperationalError as e:
        logger.warning(f"RTree spatial index initialization notice: {e}")

    # Metadata & Attribution Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS osm_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cur.execute("INSERT OR REPLACE INTO osm_metadata (key, value) VALUES (?, ?)", ("attribution", OSM_ATTRIBUTION))
    cur.execute("INSERT OR REPLACE INTO osm_metadata (key, value) VALUES (?, ?)", ("license", OSM_LICENSE_URL))

    con.commit()
    return con


def classify_osm_tags(tags: dict[str, str]) -> tuple[str, str] | None:
    """
    Classifies OSM tags into a destination feature_class and category.
    Returns (feature_class, category) or None if not relevant as a cab destination.
    """
    if "name" not in tags or not tags["name"].strip():
        return None

    # 1. Localities & Neighborhoods
    if "place" in tags:
        p = tags["place"]
        if p in ("suburb", "neighbourhood", "quarter", "village", "hamlet", "locality", "town", "city_block"):
            return ("locality", f"place/{p}")

    # 2. Public Amenities & Institutions
    if "amenity" in tags:
        a = tags["amenity"]
        if a in (
            "hospital", "clinic", "pharmacy", "doctors", "bank", "atm",
            "restaurant", "cafe", "fast_food", "college", "university",
            "school", "bus_station", "fuel", "police", "post_office",
            "cinema", "place_of_worship", "courthouse", "townhall",
            "library", "theatre", "community_centre", "marketplace"
        ):
            return ("amenity", f"amenity/{a}")

    # 3. Shops & Commercial
    if "shop" in tags:
        return ("shop", f"shop/{tags['shop']}")

    # 4. Tourism & Lodging
    if "tourism" in tags:
        t = tags["tourism"]
        if t in ("hotel", "guest_house", "motel", "hostel", "attraction", "museum", "zoo", "viewpoint", "theme_park"):
            return ("tourism", f"tourism/{t}")

    # 5. Leisure & Recreation
    if "leisure" in tags:
        l = tags["leisure"]
        if l in ("park", "garden", "sports_centre", "stadium", "playground", "fitness_centre", "pitch"):
            return ("leisure", f"leisure/{l}")

    # 6. Transport Hubs
    if "railway" in tags and tags["railway"] in ("station", "halt", "subway_entrance"):
        return ("transport", f"railway/{tags['railway']}")
    if "aeroway" in tags and tags["aeroway"] in ("aerodrome", "terminal"):
        return ("transport", f"aeroway/{tags['aeroway']}")

    # 7. Named Roads / Streets
    if "highway" in tags and tags["highway"] in ("primary", "secondary", "tertiary", "trunk", "residential", "motorway"):
        return ("street", f"highway/{tags['highway']}")

    # 8. Offices
    if "office" in tags:
        return ("office", f"office/{tags['office']}")

    return None


def extract_address(tags: dict[str, str]) -> str:
    """Extracts short address/area locality information from address tags."""
    parts = []
    for k in ("addr:suburb", "addr:neighbourhood", "addr:district", "addr:street", "addr:city", "is_in"):
        if k in tags and tags[k].strip() and tags[k].strip() not in parts:
            parts.append(tags[k].strip())
    return ", ".join(parts) if parts else ""


def extract_alt_names(tags: dict[str, str]) -> str:
    """Collects alternate and multilingual names from OSM tags."""
    alt = []
    for k, v in tags.items():
        if k in ("name:en", "name:hi", "name:mr", "alt_name", "loc_name", "official_name", "short_name", "int_name"):
            val = v.strip()
            if val and val not in alt and val != tags.get("name"):
                alt.append(val)
    return ", ".join(alt) if alt else ""


class OSMPlaceImporter:
    """Idempotent importer consuming OpenStreetMap extracts into the local SQLite database."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, bbox: tuple[float, float, float, float] = DEFAULT_SERVICE_BBOX):
        self.db_path = db_path
        self.bbox = bbox  # min_lat, min_lon, max_lat, max_lon
        self.min_lat, self.min_lon, self.max_lat, self.max_lon = bbox

    def _is_within_bbox(self, lat: float, lon: float) -> bool:
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)

    def import_pbf(self, pbf_path: str | Path) -> dict[str, Any]:
        """Imports an OSM PBF file using pyosmium."""
        try:
            import osmium
        except ImportError as e:
            raise RuntimeError(f"osmium library is required for PBF import: {e}")

        pbf_file = Path(pbf_path).resolve()
        if not pbf_file.exists():
            raise FileNotFoundError(f"PBF extract file not found: {pbf_file}")

        t0 = time.time()
        logger.info(f"Starting PBF import from {pbf_file} ({pbf_file.stat().st_size / (1024*1024):.2f} MB)...")

        con = init_database(self.db_path)
        cur = con.cursor()

        # Idempotent cleanup for this provenance
        provenance = pbf_file.name
        cur.execute("DELETE FROM places WHERE provenance = ?", (provenance,))
        try:
            cur.execute("DELETE FROM places_fts WHERE rowid NOT IN (SELECT id FROM places)")
            cur.execute("DELETE FROM places_rtree WHERE id NOT IN (SELECT id FROM places)")
        except Exception:
            pass
        con.commit()

        batch: list[tuple] = []
        counts: dict[str, int] = {}
        total_imported = 0

        importer_ref = self

        class EntityHandler(osmium.SimpleHandler):
            def node(self, n):
                nonlocal total_imported
                try:
                    lat, lon = n.location.lat, n.location.lon
                except (ValueError, osmium.InvalidLocationError):
                    return
                if not importer_ref._is_within_bbox(lat, lon):
                    return
                tags = {t.k: t.v for t in n.tags}
                cls = classify_osm_tags(tags)
                if not cls:
                    return
                feat_class, cat = cls
                name = tags["name"].strip()
                name_norm = normalize_token(name)
                name_comp = name_norm.replace(" ", "")
                phon = _phonetic_skeleton(name)
                alt = extract_alt_names(tags)
                addr = extract_address(tags)
                raw = json.dumps({k: v for k, v in tags.items() if len(v) < 200}, ensure_ascii=False)

                batch.append((
                    n.id, "node", name, name_norm, name_comp, phon, alt, cat, feat_class,
                    round(lat, 7), round(lon, 7), addr, raw, provenance
                ))
                counts[feat_class] = counts.get(feat_class, 0) + 1
                total_imported += 1
                if len(batch) >= 2000:
                    importer_ref._flush_batch(cur, con, batch)
                    batch.clear()

            def way(self, w):
                nonlocal total_imported
                if not len(w.nodes):
                    return
                tags = {t.k: t.v for t in w.tags}
                cls = classify_osm_tags(tags)
                if not cls:
                    return
                try:
                    lat = sum(n.location.lat for n in w.nodes) / len(w.nodes)
                    lon = sum(n.location.lon for n in w.nodes) / len(w.nodes)
                except (ValueError, osmium.InvalidLocationError):
                    return
                if not importer_ref._is_within_bbox(lat, lon):
                    return

                feat_class, cat = cls
                name = tags["name"].strip()
                name_norm = normalize_token(name)
                name_comp = name_norm.replace(" ", "")
                phon = _phonetic_skeleton(name)
                alt = extract_alt_names(tags)
                addr = extract_address(tags)
                raw = json.dumps({k: v for k, v in tags.items() if len(v) < 200}, ensure_ascii=False)

                batch.append((
                    w.id, "way", name, name_norm, name_comp, phon, alt, cat, feat_class,
                    round(lat, 7), round(lon, 7), addr, raw, provenance
                ))
                counts[feat_class] = counts.get(feat_class, 0) + 1
                total_imported += 1
                if len(batch) >= 2000:
                    importer_ref._flush_batch(cur, con, batch)
                    batch.clear()

        handler = EntityHandler()
        handler.apply_file(str(pbf_file), locations=True, idx="flex_mem")
        if batch:
            self._flush_batch(cur, con, batch)
            batch.clear()

        # Rebuild FTS index to ensure external content matches places table
        try:
            cur.execute("INSERT INTO places_fts(places_fts) VALUES('rebuild')")
            con.commit()
        except Exception as e:
            logger.warning(f"FTS rebuild notice: {e}")

        # Record metadata
        dur = time.time() - t0
        meta = {
            "source_file": provenance,
            "source_size_bytes": pbf_file.stat().st_size,
            "import_duration_seconds": round(dur, 2),
            "import_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_entities": total_imported,
            "counts_by_category": counts,
            "bbox": list(self.bbox),
        }
        cur.execute("INSERT OR REPLACE INTO osm_metadata (key, value) VALUES (?, ?)",
                    ("last_import", json.dumps(meta)))
        con.commit()
        con.close()

        logger.info(f"PBF import finished in {dur:.2f}s: {total_imported} places imported ({counts})")
        return meta

    def import_xml(self, xml_path: str | Path) -> dict[str, Any]:
        """Imports an OSM XML (.osm) file using iterparse (memory-efficient)."""
        import xml.etree.ElementTree as ET

        xml_file = Path(xml_path).resolve()
        if not xml_file.exists():
            raise FileNotFoundError(f"OSM XML extract file not found: {xml_file}")

        t0 = time.time()
        logger.info(f"Starting OSM XML import from {xml_file}...")

        con = init_database(self.db_path)
        cur = con.cursor()

        provenance = xml_file.name
        cur.execute("DELETE FROM places WHERE provenance = ?", (provenance,))
        try:
            cur.execute("DELETE FROM places_fts WHERE rowid NOT IN (SELECT id FROM places)")
            cur.execute("DELETE FROM places_rtree WHERE id NOT IN (SELECT id FROM places)")
        except Exception:
            pass
        con.commit()

        batch: list[tuple] = []
        counts: dict[str, int] = {}
        total_imported = 0

        context = ET.iterparse(str(xml_file), events=("end",))
        for _, elem in context:
            if elem.tag == "node":
                try:
                    lat = float(elem.attrib.get("lat", 0))
                    lon = float(elem.attrib.get("lon", 0))
                    node_id = int(elem.attrib.get("id", 0))
                except (ValueError, TypeError):
                    elem.clear()
                    continue

                if self._is_within_bbox(lat, lon):
                    tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in elem.findall("tag")}
                    cls = classify_osm_tags(tags)
                    if cls:
                        feat_class, cat = cls
                        name = tags["name"].strip()
                        name_norm = normalize_token(name)
                        name_comp = name_norm.replace(" ", "")
                        phon = _phonetic_skeleton(name)
                        alt = extract_alt_names(tags)
                        addr = extract_address(tags)
                        raw = json.dumps(tags, ensure_ascii=False)

                        batch.append((
                            node_id, "node", name, name_norm, name_comp, phon, alt, cat, feat_class,
                            round(lat, 7), round(lon, 7), addr, raw, provenance
                        ))
                        counts[feat_class] = counts.get(feat_class, 0) + 1
                        total_imported += 1
                        if len(batch) >= 1000:
                            self._flush_batch(cur, con, batch)
                            batch.clear()
                elem.clear()

        if batch:
            self._flush_batch(cur, con, batch)
            batch.clear()

        try:
            cur.execute("INSERT INTO places_fts(places_fts) VALUES('rebuild')")
            con.commit()
        except Exception as e:
            logger.warning(f"FTS rebuild notice: {e}")

        dur = time.time() - t0
        meta = {
            "source_file": provenance,
            "source_size_bytes": xml_file.stat().st_size,
            "import_duration_seconds": round(dur, 2),
            "import_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_entities": total_imported,
            "counts_by_category": counts,
            "bbox": list(self.bbox),
        }
        cur.execute("INSERT OR REPLACE INTO osm_metadata (key, value) VALUES (?, ?)",
                    ("last_import", json.dumps(meta)))
        con.commit()
        con.close()
        logger.info(f"OSM XML import finished in {dur:.2f}s: {total_imported} places imported ({counts})")
        return meta

    def _flush_batch(self, cur: sqlite3.Cursor, con: sqlite3.Connection, batch: list[tuple]):
        cur.executemany("""
            INSERT INTO places (
                osm_id, osm_type, name, name_normalized, name_compacted, phonetic,
                name_alt, category, feature_class, lat, lon, short_address, raw_tags, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)

        # Update FTS5 & RTree for the newly inserted batch
        cur.execute("""
            INSERT INTO places_fts (rowid, name, name_alt, short_address)
            SELECT id, name, name_alt, short_address FROM places
            WHERE id NOT IN (SELECT rowid FROM places_fts)
        """)
        cur.execute("""
            INSERT INTO places_rtree (id, min_lat, max_lat, min_lon, max_lon)
            SELECT id, lat, lat, lon, lon FROM places
            WHERE id NOT IN (SELECT id FROM places_rtree)
        """)
        con.commit()


def get_index_stats(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Inspects database row counts by feature category, database size, and provenance."""
    p = Path(db_path)
    if not p.exists():
        return {
            "database_exists": False,
            "database_path": str(p),
            "total_rows": 0,
            "categories": {},
            "size_bytes": 0,
            "metadata": None,
        }

    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT count(*) as cnt FROM places")
    total = cur.fetchone()["cnt"]

    cur.execute("SELECT feature_class, count(*) as cnt FROM places GROUP BY feature_class")
    cats = {row["feature_class"]: row["cnt"] for row in cur.fetchall()}

    cur.execute("SELECT value FROM osm_metadata WHERE key = 'last_import'")
    row = cur.fetchone()
    last_import = json.loads(row["value"]) if row else None

    con.close()
    return {
        "database_exists": True,
        "database_path": str(p),
        "total_rows": total,
        "categories": cats,
        "size_bytes": p.stat().st_size,
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "metadata": last_import,
    }


def _query_variants(query: str) -> list[str]:
    """Generates generic toponym transliteration, spacing and suffix variants."""
    q = " ".join(query.split())
    variants = [q]
    spaced = re.sub(r"\b([A-Z])([A-Z][a-z])", r"\1 \2", q)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
    for v in (spaced, q.replace("-", " "), q.replace(" ", "")):
        if v not in variants and len(v) >= 3:
            variants.append(v)
    q_low = q.lower()

    if re.match(r"^c[ie]", q_low):
        variants.append("s" + q_low[1:])

    for v in list(variants):
        dedup = re.sub(r"([b-df-hj-np-tv-z])\1+", r"\1", v)
        if dedup not in variants:
            variants.append(dedup)

    suffix_pairs = [
        (r"path\b", "peth"),
        (r"peth\b", "path"),
        (r"pet\b", "peth"),
        (r"peth\b", "pet"),
        (r"ch[ao]wk\b", "chauk"),
        (r"chauk\b", "chowk"),
        (r"rly\b", "railway"),
        (r"railway\b", "rly"),
        (r"pur\b", "pore"),
        (r"pore\b", "pur"),
    ]
    for v in list(variants):
        for pat, rep in suffix_pairs:
            cand = re.sub(pat, rep, v, flags=re.IGNORECASE)
            if cand != v and cand not in variants:
                variants.append(cand)

    toponym_pattern = re.compile(r"(?:path|peth|pur|pore|chowk|chawk|chauk|wadi|vadi|nagar|ngr)\b", re.IGNORECASE)
    for v in list(variants):
        if toponym_pattern.search(v):
            for m in re.finditer(r"([b-df-hj-np-tv-z])i(?=([b-df-hj-np-tv-z]))", v, flags=re.IGNORECASE):
                start = m.start()
                alt = v[:start+1] + "a" + v[start+2:]
                if alt not in variants:
                    variants.append(alt)
                    for pat, rep in suffix_pairs:
                        cand = re.sub(pat, rep, alt, flags=re.IGNORECASE)
                        if cand != alt and cand not in variants:
                            variants.append(cand)

    return variants[:8]


def query_local_places(
    raw_query: str,
    user_lat: float,
    user_lon: float,
    radius_km: float = 15.0,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieves candidate places from the local SQLite index.
    Combines exact normalized matching, compacted multi-word matching, phonetic skeleton
    matching, FTS5 prefix matching, and RTree spatial filtering with calibrated scoring.
    """
    from difflib import SequenceMatcher

    p = Path(db_path)
    if not p.exists() or p.stat().st_size == 0:
        return []

    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    norm_q = normalize_token(raw_query)
    if not norm_q:
        con.close()
        return []

    compact_q = norm_q.replace(" ", "")
    q_skel = _phonetic_skeleton(raw_query)
    tokens = norm_q.split()
    variants = _query_variants(raw_query)

    # Spatial bounding box for RTree
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * max(math.cos(math.radians(user_lat)), 0.01))
    min_lat, max_lat = user_lat - delta_lat, user_lat + delta_lat
    min_lon, max_lon = user_lon - delta_lon, user_lon + delta_lon

    candidates_map: dict[int, dict[str, Any]] = {}

    # Strategy 1: Exact normalized name and compacted name match within bbox (including query variants)
    for v in variants:
        nv = normalize_token(v)
        cv = nv.replace(" ", "")
        cur.execute("""
            SELECT p.*
            FROM places p
            JOIN places_rtree r ON p.id = r.id
            WHERE (p.name_normalized = ? OR p.name_compacted = ? OR p.name_normalized = ?)
              AND r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lon >= ? AND r.max_lon <= ?
            LIMIT ?
        """, (nv, cv, cv, min_lat, max_lat, min_lon, max_lon, limit * 2))
        for r in cur.fetchall():
            candidates_map[r["id"]] = dict(r)

    # Strategy 2: Exact phonetic skeleton match within bbox
    if q_skel:
        cur.execute("""
            SELECT p.*
            FROM places p
            JOIN places_rtree r ON p.id = r.id
            WHERE p.phonetic = ?
              AND r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lon >= ? AND r.max_lon <= ?
            LIMIT ?
        """, (q_skel, min_lat, max_lat, min_lon, max_lon, limit * 2))
        for r in cur.fetchall():
            if r["id"] not in candidates_map:
                candidates_map[r["id"]] = dict(r)

    # Strategy 3: FTS5 prefix & token match
    fts_queries = []
    if len(tokens) > 1:
        fts_queries.append(" ".join(f'"{t}"*' for t in tokens))
    if compact_q and len(compact_q) >= 4:
        fts_queries.append(f'"{compact_q}"*')
    for t in tokens:
        if len(t) >= 4:
            fts_queries.append(f'"{t}"*')

    for fts_q in fts_queries:
        try:
            cur.execute("""
                SELECT p.*
                FROM places p
                JOIN places_fts fts ON p.id = fts.rowid
                JOIN places_rtree r ON p.id = r.id
                WHERE places_fts MATCH ?
                  AND r.min_lat >= ? AND r.max_lat <= ?
                  AND r.min_lon >= ? AND r.max_lon <= ?
                LIMIT 50
            """, (fts_q, min_lat, max_lat, min_lon, max_lon))
            for r in cur.fetchall():
                if r["id"] not in candidates_map:
                    candidates_map[r["id"]] = dict(r)
        except sqlite3.OperationalError as e:
            logger.debug(f"FTS5 query notice: {e}")

    # Strategy 4: Category discovery (hospital, cafe, restaurant, park, etc.)
    common_cats = (
        "hospital", "cafe", "restaurant", "pharmacy", "bank", "atm",
        "bakery", "hotel", "supermarket", "park", "bar", "cinema", "mall",
        "clinic", "dhaba", "pub", "gym", "store", "shop"
    )
    matched_cat = next((c for c in common_cats if re.search(r"\b" + re.escape(c) + r"\b", norm_q)), None)
    if matched_cat and len(candidates_map) < limit * 2:
        cur.execute("""
            SELECT p.*
            FROM places p
            JOIN places_rtree r ON p.id = r.id
            WHERE (p.category LIKE ? OR p.feature_class = ?)
              AND r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lon >= ? AND r.max_lon <= ?
            LIMIT ?
        """, (f"%{matched_cat}%", matched_cat, min_lat, max_lat, min_lon, max_lon, limit * 2))
        for r in cur.fetchall():
            if r["id"] not in candidates_map:
                candidates_map[r["id"]] = dict(r)

    # Strategy 5: Fast fuzzy scan over bbox places if candidate pool is small or lacks a strong match
    has_strong_match = any(
        (c.get("name_compacted") == compact_q or SequenceMatcher(None, compact_q, c.get("name_compacted") or "").ratio() >= 0.85)
        for c in candidates_map.values()
    )
    if len(candidates_map) < 5 or not has_strong_match:
        cur.execute("""
            SELECT p.*
            FROM places p
            JOIN places_rtree r ON p.id = r.id
            WHERE r.min_lat >= ? AND r.max_lat <= ?
              AND r.min_lon >= ? AND r.max_lon <= ?
        """, (min_lat, max_lat, min_lon, max_lon))
        l_q = len(compact_q)
        for r in cur.fetchall():
            c_comp = r["name_compacted"] or r["name_normalized"].replace(" ", "")
            # Avoid comparing whole compact string for multi-token queries if no token overlaps
            if len(tokens) >= 2:
                c_tokens = r["name_normalized"].split()
                if not any(
                    (len(qt) >= 3 and any(ct.startswith(qt[:3]) or qt.startswith(ct[:3]) for ct in c_tokens))
                    for qt in tokens
                ):
                    continue
            if abs(len(c_comp) - l_q) <= 4 or c_comp.startswith(compact_q[:3]) or compact_q.startswith(c_comp[:3]):
                sim = SequenceMatcher(None, compact_q, c_comp).ratio()
                if sim >= 0.65 and r["id"] not in candidates_map:
                    candidates_map[r["id"]] = dict(r)

    con.close()

    # Scoring and ranking
    out: list[dict[str, Any]] = []
    for c in candidates_map.values():
        dist = calculate_distance(user_lat, user_lon, c["lat"], c["lon"])
        if dist > radius_km:
            continue

        c_name = c["name"]
        c_norm = c["name_normalized"]
        c_comp = c.get("name_compacted") or c_norm.replace(" ", "")
        c_short = c.get("short_address") or ""
        c_cat = c.get("category") or ""
        feat = c.get("feature_class") or ""

        # Compute name similarity
        if norm_q == c_norm or compact_q == c_comp:
            name_sim = 1.0
        else:
            s_char = SequenceMatcher(None, compact_q, c_comp).ratio()
            s_phon = 1.0 if (q_skel and q_skel == c.get("phonetic")) else 0.0
            s_cont = 1.0 if (compact_q in c_comp or c_comp in compact_q) and min(len(compact_q), len(c_comp)) >= 4 else 0.0
            name_sim = max(s_char, s_phon * 0.90, s_cont * 0.88)
            # For multi-word queries, penalize if substantive token coverage is low
            if len(tokens) >= 2:
                c_toks = c_norm.split()
                matched_t = sum(1 for t in tokens if any(ct == t or (len(t) >= 4 and ct.startswith(t[:4])) for ct in c_toks))
                cov = matched_t / len(tokens)
                if cov < 0.50:
                    name_sim = name_sim * ((cov / 0.50) ** 2)

        # Proximity score
        prox_score = max(0.0, 1.0 - (dist / 30.0))

        # Locality match in short address
        locality_sim = 1.0 if any(tok in c_short.lower() for tok in tokens if len(tok) >= 3) else 0.0

        # Calibration
        if name_sim >= 0.98:
            total_score = round(0.85 * name_sim + 0.15 * prox_score, 4)
        elif name_sim < 0.35 and not matched_cat:
            total_score = name_sim * 0.5
        elif matched_cat:
            total_score = round(0.50 + 0.35 * prox_score + 0.15 * name_sim, 4)
        else:
            total_score = round(0.70 * name_sim + 0.15 * locality_sim + 0.15 * prox_score, 4)

        # Importance boost for localities and transport nodes
        importance = 0.85 if feat in ("locality", "transport") else 0.50
        if feat in ("locality", "transport"):
            total_score = min(1.0, total_score + 0.02)

        if total_score >= 0.35 or matched_cat:
            out.append({
                "id": c["id"],
                "osm_id": c["osm_id"],
                "osm_type": c["osm_type"],
                "name": c_name,
                "name_alt": c.get("name_alt", ""),
                "full_address": f"{c_name}, {c_short}" if c_short else c_name,
                "short_address": c_short or DEFAULT_CITY_NAME,
                "lat": c["lat"],
                "lon": c["lon"],
                "distance_km": round(dist, 2),
                "category": c_cat,
                "feature_class": feat,
                "rating": 4.5,
                "importance": importance,
                "total_score": total_score,
                "name_sim": round(name_sim, 3),
            })

    out.sort(key=lambda x: (x["total_score"], -x["distance_km"]), reverse=True)
    return out[:limit]
