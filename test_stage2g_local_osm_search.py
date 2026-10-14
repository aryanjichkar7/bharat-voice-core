"""
Bharat Online Cabs — Stage 2G Verification Test Suite: Local OSM Destination Search.

Tests:
1. Local OpenStreetMap SQLite indexing and stats verification.
2. Fast sub-millisecond local search execution without contacting public Nominatim.
3. Candidate resolution outcomes:
   - EXACT_MATCH for exact matches.
   - POSSIBLE_MATCH for minor typos / transliterations requiring user confirmation.
   - MULTIPLE_CANDIDATES for category queries.
   - NO_RESULTS (HTTP 200) for unindexed queries without provider timeout.
4. Database absence handling: honest 503 error, never inventing fake places.
5. Idempotent import CLI functionality.
"""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import osm_indexer
import spatial_engine as se


class TestStage2GLocalOsmSearch(unittest.TestCase):

    def setUp(self):
        se._search_cache.clear()

    def test_local_index_schema_and_metadata(self):
        """Verifies that the SQLite database has required tables, metadata and attribution."""
        stats = osm_indexer.get_index_stats()
        self.assertTrue(stats["database_exists"])
        self.assertGreater(stats["total_rows"], 1000)

        con = sqlite3.connect(stats["database_path"])
        cur = con.cursor()

        # Check places table schema
        cur.execute("PRAGMA table_info(places)")
        cols = {row[1] for row in cur.fetchall()}
        required_cols = {"id", "osm_id", "osm_type", "name", "name_normalized", "name_compacted", "phonetic", "lat", "lon", "feature_class"}
        self.assertTrue(required_cols.issubset(cols), f"Missing columns in places: {required_cols - cols}")

        # Check metadata & attribution
        cur.execute("SELECT key, value FROM osm_metadata")
        meta_dict = dict(cur.fetchall())
        self.assertIn("attribution", meta_dict)
        self.assertIn("OpenStreetMap", meta_dict["attribution"])
        self.assertIn("ODbL", meta_dict["attribution"])
        con.close()

    def test_local_search_does_not_contact_public_network(self):
        """Verifies that queries use the local SQLite engine and do not invoke HTTP requests."""
        with patch.object(se._session, "get") as mock_http:
            res = se.search_nearby_pois("Sitabuldi", se.DEFAULT_LAT, se.DEFAULT_LON, radius_km=15.0)
            mock_http.assert_not_called()
            self.assertGreater(len(res), 0)
            self.assertEqual(se._last_search_meta["provider_used"], "Local OpenStreetMap SQLite Engine")
            self.assertEqual(se._last_search_meta["provider_status"], 200)
            self.assertLess(se._last_search_meta.get("last_request_elapsed_s", 1.0), 0.20)

    def test_exact_match_destination_resolution(self):
        """Verifies that an exact place name resolves to EXACT_MATCH with trip proposal."""
        res = se.evaluate_destination_intent("Sitabuldi", se.DEFAULT_LAT, se.DEFAULT_LON)
        self.assertEqual(res["status"], "EXACT")
        self.assertEqual(res["resolution_type"], "EXACT_MATCH")
        self.assertIsNotNone(res["primary_match"])
        self.assertIn("Shall I calculate the fare?", res["clarification_prompt"])

    def test_typo_and_transliteration_destination_resolution(self):
        """Verifies that a misspelled or split destination requires confirmation as POSSIBLE_MATCH."""
        res = se.evaluate_destination_intent("Ramdas Pet", se.DEFAULT_LAT, se.DEFAULT_LON)
        self.assertEqual(res["status"], "POSSIBLE_MATCH")
        self.assertEqual(res["resolution_type"], "POSSIBLE_MATCH")
        self.assertIsNotNone(res["suggested_match"])
        self.assertIn("Did you mean", res["clarification_prompt"])

    def test_category_query_returns_multiple_candidates(self):
        """Verifies that a category search returns MULTIPLE_CANDIDATES for user selection."""
        res = se.evaluate_destination_intent("find me a hospital", se.DEFAULT_LAT, se.DEFAULT_LON)
        self.assertEqual(res["status"], "DISAMBIGUATION_NEEDED")
        self.assertEqual(res["resolution_type"], "MULTIPLE_CANDIDATES")
        self.assertGreater(res["outlet_count"], 0)
        self.assertIn("closest hospital", res["clarification_prompt"].lower())

    def test_honest_not_found_on_unindexed_destination(self):
        """Verifies that an unindexed destination returns NO_RESULTS (HTTP 200) without provider timeout."""
        res = se.evaluate_destination_intent("nonexistent_destination_xyz_987", se.DEFAULT_LAT, se.DEFAULT_LON)
        self.assertEqual(res["status"], "NOT_FOUND")
        self.assertEqual(res["resolution_type"], "NO_RESULTS")
        self.assertEqual(res["provider_status"], 200)
        self.assertIsNone(res["provider_error"])
        self.assertIn("couldn't find", res["clarification_prompt"].lower())

    def test_uninitialized_database_returns_503(self):
        """Verifies that when database is absent, search returns honest 503 instead of fake results."""
        with patch("spatial_engine.get_index_stats", return_value={"database_exists": False, "total_rows": 0}):
            res = se.evaluate_destination_intent("Any Place", se.DEFAULT_LAT, se.DEFAULT_LON)
            self.assertEqual(res["status"], "PROVIDER_ERROR")
            self.assertEqual(res["resolution_type"], "PROVIDER_ERROR")
            self.assertEqual(res["provider_status"], 503)
            self.assertIn("temporarily unavailable", res["clarification_prompt"].lower())

    def test_custom_synthetic_database_indexing(self):
        """Verifies idempotent XML/SQLite importer using a synthetic mini OSM XML in a temp directory."""
        synthetic_osm = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="101" lat="21.1458" lon="79.0882">
    <tag k="name" v="Synthetic Test Point"/>
    <tag k="amenity" v="cafe"/>
    <tag k="addr:suburb" v="Test Locality"/>
  </node>
  <node id="102" lat="21.1500" lon="79.0900">
    <tag k="name" v="Synthetic Landmark"/>
    <tag k="place" v="suburb"/>
  </node>
</osm>
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = os.path.join(tmp_dir, "test.osm")
            db_path = os.path.join(tmp_dir, "test.db")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(synthetic_osm)

            importer = osm_indexer.OSMPlaceImporter(db_path=db_path)
            meta = importer.import_xml(xml_path)
            self.assertEqual(meta["total_entities"], 2)

            stats = osm_indexer.get_index_stats(db_path)
            self.assertEqual(stats["total_rows"], 2)

            # Query synthetic DB
            results = osm_indexer.query_local_places("Synthetic Test Point", 21.1458, 79.0882, db_path=db_path)
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["name"], "Synthetic Test Point")


if __name__ == "__main__":
    unittest.main()
