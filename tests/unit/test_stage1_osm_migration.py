"""
test_stage1_osm_migration.py

Automated Verification Suite for Bharat Online Cabs — Stage 1: OpenStreetMap Migration
Tests:
1. Google Dependency Audit (Static & Runtime inspection: no Google Maps JS SDK or Google Places in active path)
2. Local Leaflet Vendor Assets & Attribution Verification
3. Tariff Calculation Formula Adherence (Base ₹50, ₹14/km, Min ₹70)
4. FastAPI /api/config Endpoint Audit (Zero Google keys, valid OSM parameters)
5. Dynamic OSM Category Discovery (Live OSM queries without hardcoded venue lists)
6. Dynamic OSM Named Place Search (Live OSM queries without hardcoded venue lists)
7. Reverse Geocoding of GPS Coordinates via OSM
8. OSRM Road Route Geometry, Driving Distance, Duration, and Fare Calculation
9. Honest Error Handling on Routing Failure (No straight-line fabrication)
10. Strict Rule Verification: No hardcoded venue dictionaries in application code
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spatial_engine as se
from app import app


class TestStage1OsmMigration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_no_google_dependencies_in_active_code(self):
        """Ensure active application code contains zero calls to Google Places API or Google Maps JS SDK."""
        app_code = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
        spatial_code = (REPO_ROOT / "spatial_engine.py").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "static/app.js").read_text(encoding="utf-8")
        index_html = (REPO_ROOT / "static/index.html").read_text(encoding="utf-8")

        # 1. No Google Maps JS script tag in HTML
        self.assertNotIn("maps.googleapis.com", index_html, "Google Maps JavaScript SDK script tag found in index.html")

        # 2. No places.googleapis.com call in spatial_engine.py
        self.assertNotIn("places.googleapis.com", spatial_code, "Google Places API endpoint found in spatial_engine.py")

        # 3. No google.maps object usage in active JS logic
        self.assertNotIn("new google.maps.Map", app_js, "Google Maps Map instantiation found in app.js")
        self.assertNotIn("new google.maps.Marker", app_js, "Google Maps Marker instantiation found in app.js")
        self.assertNotIn("new google.maps.Polyline", app_js, "Google Maps Polyline instantiation found in app.js")

        # 4. /api/config does not return google_maps_browser_key
        self.assertNotIn("google_maps_browser_key", app_code, "google_maps_browser_key still returned in app.py")

    def test_02_local_leaflet_vendor_assets_and_attribution(self):
        """Verify Leaflet vendor assets are stored locally and OSM attribution is rendered."""
        css_path = REPO_ROOT / "static/vendor/leaflet.css"
        js_path = REPO_ROOT / "static/vendor/leaflet.js"
        index_html = (REPO_ROOT / "static/index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "static/app.js").read_text(encoding="utf-8")

        self.assertTrue(css_path.exists() and css_path.stat().st_size > 5000, "Local Leaflet CSS asset missing or invalid")
        self.assertTrue(js_path.exists() and js_path.stat().st_size > 50000, "Local Leaflet JS asset missing or invalid")

        self.assertIn('<link rel="stylesheet" href="/static/vendor/leaflet.css">', index_html)
        self.assertIn('<script src="/static/vendor/leaflet.js"></script>', index_html)
        self.assertIn('id="map"', index_html)

        # Verify OpenStreetMap attribution in app.js
        self.assertIn("OpenStreetMap", app_js, "OpenStreetMap attribution missing from app.js")

    def test_03_tariff_calculation_formula(self):
        """Validate Bharat Online Cabs tariff rules: ₹50 base, ₹14/km, minimum ₹70."""
        self.assertEqual(se.BASE_FARE_INR, 50.0)
        self.assertEqual(se.PER_KM_INR, 14.0)
        self.assertEqual(se.MIN_FARE_INR, 70.0)

        # Short trip under minimum fare (e.g. 1.0 km -> 50 + 14 = 64 < 70)
        self.assertEqual(se.calculate_fare(1.0), 70.0)

        # Zero distance -> minimum fare ₹70
        self.assertEqual(se.calculate_fare(0.0), 70.0)

        # Trip where formula exceeds minimum (e.g. 5.0 km -> 50 + 5*14 = 120.0)
        self.assertEqual(se.calculate_fare(5.0), 120.0)

        # Fractional distance (e.g. 3.5 km -> 50 + 3.5*14 = 99.0)
        self.assertEqual(se.calculate_fare(3.5), 99.0)

    def test_04_api_config_endpoint(self):
        """Verify /api/config returns OSM configuration without exposing any Google secrets."""
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertNotIn("google_maps_browser_key", data, "Google key leaked in /api/config")
        self.assertEqual(data.get("map_provider"), "osm")
        self.assertIn("openstreetmap.org", data.get("map_tile_url", ""))
        self.assertIn("osrm", data.get("osm_routing_url", "").lower())
        self.assertIn("nominatim", data.get("osm_geocoder_url", "").lower())
        self.assertEqual(data.get("default_lat"), 21.1458)
        self.assertEqual(data.get("default_lon"), 79.0882)

    def test_05_dynamic_osm_category_discovery(self):
        """Verify dynamic business category discovery returns real candidates via OSM without hardcoding."""
        results = se.search_nearby_pois("hospital", se.DEFAULT_LAT, se.DEFAULT_LON, radius_km=10.0)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "OSM category search returned zero results for generic category")
        for r in results:
            self.assertIn("name", r)
            self.assertTrue(len(r["name"]) > 0)
            self.assertIn("lat", r)
            self.assertIn("lon", r)
            self.assertIn("distance_km", r)
            self.assertLessEqual(r["distance_km"], 15.0)

    def test_06_dynamic_osm_named_place_search(self):
        """Verify dynamic named place resolution via OSM without hardcoded venue lists."""
        results = se.search_nearby_pois("Sitabuldi", se.DEFAULT_LAT, se.DEFAULT_LON, radius_km=10.0)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "OSM search returned zero results for Sitabuldi")
        first = results[0]
        self.assertTrue("sitabuldi" in first["name"].lower() or "sitabuldi" in first.get("full_address", "").lower())

    def test_07_reverse_geocoding_via_osm(self):
        """Verify reverse-geocoding of GPS coordinate returns structured address."""
        rev = se.reverse_geocode_coordinate(se.DEFAULT_LAT, se.DEFAULT_LON)
        self.assertIsInstance(rev, dict)
        self.assertIn("display_name", rev)
        self.assertIn("lat", rev)
        self.assertIn("lon", rev)
        self.assertAlmostEqual(rev["lat"], se.DEFAULT_LAT, delta=0.01)

    def test_08_osrm_road_routing_and_fare(self):
        """Verify OSRM returns actual road route geometry, driving distance, duration, and fare."""
        dest_lat = se.DEFAULT_LAT + 0.012
        dest_lon = se.DEFAULT_LON + 0.012
        trip = se.calculate_trip_details(se.DEFAULT_LAT, se.DEFAULT_LON, dest_lat, dest_lon, "Dynamic Landmark")

        self.assertEqual(trip.get("status"), "OK")
        self.assertEqual(trip.get("route_source"), "osrm")
        self.assertIsNotNone(trip.get("distance_km"))
        self.assertGreater(trip["distance_km"], 0)
        self.assertIsNotNone(trip.get("duration_mins"))
        self.assertGreaterEqual(trip["duration_mins"], 1)
        self.assertIsNotNone(trip.get("fare_inr"))
        self.assertGreaterEqual(trip["fare_inr"], se.MIN_FARE_INR)

        geom = trip.get("route_geometry")
        self.assertIsNotNone(geom)
        self.assertEqual(geom.get("type"), "LineString")
        self.assertGreater(len(geom.get("coordinates", [])), 5, "Expected multi-segment road route geometry")

    def test_09_honest_routing_error_handling(self):
        """Verify system returns honest error when routing is unavailable, never fabricating straight lines."""
        with patch.object(se._session, "get", side_effect=se.requests.RequestException("Connection refused")):
            trip = se.calculate_trip_details(se.DEFAULT_LAT, se.DEFAULT_LON, 21.16, 79.10, "Unreachable Destination")
            self.assertEqual(trip.get("status"), "ERROR")
            self.assertEqual(trip.get("error_code"), "ROUTING_UNAVAILABLE")
            self.assertIsNone(trip.get("distance_km"))
            self.assertIsNone(trip.get("fare_inr"))
            self.assertIsNone(trip.get("route_geometry"))
            self.assertEqual(trip.get("route_source"), "failed")

    def test_10_no_hardcoded_venues_in_code_or_tests(self):
        """Verify no hardcoded venue lists or Nagpur venue dictionaries were added."""
        spatial_code = (REPO_ROOT / "spatial_engine.py").read_text(encoding="utf-8")
        self.assertNotIn("NAGPUR_VENUES", spatial_code)
        self.assertNotIn("KNOWN_PLACES", spatial_code)
        self.assertNotIn("FALLBACK_VENUES", spatial_code)
        self.assertNotIn("VENUE_REGISTRY", spatial_code)


if __name__ == "__main__":
    unittest.main()
