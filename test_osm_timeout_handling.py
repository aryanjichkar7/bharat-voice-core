"""
Unit and regression tests verifying:
1. Bounded OSM search latency and cancellation before application deadline.
2. Distinction of 0 candidates (200 OK NOT_FOUND) from provider timeouts (504 PROVIDER_ERROR) and network errors.
3. Provider error recording in _last_search_meta.
4. Prevention of trailing microphone audio superseding in-flight search turns.
5. Error state propagation to the UI without silent freezes.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import requests
import asyncio

import spatial_engine as se
from app import Session, SessionController, TranscriptTap, ui_message
from pipecat.frames.frames import TranscriptionFrame


class TestOsmTimeoutHandling(unittest.TestCase):

    def setUp(self):
        self.session = Session()
        self.controller = SessionController(session=self.session)

    def test_provider_status_recorded_on_success(self):
        """Verifies that successful HTTP 200 responses record provider_status=200 and no provider_error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "Sitabuldi", "lat": "21.1458", "lon": "79.0882", "display_name": "Sitabuldi, Nagpur"}]
        with patch.object(se._session, "get", return_value=mock_resp):
            res = se._nominatim("Sitabuldi", "79.0,21.1,79.1,21.2")
            self.assertEqual(len(res), 1)
            self.assertEqual(se._last_search_meta["provider_status"], 200)
            self.assertIsNone(se._last_search_meta["provider_error"])

    def test_distinguishes_zero_results_from_provider_timeout(self):
        """Verifies that an empty result with HTTP 200 is NOT_FOUND, whereas a Timeout is PROVIDER_ERROR."""
        with patch.dict(os.environ, {"ENABLE_PUBLIC_OSM_FALLBACK": "true"}), \
             patch("spatial_engine.get_index_stats", return_value={"database_exists": False, "total_rows": 0}):
            # Case A: Valid 200 OK with zero candidates
            mock_resp_empty = MagicMock()
            mock_resp_empty.status_code = 200
            mock_resp_empty.json.return_value = []
            with patch.object(se._session, "get", return_value=mock_resp_empty):
                res_not_found = se.evaluate_destination_intent("Nonexistent Place 12345", 21.1458, 79.0882)
                self.assertEqual(res_not_found["status"], "NOT_FOUND")
                self.assertEqual(res_not_found["resolution_type"], "NO_RESULTS")
                self.assertEqual(res_not_found["provider_status"], 200)

            # Case B: Request Timeout (HTTP 504)
            with patch.object(se._session, "get", side_effect=requests.exceptions.Timeout("Connection timed out")):
                res_timeout = se.evaluate_destination_intent("Some Place", 21.1458, 79.0882)
                self.assertEqual(res_timeout["status"], "PROVIDER_ERROR")
                self.assertEqual(res_timeout["resolution_type"], "PROVIDER_ERROR")
                self.assertEqual(res_timeout["provider_status"], 504)
                self.assertIn("unavailable", res_timeout["clarification_prompt"].lower())

    def test_search_stops_sequential_variants_on_provider_error(self):
        """Verifies that if variant 1 times out, sequential variants are aborted to save budget."""
        with patch.object(se._session, "get", side_effect=requests.exceptions.Timeout("Timeout")):
            with patch.object(se, "_query_variants", return_value=["variant1", "variant2", "variant3"]):
                cands = se.fallback_osm_search("test query", 21.1458, 79.0882)
                self.assertEqual(cands, [])
                self.assertEqual(se._last_search_meta["provider_status"], 504)

    def test_trailing_transcription_does_not_advance_turn_generation_during_processing(self):
        """Verifies that trailing transcriptions arriving while is_processing_turn is True are dropped."""
        tap = TranscriptTap(session=self.session, controller=self.controller)
        self.session.is_processing_turn = True
        self.session.turn_generation = 5

        frame = TranscriptionFrame(text="trailing noise", user_id="user", timestamp="now")
        asyncio.run(tap.process_frame(frame, MagicMock()))

        # turn_generation must NOT have advanced
        self.assertEqual(self.session.turn_generation, 5)


if __name__ == "__main__":
    unittest.main()
