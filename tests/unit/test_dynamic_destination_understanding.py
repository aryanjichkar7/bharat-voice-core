"""
Verification Suite: Dynamic Destination Understanding & Reliable Voice Turn Management
Bharat Online Cabs (bharat-voice-core)

Verifies:
1. Dynamic destination entity extraction across English, Hindi, and Marathi conversational phrases
2. Zero hardcoded destination/venue lists in spatial_engine.py and app.py
3. Accurate POI resolution for conversational queries without quoting carrier sentences in prompts
4. Correct radius_km parameter passing in spatial_engine.py (resolving 15m radius bug)
5. VAD timing parameters (stop_secs = 0.9s for natural pauses)
6. Frontend voice lifecycle integrity (no ws.close on orb toggle, 8s speech-start timeout)
"""
import inspect
import json
import os
import re
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spatial_engine as se
import app

class TestDynamicDestinationUnderstanding(unittest.TestCase):
    def test_multilingual_entity_extraction(self):
        cases = [
            # English conversational phrasing
            ("Take me to Maharaj Bagh Zoo", "Maharaj Bagh Zoo"),
            ("Can you please take me to Sitabuldi", "Sitabuldi"),
            ("Actually take me to Sadar", "Sadar"),
            ("Take me from here to Airport", "Airport"),
            ("I would like to book a cab to Airport", "Airport"),
            ("Drive me to Dharampeth", "Dharampeth"),
            ("Drop me off at Civil Lines", "Civil Lines"),
            ("Can you book a ride to Kasturchand Park", "Kasturchand Park"),
            ("I want to go to Nagpur Railway Station please", "Nagpur Railway Station"),
            ("Let's go to Futala Lake", "Futala Lake"),
            ("How much to go to IT Park", "IT Park"),
            ("Cab to Medical Square", "Medical Square"),
            
            # Hindi conversational phrasing (Latin & Devanagari)
            ("Mujhe Sitabuldi jana hai", "Sitabuldi"),
            ("Maharaj Bagh le chalo", "Maharaj Bagh"),
            ("Airport pahuncha do", "Airport"),
            ("हमे सीताबर्डी जाना है", "सीताबर्डी"),
            ("सदर ले चलो", "सदर"),
            
            # Marathi conversational phrasing (Latin & Devanagari)
            ("Mala Maharaj Bagh la jaycha ahe", "Maharaj Bagh"),
            ("Sitabuldila jaycha ahe", "Sitabuldi"),
            ("मला सीताबर्डीला जायचे आहे", "सीताबर्डी"),
            ("सिव्हिल लाईन्सला जायचे आहे", "सिव्हिल लाईन्स"),
            
            # Direct place names (passthrough)
            ("Chitnavis Centre", "Chitnavis Centre"),
            ("Betwyn By Checkers", "Betwyn By Checkers"),
        ]
        
        for input_text, expected_target in cases:
            extracted = se.extract_destination_entity(input_text)
            self.assertEqual(
                extracted.strip().lower(),
                expected_target.strip().lower(),
                f"Failed extraction for '{input_text}': got '{extracted}', expected '{expected_target}'"
            )

    def test_no_hardcoded_venues_in_codebase(self):
        """Strict check ensuring no hardcoded venue lists exist in application logic."""
        with open(REPO_ROOT / "spatial_engine.py", "r", encoding="utf-8") as f:
            spatial_src = f.read()
        with open(REPO_ROOT / "app.py", "r", encoding="utf-8") as f:
            app_src = f.read()

        # Check NAGPUR_LOCALITIES is completely gone from spatial_engine
        self.assertNotIn("NAGPUR_LOCALITIES", spatial_src, "NAGPUR_LOCALITIES must not exist in spatial_engine.py")

        # Check STT_HOTWORDS has no hardcoded venue list in app.py
        banned_venues = ["sitabuldi", "dharampeth", "haldiram", "dmart", "batukbhai", "londe jewellers"]
        for line in app_src.splitlines():
            if line.startswith("STT_HOTWORDS ="):
                for venue in banned_venues:
                    self.assertNotIn(venue, line.lower(), f"Hardcoded venue '{venue}' found in STT_HOTWORDS")

    def test_conversational_poi_evaluation(self):
        """Conversational query must extract the entity and evaluate successfully."""
        query = "Can you please take me to Sitabuldi"
        res = se.evaluate_destination_intent(query, 21.1458, 79.0882, radius_km=15.0)
        
        # Must not report 'Can you please take me to Sitabuldi' in clarification prompt
        self.assertNotIn(
            "Can you please take me to Sitabuldi",
            res.get("clarification_prompt", ""),
            "Full sentence carrier was erroneously used in clarification prompt!"
        )
        self.assertIn(res["status"], ("EXACT", "DISAMBIGUATION_NEEDED"))
        self.assertIsNotNone(res["primary_match"])

    def test_disambiguation_radius_parameter_fix(self):
        """Verifies that search_nearby_pois receives radius_km properly and finds multiple outlets."""
        res = se.evaluate_destination_intent("DMart", 21.1458, 79.0882, radius_km=15.0)
        self.assertEqual(res["status"], "DISAMBIGUATION_NEEDED")
        self.assertGreaterEqual(res["outlet_count"], 2, "Expected >= 2 DMart outlets across Nagpur")

    def test_vad_and_timing_config(self):
        """Verifies VAD stop_secs is in the natural speech range (0.8s - 1.2s)."""
        self.assertGreaterEqual(app.VAD_STOP_SECS, 0.8)
        self.assertLessEqual(app.VAD_STOP_SECS, 1.2)
        self.assertEqual(app.VAD_STOP_SECS, 0.9)

    def test_frontend_lifecycle_contract(self):
        """Verifies frontend does not kill the WebSocket on recording toggle."""
        with open(REPO_ROOT / "static/app.js", "r", encoding="utf-8") as f:
            js_src = f.read()

        # Check voiceOrb click handler does not close WebSocket on toggle
        orb_click_idx = js_src.find("voiceOrb.addEventListener('click'")
        self.assertGreater(orb_click_idx, 0, "voiceOrb click listener not found")
        orb_block = js_src[orb_click_idx:orb_click_idx + 600]
        self.assertNotIn("ws.close()", orb_block, "ws.close() found in voiceOrb click handler!")

        # Check speechStartTimeout exists
        self.assertIn("speechStartTimeout", js_src, "speechStartTimeout missing from static/app.js")
        self.assertIn("scheduleSpeechStartTimeout", js_src, "scheduleSpeechStartTimeout missing from static/app.js")

    def test_unknown_destination_prompt_quotes_clean_entity(self):
        """Unknown destination prompt must quote the extracted entity, not the conversational carrier."""
        query = "Take me to Atlantis Floating Castle"
        res = se.evaluate_destination_intent(query, 21.1458, 79.0882, radius_km=15.0)
        self.assertEqual(res["status"], "NOT_FOUND")
        prompt = res.get("clarification_prompt", "")
        self.assertIn("Atlantis Floating Castle", prompt)
        self.assertNotIn("Take me to Atlantis Floating Castle", prompt)

    # -------------------------------------------------------------------------
    # Stage 2J: Flexible Natural Intent & Sentence Structure Understanding
    # -------------------------------------------------------------------------
    def test_stage2j_freeform_sentence_structures(self):
        """Verifies destination extraction across freeform sentence structures, corrections, and qualifiers."""
        cases = [
            ("I changed my mind, take me to Central Mall", "Central Mall"),
            ("No, not that, I want to go to City Hospital", "City Hospital"),
            ("Can we change destination to Airport", "Airport"),
            ("Actually, let's head to Railway Station", "Railway Station"),
            ("Could you drop us off near the university", "university"),
            ("Please arrange a car to the High Court by 5pm", "High Court"),
            ("How much to go to Botanical Gardens", "Botanical Gardens"),
            ("We are heading towards IT Park", "IT Park"),
            ("Airport", "Airport"),
            ("Central Mall please", "Central Mall"),
            ("Drop me at Sector 4", "Sector 4"),
        ]
        for inp, expected in cases:
            extracted = se.extract_destination_entity(inp)
            self.assertEqual(
                extracted.strip().lower(),
                expected.strip().lower(),
                f"Failed freeform extraction for '{inp}': got '{extracted}', expected '{expected}'"
            )

    def test_stage2j_natural_intent_classification(self):
        """Verifies structured intent categorization for cancellations, confirmations, general queries, and booking."""
        p1 = se.parse_natural_intent_and_destination("What is the weather today")
        self.assertEqual(p1["intent"], "GENERAL_QUERY")

        p2 = se.parse_natural_intent_and_destination("Who are you")
        self.assertEqual(p2["intent"], "GENERAL_QUERY")

        p3 = se.parse_natural_intent_and_destination("Cancel the booking please")
        self.assertEqual(p3["intent"], "CANCELLATION")

        p4 = se.parse_natural_intent_and_destination("Yes, that's right, book it")
        self.assertEqual(p4["intent"], "CONFIRMATION")

        p5 = se.parse_natural_intent_and_destination("Please book a cab to Airport")
        self.assertEqual(p5["intent"], "BOOK_CAB")
        self.assertEqual(p5["destination"], "Airport")

        p6 = se.parse_natural_intent_and_destination("Railway Station")
        self.assertEqual(p6["intent"], "DESTINATION_SEARCH")
        self.assertEqual(p6["destination"], "Railway Station")

    def test_stage2j_general_query_bypasses_spatial_search(self):
        """Verifies that non-booking general queries do not trigger map geocoding searches."""
        self.assertTrue(se.is_general_query("What is the weather today?"))
        self.assertTrue(se.is_general_query("Who made you?"))
        self.assertTrue(se.is_general_query("How are you doing?"))
        self.assertFalse(se.is_general_query("How much to go to Airport?"))
        self.assertFalse(se.is_general_query("Take me to Airport"))
        self.assertFalse(se.is_general_query("Airport"))

    # -------------------------------------------------------------------------
    # Stage 2L: Context-Aware Booking & Natural Intent Understanding
    # -------------------------------------------------------------------------
    def test_stage2l_natural_affirmations_and_confirmations(self):
        """Verifies that natural affirmative spoken phrases are recognized as confirmations, while negations and destination prepositions are rejected."""
        valid_confirmations = [
            "ok", "okay", "alright", "sounds good", "looks good", "perfect",
            "fine", "great", "that works", "works for me", "let's do it", "lets do it",
            "let's go", "chalel", "kar do", "bilkul", "zaroor", "yes", "confirm",
            "book it", "please book", "theek hai", "sahi hai", "all right"
        ]
        for phrase in valid_confirmations:
            self.assertTrue(
                app.is_confirmation_intent(phrase),
                f"Expected '{phrase}' to be recognized as affirmative confirmation"
            )

        rejected = [
            "no", "don't book", "dont book", "nahi", "not now", "wait a minute",
            "not okay", "not good", "cancel", "never mind",
            # Utterances carrying a destination target
            "ok, but take me to airport", "sounds good, go to sadar instead"
        ]
        for phrase in rejected:
            self.assertFalse(
                app.is_confirmation_intent(phrase),
                f"Expected '{phrase}' to be rejected as confirmation"
            )

    def test_stage2l_conversational_statements_not_destinations(self):
        """Verifies that conversational remarks, gratitude, pauses, and inquiries are recognized as general queries and not bare destinations."""
        conversational = [
            "Thank you", "Thanks a lot", "Dhanyawad",
            "Wait a minute", "Hold on", "Just a second", "Thoda ruko",
            "Can you hear me", "Are you listening",
            "What can you do", "Who are you",
            "Nothing", "Never mind", "I don't know"
        ]
        for phrase in conversational:
            self.assertTrue(
                se.is_general_query(phrase),
                f"Expected '{phrase}' to be recognized as general query"
            )
            parsed = se.parse_natural_intent_and_destination(phrase)
            self.assertIn(
                parsed["intent"],
                ("GENERAL_QUERY", "CANCELLATION", "UNKNOWN"),
                f"Expected non-destination intent for '{phrase}', got {parsed['intent']}"
            )

    def test_stage2l_generic_toponym_descriptor_gating(self):
        """Verifies that candidates matching only generic descriptors with zero substantive token overlap are rejected as ungrounded."""
        # Candidate is 'City Point' (distance 1.2km)
        candidate = {
            "name": "City Point",
            "short_address": "Commercial Hub",
            "lat": 21.146,
            "lon": 79.086,
            "distance_km": 1.2,
        }
        # Query shares only generic descriptor 'point' with zero substantive overlap
        score_info = se.score_candidate("West Point", candidate, 21.1458, 79.0882)
        self.assertFalse(
            score_info["is_plausible"],
            "Candidate sharing only generic toponym descriptor 'point' must be gated as unplausible"
        )

    def test_stage2l_numbered_disambiguation_formatting(self):
        """Verifies that multiple close candidates are formatted with numbered options, locality, and distances."""
        cand1 = {"name": "Central Point", "short_address": "Sadar", "distance_km": 1.5, "rating": 4.5}
        cand2 = {"name": "Central Point", "short_address": "Wardha Road", "distance_km": 2.2, "rating": 4.2}
        with patch("spatial_engine.search_nearby_pois", return_value=[cand1, cand2]):
            res = se.evaluate_destination_intent("Central Point", 21.1458, 79.0882)
            self.assertEqual(res["status"], "DISAMBIGUATION_NEEDED")
            prompt = res.get("clarification_prompt", "")
            self.assertIn("1. Central Point", prompt)
            self.assertIn("2. Central Point", prompt)
            self.assertIn("km", prompt)


if __name__ == "__main__":
    unittest.main()

