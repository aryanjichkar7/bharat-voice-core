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
import unittest
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
        with open("spatial_engine.py", "r", encoding="utf-8") as f:
            spatial_src = f.read()
        with open("app.py", "r", encoding="utf-8") as f:
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
        with open("static/app.js", "r", encoding="utf-8") as f:
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


if __name__ == "__main__":
    unittest.main()
