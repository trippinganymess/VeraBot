"""Tests for the MuRIL-based semantic matcher.

These tests exercise the SemanticMatcher class against a curated set of
messages covering English, Hindi, Hinglish, and edge-case inputs.

Set NO_LLM=1 to skip model-loading tests (fast CI mode).
Without NO_LLM, the full model is downloaded and loaded (~950 MB on first run).
"""

import os
import unittest

from semantic_matcher import (
    AUTO_REPLY_ANCHORS,
    AUTO_REPLY_THRESHOLD,
    INTENT_TRANSITION_ANCHORS,
    INTENT_TRANSITION_THRESHOLD,
    SemanticMatcher,
)


class TestAnchorCoverage(unittest.TestCase):
    """Ensure anchor lists are well-formed and non-empty."""

    def test_auto_reply_anchors_non_empty(self):
        self.assertGreater(len(AUTO_REPLY_ANCHORS), 5)

    def test_intent_anchors_non_empty(self):
        self.assertGreater(len(INTENT_TRANSITION_ANCHORS), 5)

    def test_no_duplicate_auto_anchors(self):
        self.assertEqual(len(AUTO_REPLY_ANCHORS), len(set(AUTO_REPLY_ANCHORS)))

    def test_no_duplicate_intent_anchors(self):
        self.assertEqual(
            len(INTENT_TRANSITION_ANCHORS), len(set(INTENT_TRANSITION_ANCHORS))
        )

    def test_thresholds_are_reasonable(self):
        self.assertGreater(AUTO_REPLY_THRESHOLD, 0.3)
        self.assertLess(AUTO_REPLY_THRESHOLD, 0.95)
        self.assertGreater(INTENT_TRANSITION_THRESHOLD, 0.3)
        self.assertLess(INTENT_TRANSITION_THRESHOLD, 0.95)


class TestSemanticMatcherNoModel(unittest.TestCase):
    """Tests that run without loading the model (NO_LLM=1 compatible)."""

    def test_returns_false_when_model_unavailable(self):
        """If the model is not loaded, fallback is False — never crashes."""
        matcher = SemanticMatcher()
        # Force model to stay unloaded
        original = os.environ.get("NO_LLM")
        os.environ["NO_LLM"] = "1"
        try:
            self.assertFalse(matcher.is_auto_reply("Thank you for contacting us"))
            self.assertFalse(matcher.is_intent_transition("lets do it"))
            auto_s, intent_s = matcher.classify("hello")
            self.assertEqual(auto_s, 0.0)
            self.assertEqual(intent_s, 0.0)
        finally:
            if original is None:
                os.environ.pop("NO_LLM", None)
            else:
                os.environ["NO_LLM"] = original


@unittest.skipIf(
    os.getenv("NO_LLM") == "1",
    "Skipping model-dependent tests (NO_LLM=1)",
)
class TestSemanticMatcherWithModel(unittest.TestCase):
    """Tests that require the MuRIL model to be loaded."""

    @classmethod
    def setUpClass(cls):
        cls.matcher = SemanticMatcher()
        cls.matcher._ensure_loaded()

    # -- auto-reply detection -----------------------------------------------

    def test_english_auto_reply_detected(self):
        self.assertTrue(
            self.matcher.is_auto_reply(
                "Thank you for reaching out to our dental clinic! "
                "We will respond shortly."
            )
        )

    def test_hindi_auto_reply_detected(self):
        self.assertTrue(
            self.matcher.is_auto_reply("hum jald hi sampark karenge")
        )

    def test_formal_hindi_auto_reply_detected(self):
        self.assertTrue(
            self.matcher.is_auto_reply(
                "aapka sandesh mil gaya hai, kripya pratiksha karein"
            )
        )

    def test_office_closed_auto_reply(self):
        self.assertTrue(
            self.matcher.is_auto_reply(
                "Our office is closed right now. We'll get back to you tomorrow."
            )
        )

    # -- intent detection ---------------------------------------------------

    def test_english_intent_detected(self):
        self.assertTrue(
            self.matcher.is_intent_transition("Ok lets do it. Whats next?")
        )

    def test_hindi_intent_detected(self):
        self.assertTrue(
            self.matcher.is_intent_transition("shuru karo bhai")
        )

    def test_join_intent_detected(self):
        self.assertTrue(
            self.matcher.is_intent_transition("I want to join this program")
        )

    def test_hinglish_register_intent(self):
        self.assertTrue(
            self.matcher.is_intent_transition("haan bilkul, mujhe register karo")
        )

    # -- neither category ---------------------------------------------------

    def test_hostile_is_neither(self):
        auto_s, intent_s = self.matcher.classify(
            "Stop messaging me. This is useless spam."
        )
        self.assertLess(auto_s, AUTO_REPLY_THRESHOLD)
        self.assertLess(intent_s, INTENT_TRANSITION_THRESHOLD)

    def test_weather_is_neither(self):
        auto_s, intent_s = self.matcher.classify(
            "what is the weather like today"
        )
        self.assertLess(auto_s, AUTO_REPLY_THRESHOLD)
        self.assertLess(intent_s, INTENT_TRANSITION_THRESHOLD)

    def test_gst_question_is_neither(self):
        auto_s, intent_s = self.matcher.classify(
            "Btw can you also help me with my GST filing this month?"
        )
        self.assertLess(auto_s, AUTO_REPLY_THRESHOLD)
        self.assertLess(intent_s, INTENT_TRANSITION_THRESHOLD)

    # -- discrimination: auto > intent and vice versa -----------------------

    def test_auto_reply_scores_higher_than_intent(self):
        auto_s, intent_s = self.matcher.classify(
            "Thank you for contacting us. Our team will respond shortly."
        )
        self.assertGreater(auto_s, intent_s)

    def test_intent_scores_higher_than_auto(self):
        auto_s, intent_s = self.matcher.classify(
            "Ok lets do it. Whats next?"
        )
        self.assertGreater(intent_s, auto_s)


if __name__ == "__main__":
    unittest.main()
