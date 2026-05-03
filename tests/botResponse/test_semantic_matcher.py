"""Tests for the BGE-M3-based semantic matcher.

These tests exercise the SemanticMatcher class against a curated set of
messages covering English, Hindi, Hinglish, and edge-case inputs.

Set NO_LLM=1 to skip model-dependent tests (fast CI mode).
Without NO_LLM, the matcher will hit the Hugging Face Inference API.
"""

import os
import unittest

from semantic_matcher import (
    AUTO_REPLY_ANCHORS,
    AUTO_REPLY_THRESHOLD,
    HOSTILE_THRESHOLD,
    INTENT_TRANSITION_ANCHORS,
    INTENT_TRANSITION_THRESHOLD,
    WAIT_THRESHOLD,
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
        for threshold in (
            AUTO_REPLY_THRESHOLD,
            INTENT_TRANSITION_THRESHOLD,
            HOSTILE_THRESHOLD,
            WAIT_THRESHOLD,
        ):
            self.assertGreater(threshold, 0.3)
            self.assertLess(threshold, 0.95)


class TestSemanticMatcherNoModel(unittest.TestCase):
    """Tests that run without loading the model (NO_LLM=1 compatible)."""

    def test_returns_zero_scores_when_model_unavailable(self):
        """If the inference client is not loaded, classify() returns zeros."""
        matcher = SemanticMatcher()
        original = os.environ.get("NO_LLM")
        os.environ["NO_LLM"] = "1"
        try:
            # Strings that miss the regex fast-path so we exercise the
            # "model unavailable" branch rather than the regex hit.
            self.assertFalse(matcher.is_auto_reply("i am busy today"))
            self.assertFalse(
                matcher.is_intent_transition("i might consider proceeding later")
            )
            scores = matcher.classify("hello")
            self.assertIsInstance(scores, dict)
            self.assertEqual(scores["auto_reply"], 0.0)
            self.assertEqual(scores["intent"], 0.0)
            self.assertEqual(scores["hostile"], 0.0)
            self.assertEqual(scores["wait"], 0.0)
        finally:
            if original is None:
                os.environ.pop("NO_LLM", None)
            else:
                os.environ["NO_LLM"] = original

    def test_regex_fastpath_still_fires_under_no_llm(self):
        """Stage-1 regex must still classify obvious messages with NO_LLM=1."""
        matcher = SemanticMatcher()
        original = os.environ.get("NO_LLM")
        os.environ["NO_LLM"] = "1"
        try:
            self.assertTrue(matcher.is_intent_transition("Ok lets do it"))
            self.assertEqual(
                matcher.get_intent_type("Stop sending me these messages"),
                "hostile",
            )
            self.assertEqual(
                matcher.get_intent_type("not now, baad mein baat karte hain"),
                "wait",
            )
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
    """Tests that require the BGE-M3 model to be reachable via HF."""

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

    def test_weather_is_neither(self):
        scores = self.matcher.classify("what is the weather like today")
        self.assertLess(scores["auto_reply"], AUTO_REPLY_THRESHOLD)
        self.assertLess(scores["intent"], INTENT_TRANSITION_THRESHOLD)

    def test_gst_question_is_neither(self):
        scores = self.matcher.classify(
            "Btw can you also help me with my GST filing this month?"
        )
        self.assertLess(scores["auto_reply"], AUTO_REPLY_THRESHOLD)
        self.assertLess(scores["intent"], INTENT_TRANSITION_THRESHOLD)

    # -- discrimination: auto > intent and vice versa -----------------------

    def test_auto_reply_scores_higher_than_intent(self):
        scores = self.matcher.classify(
            "Thank you for contacting us. Our team will respond shortly."
        )
        self.assertGreater(scores["auto_reply"], scores["intent"])

    def test_intent_scores_higher_than_auto(self):
        scores = self.matcher.classify("Ok lets do it. Whats next?")
        self.assertGreater(scores["intent"], scores["auto_reply"])


if __name__ == "__main__":
    unittest.main()
