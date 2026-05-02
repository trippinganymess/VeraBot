"""Tests for binary CTA enforcement and suppression key dedup."""

import unittest

from bot import (
    ACTION_TRIGGERS,
    _check_suppression_dedup,
    _enforce_binary_cta,
    _is_action_trigger,
    _suppression_store,
    compose,
)


class TestActionTriggerClassification(unittest.TestCase):
    """Verify _is_action_trigger correctly identifies action triggers."""

    def test_recall_due_is_action_trigger(self):
        self.assertTrue(_is_action_trigger("recall_due"))

    def test_appointment_tomorrow_is_action_trigger(self):
        self.assertTrue(_is_action_trigger("appointment_tomorrow"))

    def test_trial_followup_is_action_trigger(self):
        self.assertTrue(_is_action_trigger("trial_followup"))

    def test_chronic_refill_due_is_action_trigger(self):
        self.assertTrue(_is_action_trigger("chronic_refill_due"))

    def test_renewal_due_is_action_trigger(self):
        self.assertTrue(_is_action_trigger("renewal_due"))

    def test_perf_dip_is_not_action_trigger(self):
        self.assertFalse(_is_action_trigger("perf_dip"))

    def test_research_digest_is_not_action_trigger(self):
        self.assertFalse(_is_action_trigger("research_digest"))

    def test_compound_action_trigger_detected_via_substring(self):
        self.assertTrue(_is_action_trigger("recall_due_patient_priya"))

    def test_action_triggers_frozenset_has_expected_count(self):
        self.assertEqual(len(ACTION_TRIGGERS), 7)


class TestEnforceBinaryCta(unittest.TestCase):
    """Verify _enforce_binary_cta appends YES/STOP correctly."""

    def test_appends_cta_suffix_english(self):
        result = _enforce_binary_cta("Your recall is due.", "en")
        self.assertTrue(result.endswith("YES"))
        self.assertIn("STOP", result)

    def test_appends_cta_suffix_hindi(self):
        result = _enforce_binary_cta("Aapka recall due hai.", "hi-en mix")
        self.assertTrue(result.endswith("YES"))
        self.assertIn("confirm", result.lower())

    def test_does_not_double_append_if_already_ends_with_yes(self):
        body = "Ready to proceed? Reply YES"
        result = _enforce_binary_cta(body, "en")
        self.assertEqual(result, body)

    def test_does_not_double_append_if_already_ends_with_stop(self):
        body = "Want to cancel? Reply STOP"
        result = _enforce_binary_cta(body, "en")
        self.assertEqual(result, body)


class TestSuppressionDedup(unittest.TestCase):
    """Verify _check_suppression_dedup prevents verbatim repeats."""

    def setUp(self):
        _suppression_store.clear()

    def test_first_send_is_not_repeat(self):
        self.assertFalse(
            _check_suppression_dedup("key_a", "Hello merchant")
        )

    def test_same_body_same_key_is_repeat(self):
        _check_suppression_dedup("key_b", "Hello merchant")
        self.assertTrue(
            _check_suppression_dedup("key_b", "Hello merchant")
        )

    def test_different_body_same_key_is_not_repeat(self):
        _check_suppression_dedup("key_c", "First message")
        self.assertFalse(
            _check_suppression_dedup("key_c", "Different message")
        )

    def test_none_key_never_repeats(self):
        self.assertFalse(_check_suppression_dedup(None, "Any body"))
        self.assertFalse(_check_suppression_dedup(None, "Any body"))


class TestComposeCtaEnforcement(unittest.TestCase):
    """E2E test: action triggers produce yes_no CTA in compose output."""

    def test_recall_due_trigger_produces_yes_no_cta(self):
        result = compose(
            category={"slug": "dentists"},
            merchant={
                "merchant_id": "m_test",
                "category_slug": "dentists",
                "performance": {"views": 100, "ctr": 0.02},
            },
            trigger={
                "id": "t_recall",
                "scope": "merchant",
                "kind": "recall_due",
                "source": "internal",
                "suppression_key": "recall:test",
            },
        )
        self.assertEqual(result["cta"], "yes_no")

    def test_perf_dip_trigger_does_not_produce_yes_no(self):
        result = compose(
            category={"slug": "dentists"},
            merchant={
                "merchant_id": "m_test",
                "category_slug": "dentists",
                "performance": {"views": 100, "ctr": 0.02},
            },
            trigger={
                "id": "t_perf",
                "scope": "merchant",
                "kind": "perf_dip",
                "source": "internal",
                "suppression_key": "perf:test",
            },
        )
        self.assertNotEqual(result["cta"], "yes_no")


class TestComposeSuppression(unittest.TestCase):
    """E2E test: repeated identical sends are tagged in the rationale."""

    def setUp(self):
        _suppression_store.clear()

    def test_second_identical_call_tags_suppressed_repeat(self):
        args = dict(
            category={"slug": "dentists"},
            merchant={
                "merchant_id": "m_test",
                "category_slug": "dentists",
            },
            trigger={
                "id": "t_test",
                "scope": "merchant",
                "kind": "perf_dip",
                "source": "internal",
                "suppression_key": "test:dedup",
            },
        )
        first = compose(**args)
        second = compose(**args)
        self.assertNotIn("[SUPPRESSED REPEAT]", first["rationale"])
        self.assertIn("[SUPPRESSED REPEAT]", second["rationale"])


if __name__ == "__main__":
    unittest.main()
