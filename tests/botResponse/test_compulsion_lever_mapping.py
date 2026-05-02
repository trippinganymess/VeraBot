import unittest

from bot import (
    VOICE_PREFIX_MAP,
    CategoryContext,
    _apply_compulsion_lever,
    _apply_voice_modulation,
    _build_rationale,
    _select_compulsion_lever,
    compose,
)


class TestLeverMapLookup(unittest.TestCase):
    """Verify the O(1) LEVER_MAP covers all expected trigger kinds."""

    def test_loss_aversion_triggers_map_correctly(self):
        loss_triggers = [
            "perf_dip",
            "missed_search",
            "dormant_with_vera",
            "renewal_due",
            "seasonal_acquisition_dip",
            "winback",
            "customer_lapsed_soft",
        ]
        for kind in loss_triggers:
            self.assertEqual(
                _select_compulsion_lever(kind),
                "loss_aversion",
                f"{kind} should map to loss_aversion",
            )

    def test_social_proof_triggers_map_correctly(self):
        social_triggers = [
            "milestone_reached",
            "review_theme_emerged",
            "competitor_opened",
            "perf_spike",
            "festival_upcoming",
        ]
        for kind in social_triggers:
            self.assertEqual(
                _select_compulsion_lever(kind),
                "social_proof",
                f"{kind} should map to social_proof",
            )

    def test_effort_externalization_triggers_map_correctly(self):
        effort_triggers = [
            "research_digest",
            "curious_ask_due",
            "trial_followup",
            "appointment_tomorrow",
            "recall_due",
            "chronic_refill_due",
            "unverified_gbp",
        ]
        for kind in effort_triggers:
            self.assertEqual(
                _select_compulsion_lever(kind),
                "effort_externalization",
                f"{kind} should map to effort_externalization",
            )

    def test_compound_trigger_kind_falls_back_to_substring(self):
        self.assertEqual(
            _select_compulsion_lever("research_digest_release"),
            "effort_externalization",
        )
        self.assertEqual(
            _select_compulsion_lever("perf_dip_weekly"),
            "loss_aversion",
        )
        self.assertEqual(
            _select_compulsion_lever("milestone_reached_100_reviews"),
            "social_proof",
        )

    def test_unknown_trigger_returns_neutral(self):
        self.assertEqual(_select_compulsion_lever("unknown_trigger"), "neutral")
        self.assertEqual(_select_compulsion_lever(""), "neutral")


class TestApplyCompulsionLeverLanguage(unittest.TestCase):
    """Verify lever cues are appended in the correct language."""

    def test_loss_aversion_english_cue(self):
        result = _apply_compulsion_lever("Base message.", "loss_aversion", "en")
        self.assertIn("missing demand", result.lower())

    def test_loss_aversion_hindi_cue(self):
        result = _apply_compulsion_lever("Base message.", "loss_aversion", "hi-en mix")
        self.assertIn("Missed demand", result)

    def test_social_proof_english_cue(self):
        result = _apply_compulsion_lever("Base message.", "social_proof", "en")
        self.assertIn("peers", result.lower())

    def test_social_proof_hindi_cue(self):
        result = _apply_compulsion_lever("Base message.", "social_proof", "hi-en mix")
        self.assertIn("peers", result.lower())

    def test_effort_externalization_english_cue(self):
        result = _apply_compulsion_lever("Base message.", "effort_externalization", "en")
        self.assertIn("draft it for you", result.lower())

    def test_effort_externalization_hindi_cue(self):
        result = _apply_compulsion_lever("Base message.", "effort_externalization", "hi-en mix")
        self.assertIn("draft ready", result.lower())

    def test_neutral_lever_leaves_message_unchanged(self):
        original = "Base message."
        result = _apply_compulsion_lever(original, "neutral", "en")
        self.assertEqual(result, original)


class TestVoiceModulationAllCategories(unittest.TestCase):
    """Verify voice prefix is applied for every known category slug."""

    def test_dentists_clinical_prefix(self):
        cat = CategoryContext(slug="dentists")
        result = _apply_voice_modulation(cat, "Test message")
        self.assertTrue(result.startswith("Clinical note:"))

    def test_salons_quick_tip_prefix(self):
        cat = CategoryContext(slug="salons")
        result = _apply_voice_modulation(cat, "Test message")
        self.assertTrue(result.startswith("Quick tip:"))

    def test_restaurants_ops_prefix(self):
        cat = CategoryContext(slug="restaurants")
        result = _apply_voice_modulation(cat, "Test message")
        self.assertTrue(result.startswith("Quick ops note:"))

    def test_gyms_coach_prefix(self):
        cat = CategoryContext(slug="gyms")
        result = _apply_voice_modulation(cat, "Test message")
        self.assertTrue(result.startswith("Coach's note:"))

    def test_pharmacies_compliance_prefix(self):
        cat = CategoryContext(slug="pharmacies")
        result = _apply_voice_modulation(cat, "Test message")
        self.assertTrue(result.startswith("Compliance note:"))

    def test_unknown_category_no_prefix(self):
        cat = CategoryContext(slug="unknown_slug")
        result = _apply_voice_modulation(cat, "Test message")
        self.assertEqual(result, "Test message")


class TestBuildRationale(unittest.TestCase):
    """Verify _build_rationale returns meaningful strategy explanations."""

    def test_auto_reply_rationale_mentions_exit(self):
        rationale = _build_rationale("auto_reply_exit", "neutral", "dentists", "perf_dip")
        self.assertIn("auto-reply", rationale.lower())

    def test_intent_rationale_mentions_action_mode(self):
        rationale = _build_rationale("intent_transition", "neutral", "salons", "onboarding")
        self.assertIn("action mode", rationale.lower())

    def test_digest_rationale_mentions_lever(self):
        rationale = _build_rationale(
            "digest_anchor", "effort_externalization", "dentists", "research_digest"
        )
        self.assertIn("effort externalization", rationale.lower())
        self.assertIn("research_digest", rationale)

    def test_benchmark_rationale_mentions_peer(self):
        rationale = _build_rationale(
            "benchmark_anchor", "loss_aversion", "dentists", "perf_dip"
        )
        self.assertIn("peer-median", rationale.lower())
        self.assertIn("loss aversion", rationale.lower())

    def test_rationale_always_includes_voice_label(self):
        for slug in VOICE_PREFIX_MAP:
            rationale = _build_rationale("fallback", "neutral", slug, "test")
            self.assertIn(slug, rationale)


class TestComposeEndToEndStage4(unittest.TestCase):
    """End-to-end compose tests validating lever + voice integration."""

    def test_loss_aversion_lever_in_perf_dip_output(self):
        result = compose(
            {"slug": "dentists", "peer_stats": {"avg_ctr": 0.03}},
            {
                "merchant_id": "m_020",
                "category_slug": "dentists",
                "performance": {"ctr": 0.02, "views": 900},
                "identity": {"languages": ["en"], "owner_first_name": "Meera"},
            },
            {
                "id": "trg_020",
                "scope": "merchant",
                "kind": "perf_dip",
                "source": "internal",
                "suppression_key": "s20",
            },
            None,
        )
        self.assertIn("missing demand", result["body"].lower())
        self.assertTrue(result["body"].startswith("Clinical note:"))
        self.assertIn("loss aversion", result["rationale"].lower())

    def test_voice_modulation_prefix_for_dentists(self):
        result = compose(
            {"slug": "dentists"},
            {
                "merchant_id": "m_021",
                "category_slug": "dentists",
                "performance": {"ctr": 0.02, "views": 1200},
                "identity": {"languages": ["en"], "owner_first_name": "Meera"},
            },
            {
                "id": "trg_021",
                "scope": "merchant",
                "kind": "perf_spike",
                "source": "internal",
                "suppression_key": "s21",
            },
            None,
        )
        self.assertTrue(result["body"].startswith("Clinical note:"))

    def test_social_proof_lever_for_milestone_trigger(self):
        result = compose(
            {"slug": "salons"},
            {
                "merchant_id": "m_022",
                "category_slug": "salons",
                "identity": {"languages": ["en"], "owner_first_name": "Priya"},
            },
            {
                "id": "trg_022",
                "scope": "merchant",
                "kind": "milestone_reached",
                "source": "internal",
                "suppression_key": "s22",
            },
            None,
        )
        self.assertIn("peers", result["body"].lower())
        self.assertTrue(result["body"].startswith("Quick tip:"))

    def test_effort_externalization_lever_for_digest_trigger(self):
        result = compose(
            {
                "slug": "dentists",
                "digest": [
                    {
                        "title": "Fluoride recall improves outcomes",
                        "source": "JIDA Oct 2026",
                        "trial_n": 2100,
                    }
                ],
            },
            {
                "merchant_id": "m_023",
                "category_slug": "dentists",
                "identity": {"languages": ["en"], "owner_first_name": "Rajan"},
            },
            {
                "id": "trg_023",
                "scope": "merchant",
                "kind": "research_digest_release",
                "source": "external",
                "suppression_key": "s23",
                "payload": {},
            },
            None,
        )
        self.assertIn("draft it for you", result["body"].lower())
        self.assertIn("effort externalization", result["rationale"].lower())

    def test_restaurants_voice_prefix(self):
        result = compose(
            {"slug": "restaurants"},
            {
                "merchant_id": "m_024",
                "category_slug": "restaurants",
                "performance": {"ctr": 0.015, "views": 500},
                "identity": {"languages": ["en"], "owner_first_name": "Amit"},
            },
            {
                "id": "trg_024",
                "scope": "merchant",
                "kind": "perf_dip",
                "source": "internal",
                "suppression_key": "s24",
            },
            None,
        )
        self.assertTrue(result["body"].startswith("Quick ops note:"))


if __name__ == "__main__":
    unittest.main()
