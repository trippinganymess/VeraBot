"""Tests for the modular prompt template system and JIT fact extraction."""

import unittest

from bot import (
    LEVER_TEMPLATES,
    SYSTEM_PROMPT,
    CategoryContext,
    CustomerContext,
    MerchantContext,
    TriggerContext,
    _build_llm_prompt,
    _extract_jit_facts,
)


class TestSystemPromptContent(unittest.TestCase):
    """Verify the base system prompt contains required persona and rules."""

    def test_system_prompt_defines_vera_persona(self):
        self.assertIn("Vera", SYSTEM_PROMPT)

    def test_system_prompt_enforces_json_output_keys(self):
        self.assertIn("body", SYSTEM_PROMPT)
        self.assertIn("cta", SYSTEM_PROMPT)
        self.assertIn("send_as", SYSTEM_PROMPT)
        self.assertIn("rationale", SYSTEM_PROMPT)

    def test_system_prompt_prohibits_fabrication(self):
        self.assertIn("Never fabricate", SYSTEM_PROMPT)

    def test_system_prompt_enforces_yes_stop_cta_rule(self):
        self.assertIn("YES or STOP", SYSTEM_PROMPT)


class TestLeverTemplatesCoverage(unittest.TestCase):
    """Verify all compulsion levers have a corresponding template."""

    def test_social_proof_template_references_peers(self):
        self.assertIn("peers", LEVER_TEMPLATES["social_proof"].lower())

    def test_loss_aversion_template_references_missing(self):
        self.assertIn("missing", LEVER_TEMPLATES["loss_aversion"].lower())

    def test_effort_externalization_template_references_draft(self):
        self.assertIn("drafted", LEVER_TEMPLATES["effort_externalization"].lower())

    def test_neutral_template_exists(self):
        self.assertIn("neutral", LEVER_TEMPLATES)

    def test_all_lever_templates_are_non_empty_strings(self):
        for lever, template in LEVER_TEMPLATES.items():
            self.assertIsInstance(template, str, f"{lever} is not a string")
            self.assertTrue(len(template) > 20, f"{lever} template too short")


class TestJitFactExtraction(unittest.TestCase):
    """Verify _extract_jit_facts extracts only verifiable data points."""

    def test_extracts_merchant_name_from_identity(self):
        merchant = MerchantContext(
            merchant_id="m_test",
            category_slug="dentists",
            identity={"owner_first_name": "Dr. Meera"},
        )
        category = CategoryContext(slug="dentists")
        trigger = TriggerContext(
            id="t_test", scope="merchant", kind="perf_dip",
            source="internal", suppression_key="sk",
        )
        facts = _extract_jit_facts(merchant, category, trigger, None, {}, None)
        self.assertEqual(facts["merchant_name"], "Dr. Meera")

    def test_extracts_performance_metrics(self):
        merchant = MerchantContext(
            merchant_id="m_test",
            category_slug="dentists",
            performance={"views": 2410, "ctr": 0.021, "calls": 18},
        )
        category = CategoryContext(slug="dentists")
        trigger = TriggerContext(
            id="t_test", scope="merchant", kind="perf_dip",
            source="internal", suppression_key="sk",
        )
        facts = _extract_jit_facts(merchant, category, trigger, None, {}, None)
        self.assertEqual(facts["views_30d"], 2410)
        self.assertEqual(facts["ctr"], "2.1%")
        self.assertEqual(facts["calls_30d"], 18)

    def test_includes_benchmark_when_provided(self):
        merchant = MerchantContext(
            merchant_id="m_test", category_slug="dentists",
        )
        category = CategoryContext(slug="dentists")
        trigger = TriggerContext(
            id="t_test", scope="merchant", kind="perf_dip",
            source="internal", suppression_key="sk",
        )
        benchmark = {"ctr_gap": "2.1% vs peer 3.0%"}
        facts = _extract_jit_facts(merchant, category, trigger, None, benchmark, None)
        self.assertEqual(facts["benchmark"], benchmark)

    def test_includes_digest_when_provided(self):
        merchant = MerchantContext(
            merchant_id="m_test", category_slug="dentists",
        )
        category = CategoryContext(slug="dentists")
        trigger = TriggerContext(
            id="t_test", scope="merchant", kind="research_digest",
            source="external", suppression_key="sk",
        )
        digest = {"title": "Fluoride trial", "source": "JIDA", "trial_n": "2100"}
        facts = _extract_jit_facts(merchant, category, trigger, None, {}, digest)
        self.assertEqual(facts["digest_title"], "Fluoride trial")
        self.assertEqual(facts["digest_source"], "JIDA")

    def test_omits_digest_fields_when_none(self):
        merchant = MerchantContext(
            merchant_id="m_test", category_slug="dentists",
        )
        category = CategoryContext(slug="dentists")
        trigger = TriggerContext(
            id="t_test", scope="merchant", kind="perf_dip",
            source="internal", suppression_key="sk",
        )
        facts = _extract_jit_facts(merchant, category, trigger, None, {}, None)
        self.assertNotIn("digest_title", facts)

    def test_includes_customer_name_when_customer_present(self):
        merchant = MerchantContext(
            merchant_id="m_test", category_slug="dentists",
        )
        category = CategoryContext(slug="dentists")
        trigger = TriggerContext(
            id="t_test", scope="customer", kind="recall_due",
            source="internal", suppression_key="sk",
        )
        customer = CustomerContext(
            customer_id="c_test", merchant_id="m_test",
            identity={"name": "Priya"},
        )
        facts = _extract_jit_facts(merchant, category, trigger, customer, {}, None)
        self.assertEqual(facts["customer_name"], "Priya")

    def test_voice_prefix_matches_category(self):
        merchant = MerchantContext(
            merchant_id="m_test", category_slug="gyms",
        )
        category = CategoryContext(slug="gyms")
        trigger = TriggerContext(
            id="t_test", scope="merchant", kind="perf_dip",
            source="internal", suppression_key="sk",
        )
        facts = _extract_jit_facts(merchant, category, trigger, None, {}, None)
        self.assertEqual(facts["voice_prefix"], "Coach's note:")


class TestBuildLlmPrompt(unittest.TestCase):
    """Verify _build_llm_prompt assembles prompts correctly."""

    def test_prompt_contains_system_prompt(self):
        prompt = _build_llm_prompt(
            lever="loss_aversion", language_pref="en",
            cta="open_ended", send_as="vera",
            facts={"merchant_name": "Dr. Meera"},
            draft_body="Test body", draft_rationale="Test rationale",
        )
        self.assertIn("You are Vera", prompt)

    def test_prompt_contains_lever_template(self):
        prompt = _build_llm_prompt(
            lever="social_proof", language_pref="en",
            cta="open_ended", send_as="vera",
            facts={"merchant_name": "Dr. Meera"},
            draft_body="Test body", draft_rationale="Test rationale",
        )
        self.assertIn("SOCIAL PROOF", prompt)

    def test_prompt_contains_hindi_instruction_for_hi_pref(self):
        prompt = _build_llm_prompt(
            lever="neutral", language_pref="hi-en mix",
            cta="open_ended", send_as="vera",
            facts={"merchant_name": "Dr. Meera"},
            draft_body="Test body", draft_rationale="Test rationale",
        )
        self.assertIn("Hindi-English code-mix", prompt)

    def test_prompt_contains_english_instruction_for_en_pref(self):
        prompt = _build_llm_prompt(
            lever="neutral", language_pref="en",
            cta="open_ended", send_as="vera",
            facts={"merchant_name": "Dr. Meera"},
            draft_body="Test body", draft_rationale="Test rationale",
        )
        self.assertIn("Language: English", prompt)

    def test_prompt_includes_extracted_facts(self):
        facts = {"merchant_name": "Dr. Meera", "views_30d": 2410, "ctr": "2.1%"}
        prompt = _build_llm_prompt(
            lever="loss_aversion", language_pref="en",
            cta="open_ended", send_as="vera",
            facts=facts,
            draft_body="Test body", draft_rationale="Test rationale",
        )
        self.assertIn("Dr. Meera", prompt)
        self.assertIn("2410", prompt)
        self.assertIn("2.1%", prompt)

    def test_prompt_includes_draft_body(self):
        prompt = _build_llm_prompt(
            lever="neutral", language_pref="en",
            cta="open_ended", send_as="vera",
            facts={},
            draft_body="This is my draft body.", draft_rationale="Test",
        )
        self.assertIn("This is my draft body.", prompt)

    def test_prompt_falls_back_to_neutral_for_unknown_lever(self):
        prompt = _build_llm_prompt(
            lever="unknown_lever", language_pref="en",
            cta="open_ended", send_as="vera",
            facts={},
            draft_body="Test body", draft_rationale="Test rationale",
        )
        self.assertIn("peer-toned", prompt)


if __name__ == "__main__":
    unittest.main()
