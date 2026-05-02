import unittest

from bot import compose


class TestStage2Compose(unittest.TestCase):
    def test_auto_reply_triggers_exit(self):
        category = {"slug": "dentists"}
        merchant = {
            "merchant_id": "m_001",
            "category_slug": "dentists",
            "identity": {"languages": ["hi"]},
            "conversation_history": [
                {"from": "merchant", "body": "Thank you for contacting us."},
                {"from": "merchant", "body": "Thank you for contacting us."},
            ],
        }
        trigger = {
            "id": "trg_001",
            "scope": "merchant",
            "kind": "perf_dip",
            "source": "internal",
            "suppression_key": "s1",
        }
        result = compose(category, merchant, trigger, None)
        self.assertEqual(result["cta"], "none")
        self.assertIn("auto-reply", result["body"].lower())

    def test_intent_transition(self):
        category = {"slug": "dentists"}
        merchant = {
            "merchant_id": "m_002",
            "category_slug": "dentists",
            "identity": {"languages": ["en"]},
            "conversation_history": [
                {"from": "merchant", "body": "I want to join"},
            ],
        }
        trigger = {
            "id": "trg_002",
            "scope": "merchant",
            "kind": "onboarding",
            "source": "internal",
            "suppression_key": "s2",
        }
        result = compose(category, merchant, trigger, None)
        self.assertEqual(result["cta"], "yes_no")
        self.assertTrue(result["body"].strip().endswith("STOP"))


if __name__ == "__main__":
    unittest.main()
