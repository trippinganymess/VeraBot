import unittest

from bot import compose


class TestCompulsionLeverMapping(unittest.TestCase):
    def test_loss_aversion_lever(self):
        category = {"slug": "dentists", "peer_stats": {"avg_ctr": 0.03}}
        merchant = {
            "merchant_id": "m_020",
            "category_slug": "dentists",
            "performance": {"ctr": 0.02, "views": 900},
            "identity": {"languages": ["en"], "owner_first_name": "Meera"},
        }
        trigger = {
            "id": "trg_020",
            "scope": "merchant",
            "kind": "perf_dip",
            "source": "internal",
            "suppression_key": "s20",
        }
        result = compose(category, merchant, trigger, None)
        self.assertIn("missing demand", result["body"].lower())

    def test_voice_modulation_prefix(self):
        category = {"slug": "dentists"}
        merchant = {
            "merchant_id": "m_021",
            "category_slug": "dentists",
            "performance": {"ctr": 0.02, "views": 1200},
            "identity": {"languages": ["en"], "owner_first_name": "Meera"},
        }
        trigger = {
            "id": "trg_021",
            "scope": "merchant",
            "kind": "perf_spike",
            "source": "internal",
            "suppression_key": "s21",
        }
        result = compose(category, merchant, trigger, None)
        self.assertTrue(result["body"].startswith("Clinical note:"))


if __name__ == "__main__":
    unittest.main()
