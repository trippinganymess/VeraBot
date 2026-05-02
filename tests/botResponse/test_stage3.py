import unittest

from bot import compose


class TestStage3Compose(unittest.TestCase):
    def test_benchmark_includes_peer_ctr(self):
        category = {
            "slug": "dentists",
            "peer_stats": {"avg_ctr": 0.03},
        }
        merchant = {
            "merchant_id": "m_010",
            "category_slug": "dentists",
            "performance": {"ctr": 0.021, "views": 2410},
            "identity": {"languages": ["en"], "owner_first_name": "Meera"},
        }
        trigger = {
            "id": "trg_010",
            "scope": "merchant",
            "kind": "perf_dip",
            "source": "internal",
            "suppression_key": "s10",
        }
        result = compose(category, merchant, trigger, None)
        self.assertIn("CTR", result["body"])
        self.assertIn("peer", result["body"].lower())

    def test_research_digest_anchor(self):
        category = {"slug": "dentists"}
        merchant = {
            "merchant_id": "m_011",
            "category_slug": "dentists",
            "identity": {"languages": ["en"], "owner_first_name": "Meera"},
        }
        trigger = {
            "id": "trg_011",
            "scope": "merchant",
            "kind": "research_digest_release",
            "source": "external",
            "suppression_key": "s11",
            "payload": {
                "top_item": {
                    "title": "3-month fluoride recall outperforms 6-month",
                    "source": "JIDA Oct 2026",
                }
            },
        }
        result = compose(category, merchant, trigger, None)
        self.assertIn("research digest", result["body"].lower())
        self.assertIn("JIDA", result["body"])

    def test_research_digest_fallback_to_category(self):
        category = {
            "slug": "dentists",
            "digest": [
                {
                    "title": "Fluoride recall improves outcomes",
                    "source": "JIDA Oct 2026",
                    "trial_n": 2100,
                }
            ],
        }
        merchant = {
            "merchant_id": "m_012",
            "category_slug": "dentists",
            "identity": {"languages": ["en"], "owner_first_name": "Meera"},
        }
        trigger = {
            "id": "trg_012",
            "scope": "merchant",
            "kind": "research_digest_release",
            "source": "external",
            "suppression_key": "s12",
            "payload": {},
        }
        result = compose(category, merchant, trigger, None)
        self.assertIn("Fluoride recall", result["body"])
        self.assertIn("JIDA", result["body"])

    def test_benchmark_without_peer_stats(self):
        category = {"slug": "dentists"}
        merchant = {
            "merchant_id": "m_013",
            "category_slug": "dentists",
            "performance": {"ctr": 0.02, "views": 1200},
            "identity": {"languages": ["en"], "owner_first_name": "Meera"},
        }
        trigger = {
            "id": "trg_013",
            "scope": "merchant",
            "kind": "perf_spike",
            "source": "internal",
            "suppression_key": "s13",
        }
        result = compose(category, merchant, trigger, None)
        self.assertIn("CTR", result["body"])


if __name__ == "__main__":
    unittest.main()
