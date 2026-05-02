import unittest

from bot import compose


class TestStage1Compose(unittest.TestCase):
    def test_send_as_customer_scope(self):
        category = {"slug": "dentists"}
        merchant = {
            "merchant_id": "m_test",
            "category_slug": "dentists",
            "identity": {"name": "Test Clinic"}
        }
        trigger = {
            "id": "t_test",
            "scope": "customer",
            "kind": "recall_due",
            "source": "internal",
            "customer_id": "c_test",
            "merchant_id": "m_test",
            "suppression_key": "test"
        }
        customer = {
            "customer_id": "c_test",
            "merchant_id": "m_test",
            "identity": {"name": "Priya"}
        }
        result = compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )
        self.assertEqual(result["send_as"], "merchant_on_behalf")

    def test_category_mismatch_raises(self):
        category = {"slug": "salons"}
        merchant = {
            "merchant_id": "m_test",
            "category_slug": "dentists",
        }
        trigger = {
            "id": "t_test",
            "scope": "merchant",
            "kind": "perf_dip",
            "source": "internal",
            "merchant_id": "m_test",
        }
        with self.assertRaises(ValueError):
            compose(category=category, merchant=merchant, trigger=trigger, customer=None)


if __name__ == "__main__":
    unittest.main()
