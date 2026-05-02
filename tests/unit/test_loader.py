import unittest

from bot import CategoryContext, CustomerContext, MerchantContext, TriggerContext, load_data


class TestDataLoader(unittest.TestCase):
    def test_load_data_counts(self):
        data = load_data()
        self.assertEqual(set(data.keys()), {"categories", "merchants", "customers", "triggers"})
        self.assertEqual(len(data["categories"]), 5)
        self.assertEqual(len(data["merchants"]), 50)
        self.assertEqual(len(data["customers"]), 200)
        self.assertEqual(len(data["triggers"]), 100)

    def test_loaded_model_types(self):
        data = load_data()
        any_category = next(iter(data["categories"].values()))
        any_merchant = next(iter(data["merchants"].values()))
        any_customer = next(iter(data["customers"].values()))
        any_trigger = next(iter(data["triggers"].values()))
        self.assertIsInstance(any_category, CategoryContext)
        self.assertIsInstance(any_merchant, MerchantContext)
        self.assertIsInstance(any_customer, CustomerContext)
        self.assertIsInstance(any_trigger, TriggerContext)

    def test_load_data_cache(self):
        data_first = load_data()
        data_second = load_data()
        self.assertIs(data_first, data_second)


if __name__ == "__main__":
    unittest.main()
