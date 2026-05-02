import unittest
from typing import cast

from bot import CategoryContext, CustomerContext, MerchantContext, TriggerContext, compose, load_data


class TestStage1Compose(unittest.TestCase):
    def setUp(self):
        self.data = load_data()

    def _pick_customer_trigger(self):
        for trigger in self.data["triggers"].values():
            trigger = cast(TriggerContext, trigger)
            if trigger.scope == "customer" and trigger.customer_id:
                return trigger
        return None

    def test_send_as_customer_scope(self):
        trigger = self._pick_customer_trigger()
        self.assertIsNotNone(trigger, "No customer-scoped trigger found in dataset.")
        trigger = cast(TriggerContext, trigger)
        self.assertIsNotNone(trigger.merchant_id)
        self.assertIsNotNone(trigger.customer_id)
        customer_id = cast(str, trigger.customer_id)
        merchant_id = cast(str, trigger.merchant_id)
        customer = cast(CustomerContext, self.data["customers"][customer_id])
        merchant = cast(MerchantContext, self.data["merchants"][merchant_id])
        category = cast(CategoryContext, self.data["categories"][merchant.category_slug])
        result = compose(category.model_dump(), merchant.model_dump(), trigger.model_dump(), customer.model_dump())
        self.assertEqual(result["send_as"], "merchant_on_behalf")

    def test_category_mismatch_raises(self):
        trigger = cast(TriggerContext, next(iter(self.data["triggers"].values())))
        self.assertIsNotNone(trigger.merchant_id)
        merchant_id = cast(str, trigger.merchant_id)
        merchant = cast(MerchantContext, self.data["merchants"][merchant_id])
        category = cast(CategoryContext, next(iter(self.data["categories"].values())))
        if category.slug == merchant.category_slug:
            categories = [cast(CategoryContext, c) for c in self.data["categories"].values()]
            category = cast(
                CategoryContext,
                [c for c in categories if c.slug != merchant.category_slug][0],
            )
        with self.assertRaises(ValueError):
            compose(category.model_dump(), merchant.model_dump(), trigger.model_dump(), None)


if __name__ == "__main__":
    unittest.main()
