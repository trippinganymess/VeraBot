import unittest

from bot import CategoryContext, ComposedMessage


class TestComposedMessage(unittest.TestCase):
    def test_yes_no_cta_position(self):
        category = CategoryContext(slug="dentists")
        message = ComposedMessage.model_validate(
            {
                "body": "Namaste, context ready. Reply YES",
                "cta": "yes_no",
                "send_as": "vera",
                "suppression_key": "test",
                "rationale": "Test",
            },
            context={"category": category, "language_pref": "hi-en mix"},
        )
        self.assertEqual(message.cta, "yes_no")


if __name__ == "__main__":
    unittest.main()
