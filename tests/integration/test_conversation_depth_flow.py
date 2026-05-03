import unittest

from fastapi.testclient import TestClient

from bot import app


class TestConversationDepthFlow(unittest.TestCase):
    def test_seven_turn_conversation(self):
        client = TestClient(app)
        merchant_id = "m_depth_001"

        context_response = client.post(
            "/v1/context",
            json={
                "scope": "merchant",
                "context_id": merchant_id,
                "version": 1,
                "payload": {
                    "merchant_id": merchant_id,
                    "category_slug": "dentists",
                    "identity": {"owner_first_name": "Meera", "languages": ["en"]},
                },
                "delivered_at": "2026-05-03T00:00:00Z",
            },
        )
        self.assertEqual(context_response.status_code, 200)

        for turn in range(1, 8):
            reply_response = client.post(
                "/v1/reply",
                json={
                    "conversation_id": "conv_depth_001",
                    "merchant_id": merchant_id,
                    "customer_id": None,
                    "from_role": "merchant",
                    "message": "Tell me more",
                    "received_at": "2026-05-03T00:00:00Z",
                    "turn_number": turn,
                },
            )
            self.assertEqual(reply_response.status_code, 200)
            payload = reply_response.json()
            self.assertIn(payload["action"], {"send", "wait", "end"})
