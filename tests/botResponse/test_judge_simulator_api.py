import unittest
from fastapi.testclient import TestClient

from bot import app, _context_store


class TestJudgeSimulatorAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Clear the in-memory context store before each test
        for key in _context_store:
            _context_store[key].clear()

    def test_healthz(self):
        response = self.client.get("/v1/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("uptime_seconds", data)
        self.assertIn("contexts_loaded", data)
        self.assertEqual(data["contexts_loaded"]["category"], 0)
        self.assertEqual(data["contexts_loaded"]["merchant"], 0)

    def test_metadata(self):
        response = self.client.get("/v1/metadata")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("team_name", data)
        self.assertIn("model", data)
        self.assertIn("version", data)

    def test_context_push_new(self):
        payload = {
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "delivered_at": "2026-04-26T09:45:00Z",
            "payload": {
                "slug": "dentists",
                "voice": {"tone": "peer_clinical"}
            }
        }
        response = self.client.post("/v1/context", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["accepted"])
        self.assertIn("ack_id", data)

        # Verify it's stored
        self.assertIn("dentists", _context_store["category"])
        self.assertEqual(_context_store["category"]["dentists"]["version"], 1)

    def test_context_push_idempotent(self):
        payload = {
            "scope": "merchant",
            "context_id": "m_test",
            "version": 1,
            "delivered_at": "2026-04-26T09:45:00Z",
            "payload": {"merchant_id": "m_test", "category_slug": "dentists"}
        }
        self.client.post("/v1/context", json=payload)
        
        # Push same version again
        response = self.client.post("/v1/context", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["accepted"])

    def test_context_push_stale_version(self):
        payload = {
            "scope": "merchant",
            "context_id": "m_test",
            "version": 2,
            "delivered_at": "2026-04-26T09:45:00Z",
            "payload": {"merchant_id": "m_test", "category_slug": "dentists"}
        }
        self.client.post("/v1/context", json=payload)
        
        # Try pushing older version
        payload["version"] = 1
        response = self.client.post("/v1/context", json=payload)
        self.assertEqual(response.status_code, 409)
        data = response.json()
        self.assertFalse(data["accepted"])
        self.assertEqual(data["reason"], "stale_version")
        self.assertEqual(data["current_version"], 2)

    def test_tick_empty_when_no_triggers(self):
        response = self.client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["actions"], [])

    def test_tick_with_valid_trigger(self):
        # Setup context
        self.client.post("/v1/context", json={
            "scope": "category", "context_id": "dentists", "version": 1, "delivered_at": "Z",
            "payload": {"slug": "dentists"}
        })
        self.client.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_001", "version": 1, "delivered_at": "Z",
            "payload": {"merchant_id": "m_001", "category_slug": "dentists", "identity": {"name": "Test"}}
        })
        self.client.post("/v1/context", json={
            "scope": "trigger", "context_id": "t_001", "version": 1, "delivered_at": "Z",
            "payload": {"id": "t_001", "scope": "merchant", "kind": "perf_dip", "source": "internal", "merchant_id": "m_001", "suppression_key": "s_1"}
        })

        response = self.client.post("/v1/tick", json={
            "now": "2026-04-26T10:35:00Z", "available_triggers": ["t_001"]
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["actions"]), 1)
        action = data["actions"][0]
        self.assertEqual(action["merchant_id"], "m_001")
        self.assertEqual(action["trigger_id"], "t_001")
        self.assertIn("body", action)
        self.assertIn("cta", action)

    def test_reply_hostile(self):
        # Test hostile reply handling
        self.client.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_001", "version": 1, "delivered_at": "Z",
            "payload": {"merchant_id": "m_001", "category_slug": "dentists"}
        })
        
        response = self.client.post("/v1/reply", json={
            "conversation_id": "c_1", "merchant_id": "m_001", "from_role": "merchant",
            "message": "Stop messaging me. This is useless spam.", "received_at": "Z", "turn_number": 2
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["action"], "end")

    def test_reply_wait_intent(self):
        self.client.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_001", "version": 1, "delivered_at": "Z",
            "payload": {"merchant_id": "m_001", "category_slug": "dentists"}
        })
        
        response = self.client.post("/v1/reply", json={
            "conversation_id": "c_1", "merchant_id": "m_001", "from_role": "merchant",
            "message": "I am not ready, give me some time.", "received_at": "Z", "turn_number": 2
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["action"], "wait")
        self.assertGreater(data["wait_seconds"], 0)

    def test_reply_auto_reply_loop_ends(self):
        self.client.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_001", "version": 1, "delivered_at": "Z",
            "payload": {"merchant_id": "m_001", "category_slug": "dentists"}
        })

        # Send same auto-reply 4 times
        for i in range(4):
            response = self.client.post("/v1/reply", json={
                "conversation_id": "c_1", "merchant_id": "m_001", "from_role": "merchant",
                "message": "Thank you for contacting us.", "received_at": "Z", "turn_number": i + 2
            })
            
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "end")

    def test_reply_intent_transition(self):
        self.client.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_001", "version": 1, "delivered_at": "Z",
            "payload": {"merchant_id": "m_001", "category_slug": "dentists"}
        })

        response = self.client.post("/v1/reply", json={
            "conversation_id": "c_1", "merchant_id": "m_001", "from_role": "merchant",
            "message": "Ok let's do it.", "received_at": "Z", "turn_number": 2
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["action"], "send")
        self.assertIn("body", data)
        self.assertEqual(data["cta"], "open_ended")


if __name__ == "__main__":
    unittest.main()
