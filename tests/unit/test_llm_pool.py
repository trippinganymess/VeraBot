"""Deterministic tests for the multi-key Gemini load balancer.

The tests inject a fake ``client_factory`` so we never hit the real
google-genai SDK or the network. Every behaviour under test is purely
internal: round-robin acquire, per-(key, model) cooldowns, model fallback
when every key is rate-limited, and async dispatching.
"""

import asyncio
import time
import unittest
from collections import Counter
from typing import Any
from unittest import mock

from llm_pool import (
    GeminiKeyPool,
    ModelFallbackChain,
    _is_auth_error,
    _is_model_not_found_error,
    _is_rate_limit_error,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str = '{"ok": true}') -> None:
        self.text = text


class _FakeModels:
    def __init__(self, parent: "_FakeClient") -> None:
        self._parent = parent

    def generate_content(self, *, model: str, contents: Any, config: Any = None):
        self._parent.calls.append((model, contents))
        # Pop the next behaviour for (api_key, model). Default: success.
        key = (self._parent.api_key, model)
        behaviours = self._parent.behaviours.get(key, [])
        if behaviours:
            action = behaviours.pop(0)
            if isinstance(action, Exception):
                raise action
        return _FakeResponse()


class _FakeClient:
    """A fake genai.Client whose behaviour is driven by a script."""

    def __init__(self, api_key: str, behaviours: dict) -> None:
        self.api_key = api_key
        self.behaviours = behaviours
        self.calls: list[tuple[str, Any]] = []
        self.models = _FakeModels(self)


def _make_pool(
    keys: list[str],
    behaviours: dict | None = None,
) -> tuple[GeminiKeyPool, dict[str, _FakeClient]]:
    """Build a pool with fake clients. Returns (pool, key->client map)."""
    behaviours = behaviours or {}
    clients: dict[str, _FakeClient] = {}

    def factory(api_key: str) -> _FakeClient:
        client = _FakeClient(api_key, behaviours)
        clients[api_key] = client
        return client

    pool = GeminiKeyPool(keys, client_factory=factory)
    return pool, clients


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification(unittest.TestCase):
    def test_429_message_is_rate_limit(self):
        self.assertTrue(_is_rate_limit_error(Exception("HTTP 429 Too Many Requests")))

    def test_resource_exhausted_is_rate_limit(self):
        self.assertTrue(_is_rate_limit_error(Exception("RESOURCE_EXHAUSTED quota")))

    def test_unrelated_error_is_not_rate_limit(self):
        self.assertFalse(_is_rate_limit_error(ValueError("bad input")))

    def test_401_is_auth_error(self):
        self.assertTrue(_is_auth_error(Exception("HTTP 401 Unauthorized")))

    def test_invalid_api_key_is_auth_error(self):
        self.assertTrue(_is_auth_error(Exception("API key not valid")))

    def test_429_is_not_an_auth_error(self):
        self.assertFalse(_is_auth_error(Exception("HTTP 429")))

    def test_404_model_not_found_is_recognised(self):
        msg = "404 NOT_FOUND. {'error': {'message': 'models/gemma-4-31b is not found'}}"
        self.assertTrue(_is_model_not_found_error(Exception(msg)))

    def test_429_is_not_model_not_found(self):
        self.assertFalse(_is_model_not_found_error(Exception("HTTP 429")))

    def test_404_unrelated_resource_is_not_model_not_found(self):
        # A 404 that does not reference a model should not match.
        self.assertFalse(_is_model_not_found_error(Exception("HTTP 404 page gone")))


# ---------------------------------------------------------------------------
# GeminiKeyPool
# ---------------------------------------------------------------------------


class TestGeminiKeyPool(unittest.TestCase):
    def test_empty_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            GeminiKeyPool([], client_factory=lambda k: object())

    def test_size_matches_key_count(self):
        pool, _ = _make_pool(["a", "b", "c"])
        self.assertEqual(pool.size, 3)
        self.assertEqual(len(pool.keys_masked()), 3)

    def test_round_robin_distribution(self):
        pool, _ = _make_pool(["alpha", "bravo", "charlie"])
        seen = Counter()
        for _ in range(30):
            state = pool.acquire("model-x")
            assert state is not None
            seen[state.api_key] += 1
        # 30 / 3 keys = 10 each, exactly
        self.assertEqual(seen["alpha"], 10)
        self.assertEqual(seen["bravo"], 10)
        self.assertEqual(seen["charlie"], 10)

    def test_cooldown_skips_specific_key_for_specific_model(self):
        pool, _ = _make_pool(["alpha", "bravo"])
        # Bench alpha for model-x only
        pool.mark_failure(pool._states[0], "model-x", 60.0, "rate-limit")

        # 50 acquires on model-x should never return alpha
        for _ in range(50):
            state = pool.acquire("model-x")
            assert state is not None
            self.assertEqual(state.api_key, "bravo")

        # But alpha is still available for a different model
        state = pool.acquire("model-y")
        assert state is not None
        # Model-y traffic round-robins normally
        keys_seen = {pool.acquire("model-y").api_key for _ in range(10)}  # type: ignore[union-attr]
        self.assertEqual(keys_seen, {"alpha", "bravo"})

    def test_returns_none_when_all_keys_cooled_for_a_model(self):
        pool, _ = _make_pool(["alpha", "bravo", "charlie"])
        for state in pool._states:
            pool.mark_failure(state, "model-x", 60.0, "rate-limit")
        self.assertIsNone(pool.acquire("model-x"))
        # Other models are unaffected
        self.assertIsNotNone(pool.acquire("model-y"))

    def test_cooldown_expires_after_window(self):
        pool, _ = _make_pool(["alpha"])
        pool.mark_failure(pool._states[0], "model-x", 0.0, "instant")
        # cooldown ends immediately; next acquire should succeed
        time.sleep(0.001)
        self.assertIsNotNone(pool.acquire("model-x"))

    def test_reset_cooldowns_unbenches_everyone(self):
        pool, _ = _make_pool(["alpha", "bravo"])
        for state in pool._states:
            pool.mark_failure(state, "model-x", 60.0, "test")
        self.assertIsNone(pool.acquire("model-x"))
        pool.reset_cooldowns()
        self.assertIsNotNone(pool.acquire("model-x"))


# ---------------------------------------------------------------------------
# ModelFallbackChain — sync path
# ---------------------------------------------------------------------------


class TestModelFallbackChainSync(unittest.TestCase):
    def setUp(self):
        # NO_LLM may be set by the parent test runner; clear it so the
        # chain reports as available for these tests.
        self._patch = mock.patch.dict("os.environ", {}, clear=False)
        self._patch.start()
        import os

        os.environ.pop("NO_LLM", None)

    def tearDown(self):
        self._patch.stop()

    def test_first_key_first_model_happy_path(self):
        pool, clients = _make_pool(["alpha", "bravo"])
        chain = ModelFallbackChain(pool, ["model-x", "model-y"])
        response, model_used = chain.generate_content_sync("hi")
        self.assertEqual(model_used, "model-x")
        # Exactly one call total — no retries when first attempt succeeds
        total_calls = sum(len(c.calls) for c in clients.values())
        self.assertEqual(total_calls, 1)

    def test_rate_limit_rotates_to_next_key_same_model(self):
        # alpha 429s on model-x once; bravo answers
        behaviours = {("alpha", "model-x"): [Exception("HTTP 429 quota exceeded")]}
        pool, clients = _make_pool(["alpha", "bravo"], behaviours)
        chain = ModelFallbackChain(pool, ["model-x", "model-y"])

        response, model_used = chain.generate_content_sync("hi")
        self.assertEqual(model_used, "model-x")
        self.assertGreaterEqual(len(clients["alpha"].calls), 1)
        self.assertGreaterEqual(len(clients["bravo"].calls), 1)
        # alpha must now be on cooldown for model-x
        self.assertGreater(pool._states[0].cooldown.get("model-x", 0.0), 0.0)

    def test_all_keys_rate_limited_falls_back_to_next_model(self):
        # Both keys 429 on model-x but succeed on model-y
        behaviours = {
            ("alpha", "model-x"): [Exception("429 RESOURCE_EXHAUSTED")],
            ("bravo", "model-x"): [Exception("429 quota")],
        }
        pool, clients = _make_pool(["alpha", "bravo"], behaviours)
        chain = ModelFallbackChain(pool, ["model-x", "model-y"])

        response, model_used = chain.generate_content_sync("hi")
        # Should have fallen back to model-y
        self.assertEqual(model_used, "model-y")
        # Both keys must be benched on model-x
        self.assertGreater(pool._states[0].cooldown.get("model-x", 0.0), 0.0)
        self.assertGreater(pool._states[1].cooldown.get("model-x", 0.0), 0.0)
        # But neither is benched on model-y
        self.assertEqual(pool._states[0].cooldown.get("model-y", 0.0), 0.0)

    def test_full_chain_exhaustion_raises_last_error(self):
        # Every key fails on every model
        behaviours = {
            ("alpha", "model-x"): [Exception("429 a")],
            ("bravo", "model-x"): [Exception("429 b")],
            ("alpha", "model-y"): [Exception("429 c")],
            ("bravo", "model-y"): [Exception("429 d")],
        }
        pool, _ = _make_pool(["alpha", "bravo"], behaviours)
        chain = ModelFallbackChain(pool, ["model-x", "model-y"])
        with self.assertRaises(Exception) as ctx:
            chain.generate_content_sync("hi")
        # The last exception message should bubble up
        self.assertIn("429", str(ctx.exception))

    def test_auth_error_uses_long_cooldown(self):
        # alpha returns a permanent 401; bravo answers
        behaviours = {("alpha", "model-x"): [Exception("HTTP 401 Unauthorized")]}
        pool, _ = _make_pool(["alpha", "bravo"], behaviours)
        chain = ModelFallbackChain(pool, ["model-x"])

        response, model_used = chain.generate_content_sync("hi")
        self.assertEqual(model_used, "model-x")
        # alpha should be benched for an hour-class window (>= 600s)
        cooldown_left = (
            pool._states[0].cooldown.get("model-x", 0.0) - time.monotonic()
        )
        self.assertGreater(cooldown_left, 600.0)

    def test_disabled_chain_reports_unavailable(self):
        chain = ModelFallbackChain(None, ["model-x"])
        self.assertFalse(chain.is_available())

    def test_chain_with_pool_reports_available(self):
        pool, _ = _make_pool(["alpha"])
        chain = ModelFallbackChain(pool, ["model-x"])
        self.assertTrue(chain.is_available())

    def test_no_llm_env_disables_chain(self):
        pool, _ = _make_pool(["alpha"])
        chain = ModelFallbackChain(pool, ["model-x"])
        with mock.patch.dict("os.environ", {"NO_LLM": "1"}):
            self.assertFalse(chain.is_available())

    def test_404_skips_entire_model_in_one_shot(self):
        # Simulate gemma-4-31b being 404'd: the chain should NOT spend
        # one round-trip per key on it. After the first 404, every key
        # must be benched on that model and the chain steps to model-y.
        not_found = "404 NOT_FOUND models/gemma-4-31b is not found"
        behaviours: dict[tuple[str, str], list[Exception]] = {
            ("alpha", "gemma-4-31b"): [Exception(not_found)],
            # bravo would also 404 if asked, but we expect to skip it
            ("bravo", "gemma-4-31b"): [Exception(not_found)],
        }
        pool, clients = _make_pool(["alpha", "bravo"], behaviours)
        chain = ModelFallbackChain(pool, ["gemma-4-31b", "model-y"])

        response, model_used = chain.generate_content_sync("hi")
        self.assertEqual(model_used, "model-y")
        # Exactly ONE 404 attempt was made (the first key); the second key
        # was never asked for the dead model.
        gemma_calls = [
            c for c in clients["alpha"].calls + clients["bravo"].calls
            if c[0] == "gemma-4-31b"
        ]
        self.assertEqual(
            len(gemma_calls), 1,
            "404 should bench every key for the dead model after first hit",
        )


# ---------------------------------------------------------------------------
# ModelFallbackChain — async path
# ---------------------------------------------------------------------------


class TestModelFallbackChainAsync(unittest.TestCase):
    def setUp(self):
        import os

        os.environ.pop("NO_LLM", None)

    def test_async_invocation_returns_response(self):
        pool, _ = _make_pool(["alpha", "bravo"])
        chain = ModelFallbackChain(pool, ["model-x"])

        async def driver():
            return await chain.generate_content("hi")

        response, model_used = asyncio.run(driver())
        self.assertEqual(model_used, "model-x")
        self.assertEqual(response.text, '{"ok": true}')

    def test_async_concurrent_calls_distribute_keys(self):
        pool, clients = _make_pool(["alpha", "bravo", "charlie", "delta"])
        chain = ModelFallbackChain(pool, ["model-x"])

        async def driver():
            return await asyncio.gather(
                *[chain.generate_content("hi") for _ in range(40)]
            )

        results = asyncio.run(driver())
        self.assertEqual(len(results), 40)
        # Round-robin should spread roughly evenly across 4 keys
        per_key = {k: len(c.calls) for k, c in clients.items()}
        for count in per_key.values():
            self.assertGreaterEqual(count, 5)
            self.assertLessEqual(count, 15)


if __name__ == "__main__":
    unittest.main()
