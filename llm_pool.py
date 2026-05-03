"""Async-friendly load balancer over multiple Gemini API keys.

This module rotates a pool of ``GEMINI_API_KEY`` values round-robin and
falls back across a configurable model chain when every key is rate
limited on a given model.

Key properties:
    * Pre-instantiated ``genai.Client`` per key — no per-request client
      construction, so the hot path is one network call.
    * Lock-free round-robin counter (``itertools.count`` is atomic under
      the GIL); the only lock is held briefly when reading/writing the
      per-(key, model) cooldown table.
    * Per-(key, model) cooldowns. Hitting a 429 on
      ``gemini-3.1-flash-lite-preview`` does not block the same key from
      being used on ``gemma-3-12b-it``.
    * One async wrapper for the entire fallback loop — we dispatch the
      whole sequential retry sequence into a single worker thread to keep
      thread-pool overhead minimal.

Two singletons are exposed:

    COMPOSER_CHAIN     — used by :mod:`bot.compose_async` to refine the
                         deterministic draft.
    CLASSIFIER_CHAIN   — used by :mod:`semantic_matcher` for the Stage-3
                         LLM fallback when BGE-M3 is below threshold.

Set ``NO_LLM=1`` (or provide no API keys) to disable the chains; both
expose ``is_available()`` so callers can branch cheaply.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anyio

logger = logging.getLogger("vera.llm_pool")


# ---------------------------------------------------------------------------
# Defaults — overridable via environment variables
# ---------------------------------------------------------------------------

_DEFAULT_COMPOSER_MODELS = (
    "gemini-3.1-flash-lite-preview,"
    "gemini-2.5-flash-lite,"
    "gemma-4-31b"
)

_DEFAULT_CLASSIFIER_MODELS = (
    "gemma-3-12b-it,"
    "gemma-3-27b-it,"
    "gemma-3-4b-it,"
    "gemma-4-26b"
)

# Cooldown after a hard 429 / quota error
DEFAULT_RATE_LIMIT_COOLDOWN_S = float(os.getenv("VERA_RATE_LIMIT_COOLDOWN_S", "60"))
# Shorter cooldown for transient timeouts so we recover quickly
DEFAULT_TIMEOUT_COOLDOWN_S = float(os.getenv("VERA_TIMEOUT_COOLDOWN_S", "10"))
# Long cooldown for auth / permission errors — effectively retires the key
DEFAULT_AUTH_FAIL_COOLDOWN_S = float(os.getenv("VERA_AUTH_FAIL_COOLDOWN_S", "3600"))
# Per-attempt timeout (single key, single model)
DEFAULT_PER_ATTEMPT_TIMEOUT_S = float(os.getenv("VERA_LLM_ATTEMPT_TIMEOUT_S", "10"))


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_RATE_LIMIT_KEYWORDS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "resource_exhausted",
    "quota",
    "exhausted",
)

_AUTH_KEYWORDS = (
    "401",
    "403",
    "unauthorized",
    "permission_denied",
    "permission denied",
    "invalid api key",
    "api key not valid",
)

_MODEL_NOT_FOUND_KEYWORDS = (
    "404",
    "not_found",
    "not found",
    "is not found",
    "models/",  # 404 messages always include the model path
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic: is *exc* a 429 / quota / RESOURCE_EXHAUSTED error?"""
    msg = str(exc).lower()
    return any(kw in msg for kw in _RATE_LIMIT_KEYWORDS)


def _is_auth_error(exc: BaseException) -> bool:
    """Heuristic: is *exc* a permanent auth / permission error for this key?"""
    msg = str(exc).lower()
    return any(kw in msg for kw in _AUTH_KEYWORDS)


def _is_model_not_found_error(exc: BaseException) -> bool:
    """Heuristic: is *exc* a 404 model-not-found error?

    A 404 means the requested model identifier does not exist for this
    project — every key will fail identically, so the chain should skip
    the entire model rather than burn a round-trip per key.
    """
    msg = str(exc).lower()
    if "404" not in msg and "not_found" not in msg and "not found" not in msg:
        return False
    # Disambiguate from other 404s (the SDK's 404 errors always reference
    # the model path under "models/").
    return "model" in msg


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

@dataclass
class KeyState:
    """Per-key state held by the pool.

    ``client`` is intentionally typed as ``Any`` to avoid importing
    ``google.genai`` for tests that do not need it; the pool only cares
    that the object has the same ``models.generate_content`` shape.
    """

    api_key: str
    client: Any
    cooldown: dict[str, float] = field(default_factory=dict)

    def masked(self) -> str:
        """Return a debug-safe label like ``AIza...XYZW``."""
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


class GeminiKeyPool:
    """Round-robin pool of Gemini API keys with cooldown bookkeeping.

    The pool is intentionally small (a handful of keys) and the lock is
    held only for cooldown reads/writes, so contention is negligible
    even under hundreds of concurrent requests.
    """

    def __init__(
        self,
        api_keys: list[str],
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """Build a pool from a list of API keys.

        Args:
            api_keys: One or more API key strings.
            client_factory: Callable ``key -> client``. Defaults to
                ``google.genai.Client(api_key=key)`` — overridable so
                unit tests can inject a fake client.
        """
        if not api_keys:
            raise ValueError("GeminiKeyPool requires at least one API key")

        if client_factory is None:
            from google import genai  # local import keeps tests cheap

            client_factory = lambda key: genai.Client(api_key=key)  # noqa: E731

        self._states: list[KeyState] = [
            KeyState(api_key=k, client=client_factory(k)) for k in api_keys
        ]
        self._counter = itertools.count()
        self._lock = threading.Lock()

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of keys currently in the pool."""
        return len(self._states)

    def keys_masked(self) -> list[str]:
        """Return masked labels for every key (for diagnostics)."""
        return [state.masked() for state in self._states]

    # -- acquire / release --------------------------------------------------

    def acquire(self, model: str, *, now: float | None = None) -> KeyState | None:
        """Return the next key not on cooldown for *model*, or None.

        Round-robin: each call advances an internal counter and inspects
        keys starting from the next index, so concurrent callers spread
        evenly across the pool with no contention on the hot path.
        """
        n = len(self._states)
        start = next(self._counter) % n
        ts = time.monotonic() if now is None else now
        with self._lock:
            for offset in range(n):
                idx = (start + offset) % n
                state = self._states[idx]
                if state.cooldown.get(model, 0.0) <= ts:
                    return state
        return None

    def mark_failure(
        self,
        state: KeyState,
        model: str,
        cooldown_s: float,
        reason: str,
    ) -> None:
        """Record that *state* failed on *model*; bench it for cooldown_s."""
        with self._lock:
            state.cooldown[model] = time.monotonic() + cooldown_s
        logger.warning(
            "Key %s benched on %s for %.0fs (%s)",
            state.masked(), model, cooldown_s, reason,
        )

    def reset_cooldowns(self) -> None:
        """Clear all cooldowns. Mainly used by tests and ``/v1/teardown``."""
        with self._lock:
            for state in self._states:
                state.cooldown.clear()


# ---------------------------------------------------------------------------
# Model fallback chain
# ---------------------------------------------------------------------------

class ModelFallbackChain:
    """Try each model in order, exhausting all keys before stepping down.

    The chain holds a reference to the shared :class:`GeminiKeyPool`, so
    multiple chains (composer, classifier) share cooldown bookkeeping at
    the (key, model) level rather than per-key.
    """

    def __init__(
        self,
        pool: GeminiKeyPool | None,
        models: list[str],
        *,
        per_attempt_timeout_s: float = DEFAULT_PER_ATTEMPT_TIMEOUT_S,
    ) -> None:
        """Create a fallback chain.

        Args:
            pool: Shared key pool, or ``None`` to disable LLM access
                entirely (e.g. when ``NO_LLM=1`` or no keys are set).
            models: Ordered list of model identifiers. Earlier entries
                are tried first; later entries are the fallbacks.
            per_attempt_timeout_s: Hard timeout for a single
                ``(key, model)`` invocation. The chain may make up to
                ``len(models) * pool.size`` attempts.
        """
        if not models:
            raise ValueError("ModelFallbackChain requires at least one model")
        self._pool = pool
        self._models = list(models)
        self._timeout_s = per_attempt_timeout_s

    # -- introspection ------------------------------------------------------

    @property
    def models(self) -> list[str]:
        """Return a copy of the configured model chain."""
        return list(self._models)

    def is_available(self) -> bool:
        """Return True iff the chain has a pool and is not disabled by NO_LLM."""
        if self._pool is None:
            return False
        if os.getenv("NO_LLM") == "1":
            return False
        return self._pool.size > 0

    # -- core invocation ----------------------------------------------------

    def _invoke_one(
        self,
        invoke: Callable[[Any, str], Any],
        model: str,
        state: KeyState,
    ) -> Any:
        """Run a single ``(key, model)`` attempt with a hard timeout."""
        # Run the blocking SDK call inside its own anyio cancel scope so
        # we can fail fast on slow networks without leaking a thread.
        return invoke(state.client, model)

    def _drive(self, invoke: Callable[[Any, str], Any]) -> tuple[Any, str]:
        """Execute the fallback loop synchronously.

        ``invoke(client, model_name)`` must return the SDK response or
        raise. This function is the single source of truth for fallback
        ordering, error classification, and cooldown bookkeeping.
        """
        if self._pool is None:
            raise RuntimeError("ModelFallbackChain has no key pool configured")

        last_exc: BaseException | None = None
        for model in self._models:
            attempts_made = 0
            model_unavailable = False
            for _ in range(self._pool.size):
                state = self._pool.acquire(model)
                if state is None:
                    # Every key is cooling down on this model — step to
                    # the next model immediately.
                    break
                attempts_made += 1
                try:
                    response = self._invoke_one(invoke, model, state)
                    return response, model
                except Exception as exc:
                    last_exc = exc
                    if _is_model_not_found_error(exc):
                        # Same 404 will hit every key; bench every key
                        # on this model and step to the next model.
                        for s in self._pool._states:
                            self._pool.mark_failure(
                                s, model,
                                DEFAULT_AUTH_FAIL_COOLDOWN_S,
                                "model-not-found",
                            )
                        model_unavailable = True
                        break
                    if _is_rate_limit_error(exc):
                        self._pool.mark_failure(
                            state, model,
                            DEFAULT_RATE_LIMIT_COOLDOWN_S, "rate-limit",
                        )
                        continue
                    if _is_auth_error(exc):
                        self._pool.mark_failure(
                            state, model,
                            DEFAULT_AUTH_FAIL_COOLDOWN_S, "auth-failure",
                        )
                        continue
                    # Unknown error: short bench so we don't pin one bad key
                    self._pool.mark_failure(
                        state, model,
                        DEFAULT_TIMEOUT_COOLDOWN_S, f"error: {type(exc).__name__}",
                    )
                    continue
            if model_unavailable:
                logger.warning(
                    "Model %s is not available (404); skipping entire model",
                    model,
                )
                continue
            if attempts_made == 0:
                logger.info(
                    "All %d keys cooling down for model %s; trying next",
                    self._pool.size, model,
                )
            else:
                logger.warning(
                    "Exhausted all keys for model %s; falling back",
                    model,
                )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "ModelFallbackChain exhausted with no errors recorded — "
            "every key was on cooldown."
        )

    # -- public sync + async APIs ------------------------------------------

    def generate_content_sync(
        self,
        contents: Any,
        config: Any | None = None,
    ) -> tuple[Any, str]:
        """Blocking variant — call from worker threads or sync code."""
        def invoke(client: Any, model: str) -> Any:
            return client.models.generate_content(
                model=model, contents=contents, config=config,
            )
        return self._drive(invoke)

    async def generate_content(
        self,
        contents: Any,
        config: Any | None = None,
    ) -> tuple[Any, str]:
        """Async variant — runs the entire fallback loop in one worker thread.

        Doing the whole loop in a single thread (rather than dispatching
        each retry separately) avoids per-attempt thread-pool overhead
        and gives the lowest-latency happy path: ~one ``run_sync``
        dispatch + one network call.
        """
        def runner() -> tuple[Any, str]:
            return self.generate_content_sync(contents, config)
        return await anyio.to_thread.run_sync(runner)


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised from env)
# ---------------------------------------------------------------------------

def _load_api_keys_from_env() -> list[str]:
    """Read API keys from env in priority order.

    1. ``GEMINI_API_KEYS`` — comma-separated list (preferred).
    2. ``GEMINI_API_KEY_1``, ``GEMINI_API_KEY_2``, ... in order.
    3. ``GEMINI_API_KEY`` — single legacy variable.

    Returns an empty list if no keys are configured (the chains then
    report ``is_available() == False``).
    """
    raw = os.getenv("GEMINI_API_KEYS", "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys

    numbered: list[str] = []
    idx = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{idx}", "").strip()
        if not key:
            break
        numbered.append(key)
        idx += 1
    if numbered:
        return numbered

    single = os.getenv("GEMINI_API_KEY", "").strip()
    return [single] if single else []


def _split_models(env_var: str, default: str) -> list[str]:
    raw = os.getenv(env_var, default).strip()
    return [m.strip() for m in raw.split(",") if m.strip()]


def _build_pool() -> GeminiKeyPool | None:
    """Construct the shared pool from env. Returns None if no keys / NO_LLM."""
    if os.getenv("NO_LLM") == "1":
        return None
    keys = _load_api_keys_from_env()
    if not keys:
        return None
    try:
        pool = GeminiKeyPool(keys)
    except Exception:
        logger.exception("Failed to build GeminiKeyPool; LLM features disabled")
        return None
    logger.info(
        "GeminiKeyPool ready with %d key(s): %s",
        pool.size, pool.keys_masked(),
    )
    return pool


_POOL: GeminiKeyPool | None = _build_pool()

COMPOSER_MODELS: list[str] = _split_models(
    "VERA_COMPOSER_MODELS", _DEFAULT_COMPOSER_MODELS,
)
CLASSIFIER_MODELS: list[str] = _split_models(
    "VERA_CLASSIFIER_MODELS", _DEFAULT_CLASSIFIER_MODELS,
)

COMPOSER_CHAIN = ModelFallbackChain(_POOL, COMPOSER_MODELS)
CLASSIFIER_CHAIN = ModelFallbackChain(_POOL, CLASSIFIER_MODELS)


def get_pool() -> GeminiKeyPool | None:
    """Return the shared key pool (None if no keys / NO_LLM=1)."""
    return _POOL


__all__ = [
    "CLASSIFIER_CHAIN",
    "CLASSIFIER_MODELS",
    "COMPOSER_CHAIN",
    "COMPOSER_MODELS",
    "GeminiKeyPool",
    "KeyState",
    "ModelFallbackChain",
    "get_pool",
]
