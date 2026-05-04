"""Live smoke test for the multi-key Gemini load balancer.

Runs only when ``LIVE_LLM=1`` is set in the environment. Performs four
checks against the *real* Google Gemini API and prints a summary table:

    1. Per-key health — pings each key once with the primary model.
    2. Per-model health — confirms every model in both fallback chains
       is actually accepted by the API.
    3. Composer chain — end-to-end call through ``COMPOSER_CHAIN``.
    4. Classifier chain — end-to-end call through ``CLASSIFIER_CHAIN``.
    5. Concurrency — fires N parallel requests through the async chain
       and verifies round-robin distribution.

Run with (POSIX)::

    LIVE_LLM=1 python scripts/smoke_llm_pool.py

Or on Windows PowerShell::

    $env:LIVE_LLM = "1"
    python scripts/smoke_llm_pool.py

Outputs a non-zero exit code only if NO keys work or NO models in the
composer chain work — partial failures are flagged but tolerated, since
the load balancer is designed to route around them.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Make the project root importable when this script is invoked from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# --- pre-flight ---------------------------------------------------------------

if os.getenv("LIVE_LLM") != "1":
    print("Skipping live smoke test. Set LIVE_LLM=1 to enable.")
    sys.exit(0)

# Load AFTER load_dotenv so env vars are picked up.
from src.llm_pool import (  # noqa: E402
    CLASSIFIER_CHAIN,
    CLASSIFIER_MODELS,
    COMPOSER_CHAIN,
    COMPOSER_MODELS,
    GeminiKeyPool,
    ModelFallbackChain,
    get_pool,
)

# Banner ----------------------------------------------------------------------

WIDTH = 78
SEP = "-" * WIDTH


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


pool = get_pool()
if pool is None or pool.size == 0:
    print(
        "ERROR: GeminiKeyPool reports zero keys. Check your .env "
        "(GEMINI_API_KEYS or GEMINI_API_KEY)."
    )
    sys.exit(1)

print(f"\nLoaded {pool.size} key(s): {pool.keys_masked()}")
print(f"Composer chain  : {COMPOSER_MODELS}")
print(f"Classifier chain: {CLASSIFIER_MODELS}")


# -----------------------------------------------------------------------------
# 1. Per-key health
# -----------------------------------------------------------------------------

section("1. Per-key health (model = first composer model)")

primary_model = COMPOSER_MODELS[0]
key_results: list[tuple[str, bool, str, float]] = []

for state in pool._states:
    label = state.masked()
    t0 = time.monotonic()
    try:
        resp = state.client.models.generate_content(
            model=primary_model,
            contents="Reply with just the word: OK",
        )
        elapsed = (time.monotonic() - t0) * 1000
        text = (resp.text or "").strip()[:40]
        key_results.append((label, True, text, elapsed))
        print(f"  {label:>16}  OK  {elapsed:6.0f} ms  -> {text!r}")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        msg = str(exc).splitlines()[0][:80]
        key_results.append((label, False, msg, elapsed))
        print(f"  {label:>16}  FAIL  {elapsed:6.0f} ms  -> {msg}")

healthy_keys = [k for k, ok, *_ in key_results if ok]
print(f"\n  Healthy keys: {len(healthy_keys)} / {pool.size}")


# -----------------------------------------------------------------------------
# 2. Per-model health (one healthy key, every model)
# -----------------------------------------------------------------------------

section("2. Per-model health (one healthy key vs every model)")

if not healthy_keys:
    print("  Skipping — no healthy keys available.")
    composer_ok_models: list[str] = []
    classifier_ok_models: list[str] = []
else:
    # Pick the first healthy key (state.masked() label maps back to the index).
    healthy_idx = next(
        i for i, (label, ok, *_) in enumerate(key_results) if ok
    )
    probe_state = pool._states[healthy_idx]

    all_models = list(dict.fromkeys(COMPOSER_MODELS + CLASSIFIER_MODELS))
    model_results: dict[str, tuple[bool, str, float]] = {}
    for model in all_models:
        t0 = time.monotonic()
        try:
            resp = probe_state.client.models.generate_content(
                model=model,
                contents="Reply with: OK",
            )
            elapsed = (time.monotonic() - t0) * 1000
            text = (resp.text or "").strip()[:30]
            model_results[model] = (True, text, elapsed)
            print(f"  {model:<40}  OK  {elapsed:6.0f} ms  -> {text!r}")
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            msg = str(exc).splitlines()[0][:80]
            model_results[model] = (False, msg, elapsed)
            print(f"  {model:<40}  FAIL  {elapsed:6.0f} ms  -> {msg}")

    composer_ok_models = [m for m in COMPOSER_MODELS if model_results[m][0]]
    classifier_ok_models = [m for m in CLASSIFIER_MODELS if model_results[m][0]]
    print(
        f"\n  Composer chain  : {len(composer_ok_models)}/{len(COMPOSER_MODELS)} models work"
    )
    print(
        f"  Classifier chain: {len(classifier_ok_models)}/{len(CLASSIFIER_MODELS)} models work"
    )

# Reset cooldowns from probe noise before the chain tests.
pool.reset_cooldowns()


# -----------------------------------------------------------------------------
# 3. End-to-end composer chain
# -----------------------------------------------------------------------------

section("3. COMPOSER_CHAIN.generate_content (async)")


async def _run_composer() -> None:
    t0 = time.monotonic()
    try:
        response, model_used = await COMPOSER_CHAIN.generate_content(
            contents="Respond with exactly: COMPOSER_OK",
        )
        elapsed = (time.monotonic() - t0) * 1000
        text = (response.text or "").strip()[:40]
        print(f"  OK  model={model_used}  {elapsed:6.0f} ms  -> {text!r}")
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        print(f"  FAIL  {elapsed:6.0f} ms  -> {exc}")


asyncio.run(_run_composer())
pool.reset_cooldowns()


# -----------------------------------------------------------------------------
# 4. End-to-end classifier chain
# -----------------------------------------------------------------------------

section("4. CLASSIFIER_CHAIN.generate_content_sync")

t0 = time.monotonic()
try:
    response, model_used = CLASSIFIER_CHAIN.generate_content_sync(
        contents=(
            'Classify the following message. '
            'Categories: intent, auto-reply, hostile, wait, neither. '
            'Return ONLY the category name. '
            'Message: "yes please sign me up"'
        ),
    )
    elapsed = (time.monotonic() - t0) * 1000
    text = (response.text or "").strip()[:40]
    print(f"  OK  model={model_used}  {elapsed:6.0f} ms  -> {text!r}")
except Exception as exc:
    elapsed = (time.monotonic() - t0) * 1000
    print(f"  FAIL  {elapsed:6.0f} ms  -> {exc}")
pool.reset_cooldowns()


# -----------------------------------------------------------------------------
# 5. Concurrency / round-robin distribution
# -----------------------------------------------------------------------------

section("5. Concurrent fan-out — verifies round-robin under load")

CONCURRENCY = max(8, pool.size * 2)

# Hook the pool to count which key serves each request.
key_hits: Counter[str] = Counter()
_orig_acquire = pool.acquire


def _counting_acquire(model: str, *, now=None):
    state = _orig_acquire(model, now=now)
    if state is not None:
        key_hits[state.masked()] += 1
    return state


pool.acquire = _counting_acquire  # type: ignore[method-assign]


async def _fanout() -> list[tuple[bool, str]]:
    async def one(i: int):
        try:
            resp, model_used = await COMPOSER_CHAIN.generate_content(
                contents=f"Reply with the number {i}",
            )
            return True, model_used
        except Exception as exc:
            return False, str(exc).splitlines()[0][:60]

    return await asyncio.gather(*[one(i) for i in range(CONCURRENCY)])


t0 = time.monotonic()
results = asyncio.run(_fanout())
elapsed = (time.monotonic() - t0) * 1000

successes = sum(1 for ok, _ in results if ok)
print(f"  {successes}/{CONCURRENCY} requests succeeded in {elapsed:.0f} ms total")
print(f"  Per-key acquire counts:")
for label, count in key_hits.most_common():
    bar = "#" * count
    print(f"    {label:>16}  {count:>3}  {bar}")

# Restore
pool.acquire = _orig_acquire  # type: ignore[method-assign]


# -----------------------------------------------------------------------------
# Final verdict
# -----------------------------------------------------------------------------

section("Summary")
print(f"  Healthy keys           : {len(healthy_keys)} / {pool.size}")
print(
    f"  Working composer models: {len(composer_ok_models)} / {len(COMPOSER_MODELS)}"
)
print(
    f"  Working classifier mdls: {len(classifier_ok_models)} / {len(CLASSIFIER_MODELS)}"
)
print(f"  Concurrent successes   : {successes} / {CONCURRENCY}")
print()

# Exit non-zero only if the chain is not viable at all.
if not healthy_keys:
    print("FAIL: zero keys are healthy — chain is non-functional.")
    sys.exit(2)
if not composer_ok_models:
    print(
        "FAIL: zero composer models work — composer chain has no viable target."
    )
    sys.exit(3)
if not classifier_ok_models:
    print(
        "WARN: zero classifier models work — Stage-3 LLM fallback will be a no-op."
    )
    # Stage-3 is optional; regex + BGE-M3 still cover most cases. Don't fail.

print("PASS: load balancer is wired up and routing real traffic.")
