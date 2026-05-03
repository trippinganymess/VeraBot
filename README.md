# Vera Bot — magicpin AI Challenge

A merchant-facing WhatsApp assistant for magicpin's "Vera" product, built
to the spec in `requirements/challenge-brief.md` and
`requirements/challenge-testing-brief.md`.

This repo contains:

- `bot.py` — the FastAPI app exposing the 5 judge endpoints + the
  deterministic-rule + Gemini composer (`compose` / `compose_async`).
- `semantic_matcher.py` — 3-stage classifier (regex -> BGE-M3 cosine
  similarity via Hugging Face Inference -> Gemma-3 LLM fallback) that
  routes merchant replies into `auto_reply` / `intent_transition` /
  `hostile` / `wait` / `neither`.
- `conversationTest.py` — local end-to-end conversation harness; drives
  5 multi-turn scenarios against a running bot and scores each composed
  message on the 5 win-criterion dimensions.
- `judge_simulator.py` — magicpin's reference LLM-judge; vendored so we
  can self-test before submission.
- `tests/` — unit, integration, latency, and bot-response tests.
- `dataset/` — the base challenge dataset (categories, merchants,
  customers, triggers).

## Architecture (one-screen summary)

```
                     POST /v1/context, /v1/tick, /v1/reply
                                 |
                                 v
                +-------------------------------+
                |   FastAPI (bot.py)            |
                |   in-memory _context_store    |
                +---------------+---------------+
                                |
                                v
                +-------------------------------+
                |   Deterministic composer      |
                |   - Pydantic validation       |
                |   - Strategy selection        |
                |   - Compulsion lever (LEVER_  |
                |     MAP) + voice prefix       |
                |   - Binary CTA enforcement    |
                |   - Suppression dedup         |
                +---------------+---------------+
                                |
                       Just-in-time facts
                                |
                                v
        +---------------------------------------------+
        |  llm_pool.COMPOSER_CHAIN  (async)           |
        |  GeminiKeyPool round-robin over N keys      |
        |  Per-(key, model) cooldowns on 429 / auth   |
        |    Model fallback chain:                    |
        |    gemini-3.1-flash-lite-preview            |
        |       -> gemini-2.5-flash-lite              |
        |       -> gemma-4-31b                        |
        |  Silent fall-through to deterministic draft |
        +---------------------------------------------+
```

`/v1/reply` adds a 3-stage classifier in front of the composer:

1. Regex fast-path (zero latency)
2. BGE-M3 sentence embeddings (HF Inference API) over curated anchors
3. Load-balanced classifier chain (`llm_pool.CLASSIFIER_CHAIN`):
   `gemma-3-12b-it -> gemma-3-27b-it -> gemma-3-4b-it -> gemma-4-26b`

### Load-balancer hot path

```python
# llm_pool.py
state = pool.acquire(model)              # O(1), lock held microseconds
response = state.client.generate_content(model=model, contents=prompt, ...)
# -> on 429 / quota: pool.mark_failure(state, model, 60s); try next key
# -> if all keys cooled down: step to next model in chain
```

The whole fallback loop runs inside a single `anyio.to_thread.run_sync`
worker, so the happy path is **one thread dispatch + one network call**;
no per-attempt thread-pool churn.

## Endpoints (per `challenge-testing-brief.md` §2)

| Endpoint        | Method | Purpose                                |
| --------------- | ------ | -------------------------------------- |
| `/v1/context`   | POST   | Idempotent context push (per scope)    |
| `/v1/tick`      | POST   | Compose proactive actions (cap = 20)   |
| `/v1/reply`     | POST   | Route a merchant reply to next action  |
| `/v1/healthz`   | GET    | Uptime + per-scope context counts      |
| `/v1/metadata`  | GET    | Team identity + approach description   |
| `/v1/teardown`  | POST   | Wipe all in-memory state (test reset)  |

## Local development

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional credentials in a .env file:
#   GEMINI_API_KEY=...
#   HF_TOKEN=...
# When neither is set, set NO_LLM=1 to use the deterministic composer
# only — useful for offline tests / CI.

uvicorn bot:app --host 0.0.0.0 --port 8080 --reload
```

### Run the test suite

```bash
# Fast / deterministic — no external API calls
NO_LLM=1 python -m unittest discover -s tests
```

### Drive a multi-turn conversation against the running bot

```bash
# Terminal 1: bot
uvicorn bot:app --port 8080

# Terminal 2: conversation harness
python conversationTest.py
# or filter: SCENARIO=S2_studio11_auto_reply python conversationTest.py
```

## Production deployment

The repo ships a multi-stage `Dockerfile` that builds a slim, non-root
runtime image listening on port 8080.

```bash
docker build -t vera-bot:1.0.0 .
docker run --rm -p 8080:8080 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  vera-bot:1.0.0
```

Deploy the image to any container host that gives you a public URL
(Cloud Run, Fly, Render, Railway, Azure Container Apps, ECS, k8s, etc.)
and submit `https://<your-host>` via the magicpin submission portal.

### Production environment variables

| Variable                       | Default                                                     | Notes                                                                                |
| ------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `GEMINI_API_KEYS`              | _unset_                                                     | Comma-separated list of API keys (preferred — feeds the load balancer)               |
| `GEMINI_API_KEY_1` ... `_N`    | _unset_                                                     | Numbered alternative; merged left-to-right                                           |
| `GEMINI_API_KEY`               | _unset_                                                     | Legacy single-key fallback (used only if neither of the above is set)                |
| `HF_TOKEN`                     | _unset_                                                     | Token for the BGE-M3 inference endpoint                                              |
| `VERA_COMPOSER_MODELS`         | `gemini-3.1-flash-lite-preview,gemini-2.5-flash-lite,gemma-4-31b` | Composer model fallback chain                                                        |
| `VERA_CLASSIFIER_MODELS`       | `gemma-3-12b-it,gemma-3-27b-it,gemma-3-4b-it,gemma-4-26b`   | Stage-3 classifier fallback chain                                                    |
| `VERA_RATE_LIMIT_COOLDOWN_S`   | `60`                                                        | Bench a (key, model) for this long after a 429                                       |
| `VERA_TIMEOUT_COOLDOWN_S`      | `10`                                                        | Bench duration for transient errors / timeouts                                       |
| `VERA_AUTH_FAIL_COOLDOWN_S`    | `3600`                                                      | Effectively retires a key with a permanent auth error                                |
| `VERA_LLM_ATTEMPT_TIMEOUT_S`   | `10`                                                        | Hard timeout for one (key, model) attempt                                            |
| `VERA_HOST`                    | `0.0.0.0`                                                   | Bind address                                                                         |
| `VERA_PORT`                    | `8080`                                                      | Bind port                                                                            |
| `NO_LLM`                       | _unset_                                                     | Set to `1` to bypass both LLM chains (deterministic-only)                            |

### Pre-flight checklist (per testing brief §12)

- [x] All 5 endpoints implemented and returning correct schemas
- [x] `/v1/context` is idempotent on `(scope, context_id, version)`
- [x] `/v1/tick` returns within 30 s and caps at 20 actions
- [x] `/v1/reply` returns within 30 s for any conversation
- [x] Bot persists context across calls (in-memory `_context_store`)
- [x] `judge_simulator.py` runnable locally
- [x] Container image runs as non-root, exposes a healthcheck

## License / privacy

The dataset is synthetic. The bot must not transmit payload data outside
LLM provider APIs (per testing brief §11). State is wiped on
`POST /v1/teardown` — the judge harness calls this at the end of a run.
