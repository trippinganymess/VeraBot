"""Semantic message classifier for the Vera bot.

Three-stage classification pipeline used by ``bot.py`` to route merchant
replies into one of five buckets (``auto_reply``, ``intent_transition``,
``hostile``, ``wait``, ``neither``):

    Stage 1 — Fast-path regex (zero latency, ~0 cost)
    Stage 2 — Cosine similarity over BGE-M3 sentence embeddings
              (``BAAI/bge-m3`` via the Hugging Face Inference API)
    Stage 3 — Gemma-3 LLM fallback when Stage 2 is below threshold

BGE-M3 is multilingual and handles English, Devanagari Hindi, and
transliterated Hinglish well — which matches the real-world WhatsApp
traffic the judge harness simulates. Anchors are pre-computed once at
startup so steady-state classification is a single embedding call plus
4 dot-products.

Set ``NO_LLM=1`` in the environment to disable both Stage 2 and Stage 3
(useful for deterministic tests / CI).
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast-path regex patterns (Stage 1)
# ---------------------------------------------------------------------------

_AUTO_REPLY_REGEX: list[str] = [
    r"auto[- ]?reply",
    r"noreply@",
    r"no-reply@",
    r"do not reply",
]

_INTENT_REGEX: list[str] = [
    r"\b(let'?s do it|sign me up|i want to join|proceed)\b",
    r"\b(shuru karo|join karna|onboard karna|bharti karo)\b",
]

_HOSTILE_REGEX: list[str] = [
    r"\b(stop|unsubscribe|remove me|block|spam|not interested|useless|bakwaas)\b",
    r"\b(band karo|hatao|mat bhejo|nahi chahiye|disturb mat karo)\b",
]

_WAIT_REGEX: list[str] = [
    r"\b(baad mein|thodi der|give me time|not now|not ready|call later|abhi nahi)\b",
    r"\b(busy hoon|free nahi|thoda wait|baat baad mein)\b",
]

# ---------------------------------------------------------------------------
# Model name
# ---------------------------------------------------------------------------
_MODEL_NAME = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# Similarity thresholds
# ---------------------------------------------------------------------------
AUTO_REPLY_THRESHOLD = 0.75
INTENT_TRANSITION_THRESHOLD = 0.65
HOSTILE_THRESHOLD = 0.70
WAIT_THRESHOLD = 0.68

# ---------------------------------------------------------------------------
# Anchor phrases
# ---------------------------------------------------------------------------

AUTO_REPLY_ANCHORS: list[str] = [
    "thank you for contacting us, we will get back to you shortly",
    "this is an automated reply, our team will respond soon",
    "we have received your message and will get back to you",
    "our office is currently closed, please try again later",
    "thank you for your interest, someone will reach out to you",
    "we are out of office right now, will respond when back",
    "this mailbox is not monitored, please do not reply",
    "your ticket has been created, we will respond shortly",
    "thanks for reaching out, expect a response within 24 hours",
    "we are currently unavailable, please leave a message",
    "please expect a delay in our response",
    "for urgent matters please call our helpline",
    "this is an auto-generated confirmation of your message",
    "sampark karne ke liye dhanyawad, hum jald hi jawab denge",
    "aapka message mil gaya hai, hum jald reply karenge",
    "hum jald hi aapse sampark karenge",
    "aapka sandesh mil gaya hai",
    "humne aapka message dekh liya hai",
    "abhi hum available nahi hain, thodi der mein reply milega",
    "office se bahar hoon, baad mein baat karte hain",
    "kripya intezar karein, humari team sampark karegi",
    "baat karne ke liye shukriya, hum jald jawab denge",
    "aapki query receive ho gayi hai, hum jald sampark karenge",
    "swachalit uttar, kripya iska jawab na dein",
    "aapki shikayat darj kar li gayi hai",
    "asuvidha ke liye khed hai, hum jald sampark karenge",
]

INTENT_TRANSITION_ANCHORS: list[str] = [
    "lets do it, what is the next step",
    "I want to join, sign me up please",
    "ok proceed with it, i am ready to start",
    "yes I am interested, please onboard me",
    "sign me up for this, let us begin",
    "I want to join this offer",
    "let us go ahead, what do I need to do",
    "ok I am ready, proceed now",
    "yes do it, I want to start right away",
    "sounds good, count me in",
    "I want this, please register me",
    "let us start the onboarding process",
    "haan karo, shuru karo abhi",
    "join karna hai mujhe, aage batao",
    "haan bilkul, mujhe register karo",
    "shuru karo bhai, main ready hoon",
    "onboarding start karo, main taiyar hoon",
    "haan mujhe join karna hai",
    "lagao na, entry lagao meri",
    "bharti kar do mujhe",
    "aage badho, main interested hoon",
    "haan karna hai, sign up karo",
    "yes karo, mujhe bhi add karo",
    "chaliye shuru karte hain",
]

HOSTILE_ANCHORS: list[str] = [
    # English hostile patterns
    "please stop messaging me, I am not interested",
    "remove me from your list, do not contact me again",
    "this is spam, stop sending me messages",
    "I want to unsubscribe, do not message me",
    "stop bothering me, I do not want this service",
    "this is useless, please do not contact me again",
    "I am reporting this as spam",
    "block this number, do not call again",
    "not interested at all, stop this immediately",
    "leave me alone, I never asked for this",
    # Hindi / Hinglish hostile patterns
    "band karo yeh messages, mujhe nahi chahiye",
    "mujhe mat bhejo, main interested nahi hoon",
    "yeh bakwaas band karo, mujhe disturb mat karo",
    "hatao mujhe is list se, dobara mat bhejo",
    "mujhe block karo, yeh spam hai",
    "nahi chahiye yeh service, please rokiye",
    "bahut ho gaya, ab mat karo contact",
    "pareshan mat karo, bilkul interest nahi",
    "yeh faltu hai, dobara mat likhna",
    "main complaint karunga agar dobara message aaya",
]

WAIT_ANCHORS: list[str] = [
    # English wait patterns
    "I am busy right now, please check back later",
    "not the right time, can we talk later",
    "give me some time to think about it",
    "I will get back to you, just need some time",
    "call me later, I am in a meeting",
    "not ready yet, let us talk next week",
    "busy at the moment, message me later",
    "I need more time to decide, will let you know",
    "check back with me in a few days",
    "remind me later, right now is not a good time",
    # Hindi / Hinglish wait patterns
    "abhi busy hoon, baad mein baat karte hain",
    "abhi time nahi hai, thodi der baad call karo",
    "sochne do mujhe, baad mein batata hoon",
    "meeting mein hoon, baad mein contact karo",
    "thoda time chahiye, main khud call karunga",
    "abhi nahi, kal baat karte hain",
    "free nahi hoon abhi, thodi der mein",
    "baad mein dekh lete hain yeh sab",
    "abhi mat karo, kuch din baad aana",
    "busy schedule hai, next week try karo",
]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D array and return it."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class SemanticMatcher:
    """Lazy-loaded semantic similarity engine for message classification.

    Embeddings are fetched from the Hugging Face Inference API via
    InferenceClient — no model weights are loaded locally.
    Anchor embeddings are pre-computed once at first use and stored as
    normalised numpy arrays for fast cosine-similarity via dot-product.
    """

    def __init__(self) -> None:
        """Create an unloaded matcher; embeddings are computed in _ensure_loaded."""
        self._client: InferenceClient | None = None
        self._auto_reply_embeddings: np.ndarray | None = None
        self._intent_embeddings: np.ndarray | None = None
        self._hostile_embeddings: np.ndarray | None = None
        self._wait_embeddings: np.ndarray | None = None
        self._cache: dict[str, str] = {}

    # -- helpers ------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return a normalised (N, D) embedding matrix for *texts*.

        The HF InferenceClient accepts both a single string and a list of
        strings at runtime; we always pass a list so the result is
        consistently shaped and can be batched.
        """
        assert self._client is not None, "_embed called before _ensure_loaded"
        raw = self._client.feature_extraction(texts, model=_MODEL_NAME)  # type: ignore[arg-type]
        matrix = np.array(raw, dtype=np.float32)
        if matrix.ndim == 3:
            matrix = matrix[:, 0, :]
        return _normalize(matrix)

    # -- lazy init ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Initialise the InferenceClient and pre-compute all anchor embeddings."""
        if self._client is not None:
            return

        if os.getenv("NO_LLM") == "1":
            logger.info("NO_LLM=1 — skipping semantic matcher client init")
            return

        try:
            from huggingface_hub import InferenceClient

            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            logger.info("Initialising HF InferenceClient for model: %s", _MODEL_NAME)
            self._client = InferenceClient(token=hf_token)

            logger.info("Pre-computing anchor embeddings via HF API…")
            self._auto_reply_embeddings = self._embed(AUTO_REPLY_ANCHORS)
            self._intent_embeddings = self._embed(INTENT_TRANSITION_ANCHORS)
            self._hostile_embeddings = self._embed(HOSTILE_ANCHORS)
            self._wait_embeddings = self._embed(WAIT_ANCHORS)

            logger.info(
                "Semantic matcher ready — %d auto-reply, %d intent, %d hostile, %d wait anchors",
                len(AUTO_REPLY_ANCHORS),
                len(INTENT_TRANSITION_ANCHORS),
                len(HOSTILE_ANCHORS),
                len(WAIT_ANCHORS),
            )
        except Exception:
            logger.exception("Failed to initialise HF InferenceClient; falling back to regex")
            self._client = None

    # -- classification -----------------------------------------------------

    def get_intent_type(self, message: str, llm_client: Any = None) -> str:
        """Classify a merchant message into one of five categories.

        Returns one of: "auto_reply", "intent_transition", "hostile",
        "wait", "neither".

        Pipeline:
            Stage 1 — Fast-path regex (zero latency)
            Stage 2 — BGE-M3 semantic similarity via HF InferenceClient
            Stage 3 — Load-balanced Gemma classifier chain
                      (only when Stage 2 is inconclusive)

        The ``llm_client`` argument is accepted for backward compatibility
        but ignored: Stage 3 always goes through the shared
        :data:`llm_pool.CLASSIFIER_CHAIN`, which load-balances across
        every configured ``GEMINI_API_KEY`` and falls back through the
        configured Gemma model chain when keys are exhausted.
        """
        del llm_client  # unused — kept for backward compatibility
        if message in self._cache:
            return self._cache[message]

        # ------------------------------------------------------------------
        # Stage 1: Fast-path Regex
        # ------------------------------------------------------------------
        msg_lower = message.lower()

        if any(re.search(p, msg_lower) for p in _AUTO_REPLY_REGEX):
            return self._cache_and_return(message, "auto_reply")
        if any(re.search(p, message, flags=re.IGNORECASE) for p in _INTENT_REGEX):
            return self._cache_and_return(message, "intent_transition")
        if any(re.search(p, msg_lower) for p in _HOSTILE_REGEX):
            return self._cache_and_return(message, "hostile")
        if any(re.search(p, msg_lower) for p in _WAIT_REGEX):
            return self._cache_and_return(message, "wait")

        # ------------------------------------------------------------------
        # Stage 2: Semantic Similarity (BGE-M3 via HF InferenceClient)
        # ------------------------------------------------------------------
        self._ensure_loaded()
        if (
            self._client is None
            or self._auto_reply_embeddings is None
            or self._intent_embeddings is None
            or self._hostile_embeddings is None
            or self._wait_embeddings is None
        ):
            return "neither"

        emb = self._embed([message])  # shape (1, D), already normalised

        auto_sim    = float(np.max(emb @ self._auto_reply_embeddings.T))
        intent_sim  = float(np.max(emb @ self._intent_embeddings.T))
        hostile_sim = float(np.max(emb @ self._hostile_embeddings.T))
        wait_sim    = float(np.max(emb @ self._wait_embeddings.T))

        logger.debug(
            "Scores for %r: auto=%.3f intent=%.3f hostile=%.3f wait=%.3f",
            message[:60], auto_sim, intent_sim, hostile_sim, wait_sim,
        )

        # Map each category to (score, threshold) then pick the highest
        # scoring category that clears its threshold.
        candidates: dict[str, tuple[float, float]] = {
            "auto_reply":        (auto_sim,    AUTO_REPLY_THRESHOLD),
            "intent_transition": (intent_sim,  INTENT_TRANSITION_THRESHOLD),
            "hostile":           (hostile_sim, HOSTILE_THRESHOLD),
            "wait":              (wait_sim,    WAIT_THRESHOLD),
        }

        cleared = {
            label: score
            for label, (score, threshold) in candidates.items()
            if score >= threshold
        }

        if cleared:
            # Highest score wins when multiple thresholds are cleared
            ans = max(cleared, key=lambda label: cleared[label])
            return self._cache_and_return(message, ans)

        # ------------------------------------------------------------------
        # Stage 3: LLM fallback — only fires when BGE-M3 is inconclusive
        # ------------------------------------------------------------------
        logger.warning(
            "BGE-M3 below threshold for %r — "
            "auto=%.3f intent=%.3f hostile=%.3f wait=%.3f; hitting LLM fallback",
            message[:80], auto_sim, intent_sim, hostile_sim, wait_sim,
        )

        ans = "neither"
        # Local import to avoid circular import at module load time.
        from llm_pool import CLASSIFIER_CHAIN

        if CLASSIFIER_CHAIN.is_available():
            prompt = f'''Classify the following merchant message into EXACTLY ONE of \
these categories:
- intent (Explicitly expressing interest, agreeing to proceed, or asking to sign up/join)
- auto-reply (Automated out-of-office, unmonitored mailbox, or automated ticket response)
- hostile (Merchant is annoyed, wants to stop receiving messages, asks to unsubscribe)
- wait (Merchant needs time, is busy, asks to be contacted later)
- neither (General questions, providing context, or anything else)

Rules:
1. Return ONLY the exact category name. No other text.
2. No guessing. If unsure, output neither.

Message: "{message}"'''
            try:
                from google.genai import types

                response, model_used = CLASSIFIER_CHAIN.generate_content_sync(
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                logger.debug("Classifier used model %s", model_used)
                if response.text:
                    res = response.text.strip().lower()
                    if "auto-reply" in res:
                        ans = "auto_reply"
                    elif "intent" in res:
                        ans = "intent_transition"
                    elif "hostile" in res:
                        ans = "hostile"
                    elif "wait" in res:
                        ans = "wait"
            except Exception as exc:
                logger.warning(
                    "Classifier chain exhausted across all keys/models: %s", exc,
                )

        return self._cache_and_return(message, ans)

    def _cache_and_return(self, message: str, ans: str) -> str:
        """Store result in cache and return it."""
        if len(self._cache) > 1000:
            self._cache.clear()
        self._cache[message] = ans
        return ans

    # -- convenience wrappers -----------------------------------------------

    def is_auto_reply(self, message: str, llm_client: Any = None) -> bool:
        """Return True if the message is classified as a canned auto-reply."""
        return self.get_intent_type(message, llm_client) == "auto_reply"

    def is_intent_transition(self, message: str, llm_client: Any = None) -> bool:
        """Return True if the merchant explicitly signalled intent to proceed."""
        return self.get_intent_type(message, llm_client) == "intent_transition"

    def is_hostile(self, message: str, llm_client: Any = None) -> bool:
        """Return True if the merchant is hostile / wants to stop receiving messages."""
        return self.get_intent_type(message, llm_client) == "hostile"

    def is_wait(self, message: str, llm_client: Any = None) -> bool:
        """Return True if the merchant is asking to be contacted later."""
        return self.get_intent_type(message, llm_client) == "wait"

    def classify(self, message: str) -> dict[str, float]:
        """Return all four similarity scores for diagnostics.

        Returns a dict with keys: ``auto_reply``, ``intent``, ``hostile``,
        ``wait``. All values are in [0, 1]. Returns zeros if the inference
        client (or any of the anchor embedding matrices) is unavailable —
        for example when ``NO_LLM=1`` is set or the HF API call failed.
        """
        self._ensure_loaded()
        if (
            self._client is None
            or self._auto_reply_embeddings is None
            or self._intent_embeddings is None
            or self._hostile_embeddings is None
            or self._wait_embeddings is None
        ):
            return {"auto_reply": 0.0, "intent": 0.0, "hostile": 0.0, "wait": 0.0}

        emb = self._embed([message])
        return {
            "auto_reply": float(np.max(emb @ self._auto_reply_embeddings.T)),
            "intent": float(np.max(emb @ self._intent_embeddings.T)),
            "hostile": float(np.max(emb @ self._hostile_embeddings.T)),
            "wait": float(np.max(emb @ self._wait_embeddings.T)),
        }


# Module-level singleton — lazy, thread-safe via GIL
semantic_matcher = SemanticMatcher()