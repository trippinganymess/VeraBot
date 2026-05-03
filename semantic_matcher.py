"""Semantic similarity matcher using the BGE-M3 sentence transformer.

This module provides intent and auto-reply detection by comparing incoming
messages against curated anchor phrases using cosine similarity, rather than
brittle regex patterns.  The underlying model is ``BAAI/bge-m3``,
a multilingual model with excellent support for English, Hindi, and transliterated
Indian languages, producing highly discriminative sentence embeddings.

Usage::

    from semantic_matcher import semantic_matcher
    is_auto = semantic_matcher.is_auto_reply("Thank you for contacting us")
    is_intent = semantic_matcher.is_intent_transition("haan karo shuru karo")
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Sequence

import numpy as np
import re

logger = logging.getLogger(__name__)

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

# ---------------------------------------------------------------------------
# Model name — BAAI/bge-m3 for multilingual sentence similarity
# ---------------------------------------------------------------------------
_MODEL_NAME = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# Similarity thresholds (tuned from empirical testing)
# ---------------------------------------------------------------------------
AUTO_REPLY_THRESHOLD = 0.75
INTENT_TRANSITION_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Anchor phrases — the semantic "centers" for each category
# ---------------------------------------------------------------------------

AUTO_REPLY_ANCHORS: list[str] = [
    # English auto-reply patterns
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
    # Hindi / Hinglish auto-reply patterns
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
    # English intent-to-proceed patterns
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
    # Hindi / Hinglish intent patterns
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


class SemanticMatcher:
    """Lazy-loaded semantic similarity engine for message classification.

    The model is loaded on first use and cached for the lifetime of the
    process.  Anchor embeddings are pre-computed once and stored as
    normalised numpy arrays for fast cosine-similarity via dot-product.
    """

    def __init__(self) -> None:
        self._model = None
        self._auto_reply_embeddings: np.ndarray | None = None
        self._intent_embeddings: np.ndarray | None = None
        self._cache: dict[str, str] = {}

    # -- lazy init ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the sentence-transformer model and pre-compute anchor embeddings."""
        if self._model is not None:
            return

        # Skip model loading if NO_LLM is set (for fast deterministic tests)
        if os.getenv("NO_LLM") == "1":
            logger.info("NO_LLM=1 — skipping semantic matcher model load")
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading semantic matcher model: %s", _MODEL_NAME)
            self._model = SentenceTransformer(_MODEL_NAME)

            self._auto_reply_embeddings = self._model.encode(
                AUTO_REPLY_ANCHORS, normalize_embeddings=True
            )
            self._intent_embeddings = self._model.encode(
                INTENT_TRANSITION_ANCHORS, normalize_embeddings=True
            )
            logger.info(
                "Semantic matcher ready — %d auto-reply anchors, %d intent anchors",
                len(AUTO_REPLY_ANCHORS),
                len(INTENT_TRANSITION_ANCHORS),
            )
        except Exception:
            logger.exception("Failed to load semantic matcher; falling back to regex")
            self._model = None

    def get_intent_type(self, message: str, llm_client=None) -> Literal["auto_reply", "intent_transition", "none"]:
        """Classify message as auto-reply, intent-transition, or none.
        
        Strict pipeline: Regex -> BGE-M3 -> Gemma-3 LLM.
        Moves to the next stage only if the previous yields no clear classification.
        """
        if message in self._cache:
            return self._cache[message]

        # Stage 1: Fast-path Regex
        msg_lower = message.lower()
        if any(re.search(p, msg_lower) for p in _AUTO_REPLY_REGEX):
            ans = "auto_reply"
            self._cache[message] = ans
            return ans
        if any(re.search(p, message, flags=re.IGNORECASE) for p in _INTENT_REGEX):
            ans = "intent_transition"
            self._cache[message] = ans
            return ans

        # Stage 2: Semantic Similarity (BGE-M3)
        self._ensure_loaded()
        if self._model is None or self._auto_reply_embeddings is None or self._intent_embeddings is None:
            return "none"

        emb = self._model.encode([message], normalize_embeddings=True)
        auto_sim = float(np.max(emb @ self._auto_reply_embeddings.T))
        intent_sim = float(np.max(emb @ self._intent_embeddings.T))
        
        logger.debug("Classification scores for %r: auto=%.3f, intent=%.3f", message[:60], auto_sim, intent_sim)

        is_auto = auto_sim >= AUTO_REPLY_THRESHOLD
        is_intent = intent_sim >= INTENT_TRANSITION_THRESHOLD

        ans = "none"
        if is_auto and is_intent:
            ans = "auto_reply" if auto_sim >= intent_sim else "intent_transition"
        elif is_auto:
            ans = "auto_reply"
        elif is_intent:
            ans = "intent_transition"
        else:
            # Neither met the threshold. The user requested LLM fallback with temperature=0.
            # No guessing allowed. 
            if llm_client is not None and os.getenv("NO_LLM") != "1":
                prompt = f'''Classify the following merchant message into EXACTLY ONE of these categories:
- intent (Explicitly expressing interest, agreeing to proceed, or asking to sign up/join the offer)
- auto-reply (An automated out-of-office, mailbox unmonitored, or automated ticket response)
- neither (General chatter, questions about other topics, hostility, or anything else)

Rules:
1. Return ONLY the exact category name ("intent", "auto-reply", or "neither"). No other text.
2. No guessing. If unsure, output neither.

Message: "{message}"'''
                try:
                    from google.genai import types
                    response = llm_client.models.generate_content(
                        model="gemma-3-12b-it", # Strictly using Gemma 3 12B as requested
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.0)
                    )
                    if response.text:
                        res = response.text.strip().lower()
                        if "auto-reply" in res:
                            ans = "auto_reply"
                        elif "intent" in res:
                            ans = "intent_transition"
                except Exception as e:
                    logger.warning("LLM fallback classification failed: %s", e)

        # Cache to prevent double evaluation (especially LLM calls) when bot.py calls both is_auto_reply and is_intent_transition
        if len(self._cache) > 1000:
            self._cache.clear()
        self._cache[message] = ans
        return ans

    def is_auto_reply(self, message: str, llm_client=None) -> bool:
        """Return True if message is classified as auto-reply."""
        return self.get_intent_type(message, llm_client) == "auto_reply"

    def is_intent_transition(self, message: str, llm_client=None) -> bool:
        """Return True if message is classified as intent-transition."""
        return self.get_intent_type(message, llm_client) == "intent_transition"

    def classify(self, message: str) -> tuple[float, float]:
        """Return (auto_reply_score, intent_score) for diagnostics.

        Both values are in [0, 1].  Returns (0.0, 0.0) if the model
        is unavailable.
        """
        self._ensure_loaded()
        if self._model is None:
            return 0.0, 0.0

        emb = self._model.encode([message], normalize_embeddings=True)
        auto_sim = float(np.max(emb @ self._auto_reply_embeddings.T))
        intent_sim = float(np.max(emb @ self._intent_embeddings.T))
        return auto_sim, intent_sim


# Module-level singleton — lazy, thread-safe via GIL
semantic_matcher = SemanticMatcher()
