"""Semantic similarity matcher using a MuRIL-based sentence transformer.

This module provides intent and auto-reply detection by comparing incoming
messages against curated anchor phrases using cosine similarity, rather than
brittle regex patterns.  The underlying model is ``l3cube-pune/hindi-sentence-bert-nli``,
a MuRIL (google/muril-base-cased) model fine-tuned on NLI data for producing
discriminative sentence embeddings across English, Hindi, and transliterated
Indian languages.

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name — based on MuRIL, fine-tuned for sentence similarity
# ---------------------------------------------------------------------------
_MODEL_NAME = "l3cube-pune/hindi-sentence-bert-nli"

# ---------------------------------------------------------------------------
# Similarity thresholds (tuned from empirical testing)
# ---------------------------------------------------------------------------
AUTO_REPLY_THRESHOLD = 0.60
INTENT_TRANSITION_THRESHOLD = 0.60

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

    # -- public API ---------------------------------------------------------

    def is_auto_reply(self, message: str) -> bool:
        """Return True if *message* is semantically similar to auto-reply anchors.

        Falls back to False if the model is not loaded (e.g. during tests).
        """
        self._ensure_loaded()
        if self._model is None or self._auto_reply_embeddings is None:
            return False

        emb = self._model.encode([message], normalize_embeddings=True)
        max_sim = float(np.max(emb @ self._auto_reply_embeddings.T))
        logger.debug("auto-reply similarity for %r: %.3f", message[:60], max_sim)
        return max_sim >= AUTO_REPLY_THRESHOLD

    def is_intent_transition(self, message: str) -> bool:
        """Return True if *message* is semantically similar to intent anchors.

        Falls back to False if the model is not loaded (e.g. during tests).
        """
        self._ensure_loaded()
        if self._model is None or self._intent_embeddings is None:
            return False

        emb = self._model.encode([message], normalize_embeddings=True)
        max_sim = float(np.max(emb @ self._intent_embeddings.T))
        logger.debug("intent similarity for %r: %.3f", message[:60], max_sim)
        return max_sim >= INTENT_TRANSITION_THRESHOLD

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
