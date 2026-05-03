"""
Real-time conversation tester for the Vera bot.

Talks to the bot's FastAPI endpoints, drives 5-6 turn conversations
across multiple merchant/trigger scenarios, saves every turn (including
CTAs) to a JSONL file, then scores each conversation against the
5 win criteria from the challenge brief.

Usage:
    # Make sure bot is running: uvicorn bot:app --port 8080
    python test_conversations.py

    # Run against a different host:
    BOT_URL=http://localhost:9000 python test_conversations.py

Output:
    conversations.jsonl  — every turn, CTA, rationale, and score
    summary.txt          — per-scenario win-criterion breakdown
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_URL = os.getenv("BOT_URL", "http://localhost:8080")
OUTPUT_FILE = "conversations.jsonl"
SUMMARY_FILE = "summary.txt"
TIMEOUT = 30  # seconds per HTTP call

# ---------------------------------------------------------------------------
# Win-criterion scorer
# ---------------------------------------------------------------------------

# Anti-patterns from §11 of the brief — penalise if present
_ANTI_PATTERNS = [
    (r"\bflat\s*\d+%\s*off\b", "generic_discount"),
    (r"\b(amazing|best deal|hurry|limited time)\b", "promo_tone"),
    (r"reply\s+(yes|no|maybe)\s+(for|to)\s+.+,\s+reply\s+(yes|no|maybe)", "multi_cta"),
    (r"\bi hope you('re| are) doing well\b", "long_preamble"),
    (r"\bi'?m reaching out today\b", "long_preamble"),
    (r"\bas vera\b", "re_introduction"),
]

# Compulsion levers from §10 of the brief
_LEVER_SIGNALS = {
    "specificity":           [r"\d[\d,]+", r"₹\s?\d+", r"\d+%", r"p\.\d+", r"\bjida\b", r"\btrial\b"],
    "loss_aversion":         [r"miss(ing|ed)", r"gap\b", r"below\b", r"before.*close", r"losing"],
    "social_proof":          [r"peer", r"others in your area", r"locality", r"similar\s+\w+\s+in", r"\d+\s+dentist"],
    "effort_externalization":["draft", r"just say yes", r"i can set", r"ready for you", r"5.min"],
    "curiosity":             [r"want to see", r"want me to", r"shall i", r"interested\?"],
    "binary_cta":            [r"\byes\b.*\bstop\b|\bstop\b.*\byes\b"],
}

# Category-voice signals
_VOICE_SIGNALS = {
    "dentists":    ["fluoride", "caries", "recall", "clinical", "jida", "dci", "ida", "scaling", "prophylaxis"],
    "salons":      ["haircut", "blowout", "color", "treatment", "styling", "keratin"],
    "restaurants": ["cover", "footfall", "reservation", "menu", "kitchen", "rush"],
    "gyms":        ["member", "session", "trainer", "workout", "class", "fitness"],
    "pharmacies":  ["compliance", "refill", "prescription", "stock", "regulation"],
}

# Promo-tone taboo words per category
_PROMO_TABOOS = {
    "dentists":  ["guaranteed", "cure", "amazing", "best deal", "hurry"],
    "salons":    ["guaranteed", "amazing"],
    "gyms":      ["guaranteed", "amazing"],
    "pharmacies":["guaranteed", "amazing"],
    "restaurants":[],
}


def score_message(body: str, cta: str, category_slug: str,
                  trigger_kind: str, merchant_name: str) -> dict[str, int]:
    """
    Score a single composed message against the 5 win criteria.
    Returns dict with keys: specificity, category_fit, merchant_fit,
    trigger_relevance, engagement_compulsion, total (all 0-10, total 0-50).
    """
    body_lower = body.lower()
    scores: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 1. Specificity — concrete numbers, dates, citations
    # ------------------------------------------------------------------
    spec = 0
    number_hits = len(re.findall(r"\d[\d,\.]+", body))
    spec += min(number_hits * 2, 6)
    if re.search(r"₹\s?\d+", body):
        spec += 2
    if re.search(r"(jida|dci|ida|pubmed|dental tribune|jida oct)", body_lower):
        spec += 2
    if re.search(r"p\.\d+|page\s+\d+", body_lower):
        spec += 1
    # Penalise generic discount language
    if re.search(r"\bflat\s*\d+%\s*off\b", body_lower):
        spec -= 3
    scores["specificity"] = max(0, min(10, spec))

    # ------------------------------------------------------------------
    # 2. Category fit — voice, vocabulary, taboo avoidance
    # ------------------------------------------------------------------
    cat = 5  # baseline
    voice_words = _VOICE_SIGNALS.get(category_slug, [])
    hits = sum(1 for w in voice_words if w in body_lower)
    cat += min(hits * 2, 4)
    taboos = _PROMO_TABOOS.get(category_slug, [])
    taboo_hits = sum(1 for t in taboos if t in body_lower)
    cat -= taboo_hits * 3
    # Check voice prefix is present for known categories
    voice_prefixes = {
        "dentists": "clinical note",
        "salons": "quick tip",
        "restaurants": "quick ops note",
        "gyms": "coach",
        "pharmacies": "compliance note",
    }
    if voice_prefixes.get(category_slug, "") in body_lower:
        cat += 1
    scores["category_fit"] = max(0, min(10, cat))

    # ------------------------------------------------------------------
    # 3. Merchant fit — personalization, language pref
    # ------------------------------------------------------------------
    merch = 3  # baseline
    # Merchant name mentioned
    if merchant_name.lower().split()[0] in body_lower:
        merch += 2
    # Hindi-English mix (for hi-en mix merchants)
    has_hindi = bool(re.search(r"\b(ji|hai|karo|aap|hoon|mein|ka|ki|ke|baat|kya)\b", body_lower))
    has_english = bool(re.search(r"\b(hi|hello|yes|ready|plan|profile|update)\b", body_lower))
    if has_hindi and has_english:
        merch += 3
    # Penalise re-introduction
    if re.search(r"\bi'?m vera\b|\bthis is vera\b", body_lower):
        merch -= 2
    # Penalise long preamble
    if re.search(r"i hope you('re| are) doing well", body_lower):
        merch -= 2
    scores["merchant_fit"] = max(0, min(10, merch))

    # ------------------------------------------------------------------
    # 4. Trigger relevance — message communicates WHY NOW
    # ------------------------------------------------------------------
    trig = 3  # baseline
    kind_signals = {
        "research_digest":    ["digest", "jida", "research", "trial", "study", "issue"],
        "perf_dip":           ["drop", "dip", "missed", "below", "calls", "views"],
        "perf_spike":         ["spike", "surge", "up", "increase", "peak"],
        "recall_due":         ["recall", "visit", "months", "cleaning", "slot", "appointment"],
        "competitor_opened":  ["competitor", "nearby", "new", "opened"],
        "milestone_reached":  ["milestone", "reviews", "crossed", "congrat"],
        "dormant_with_vera":  ["back", "check in", "missed you", "been a while"],
        "winback":            ["back", "return", "lapsed", "reconnect"],
        "renewal_due":        ["renewal", "expir", "days remaining", "subscription"],
        "festival_upcoming":  ["festival", "diwali", "holi", "eid", "navratri"],
        "review_theme_emerged":["review", "feedback", "patients say", "mentions"],
        "reply":              [],  # catch-all for /v1/reply routing
    }
    # Substring match for compound kinds like "research_digest_release"
    matched_signals: list[str] = []
    for kind_key, signals in kind_signals.items():
        if kind_key in trigger_kind:
            matched_signals = signals
            break
    hits = sum(1 for s in matched_signals if s in body_lower)
    trig += min(hits * 2, 6)
    scores["trigger_relevance"] = max(0, min(10, trig))

    # ------------------------------------------------------------------
    # 5. Engagement compulsion — levers, CTA shape
    # ------------------------------------------------------------------
    comp = 2  # baseline
    for lever, patterns in _LEVER_SIGNALS.items():
        if any(re.search(p, body_lower) for p in patterns):
            comp += 1
    # Binary CTA ends correctly
    if cta == "yes_no" and re.search(r"\b(yes|stop)\b\s*\.?\s*$", body_lower):
        comp += 2
    # Open-ended question present
    if re.search(r"\?", body):
        comp += 1
    # Anti-patterns penalty
    for pattern, _ in _ANTI_PATTERNS:
        if re.search(pattern, body_lower):
            comp -= 2
    # Multiple CTAs penalty
    yes_count = len(re.findall(r"\byes\b", body_lower))
    stop_count = len(re.findall(r"\bstop\b", body_lower))
    if yes_count > 1 or stop_count > 1:
        comp -= 2
    scores["engagement_compulsion"] = max(0, min(10, comp))

    scores["total"] = sum(v for k, v in scores.items() if k != "total")
    return scores


# ---------------------------------------------------------------------------
# Test scenarios (from brief §9, Appendix A, Appendix B)
# ---------------------------------------------------------------------------

@dataclass
class MerchantReply:
    """A scripted merchant reply for a given turn."""
    body: str
    description: str   # what this reply is testing

@dataclass
class Scenario:
    """One end-to-end conversation scenario."""
    name: str
    description: str
    category: dict[str, Any]
    merchant: dict[str, Any]
    trigger: dict[str, Any]
    customer: dict[str, Any] | None
    merchant_replies: list[MerchantReply]   # 4-5 replies → 5-6 total turns
    expected_win_criteria: dict[str, int]   # minimum expected scores


SCENARIOS: list[Scenario] = [

    # ------------------------------------------------------------------
    # S1 — Dr. Meera: research digest, genuine engagement, 6 turns
    # ------------------------------------------------------------------
    Scenario(
        name="S1_drmeera_research_digest",
        description="Dentist merchant, research digest trigger, genuine curiosity → proceeds",
        category={
            "slug": "dentists",
            "display_name": "Dental Clinics",
            "peer_stats": {
                "scope": "South Delhi solo practices",
                "avg_rating": 4.4,
                "avg_review_count": 62,
                "avg_ctr": 0.030,
                "avg_views_30d": 3200,
                "avg_calls_30d": 28,
            },
            "offer_catalog": [
                {"title": "Dental Cleaning @ ₹299"},
                {"title": "Free Consultation"},
                {"title": "Teeth Whitening @ ₹1499"},
            ],
            "digest": [
                {
                    "title": "3-mo fluoride recall cuts caries recurrence 38% better than 6-mo",
                    "source": "JIDA Oct 2026, p.14",
                    "trial_n": 2100,
                    "patient_segment": "high-risk adults",
                }
            ],
        },
        merchant={
            "merchant_id": "m_001_drmeera",
            "category_slug": "dentists",
            "identity": {
                "name": "Dr. Meera's Dental Clinic",
                "owner_first_name": "Meera",
                "languages": ["en", "hi"],
            },
            "performance": {
                "window_days": 30,
                "views": 2410,
                "calls": 18,
                "directions": 45,
                "ctr": 0.021,
                "leads": 12,
                "delta_7d": {"views": -0.05, "calls": -0.12},
            },
        },
        trigger={
            "id": "trg_research_digest_dentists_w17",
            "scope": "merchant",
            "kind": "research_digest",
            "source": "external",
            "merchant_id": "m_001_drmeera",
            "payload": {
                "category": "dentists",
                "top_item": {
                    "title": "3-mo fluoride recall cuts caries recurrence 38% better than 6-mo",
                    "source": "JIDA Oct 2026, p.14",
                    "trial_n": 2100,
                    "patient_segment": "high-risk adults",
                },
            },
            "urgency": 2,
            "suppression_key": "research:dentists:2026-W17",
            "expires_at": "2026-05-10T00:00:00Z",
        },
        customer=None,
        merchant_replies=[
            MerchantReply("Interesting! Tell me more about the fluoride study.", "genuine curiosity — open question"),
            MerchantReply("What does 38% better mean exactly? Which patients?", "drilling deeper — high-risk adults"),
            MerchantReply("Ok that makes sense. Can you draft that patient WhatsApp?", "intent to act — effort externalization"),
            MerchantReply("Looks good. Yes, send it.", "explicit YES intent — should onboard"),
            MerchantReply("Done. Also, what are other dentists in Lajpat Nagar doing differently?", "curiosity — social proof ask"),
        ],
        expected_win_criteria={"specificity": 6, "category_fit": 6, "merchant_fit": 5, "trigger_relevance": 6, "engagement_compulsion": 5},
    ),

    # ------------------------------------------------------------------
    # S2 — Studio11 Salon: auto-reply detection, graceful exit
    # ------------------------------------------------------------------
    Scenario(
        name="S2_studio11_auto_reply",
        description="Salon merchant, perf_dip trigger, merchant sends auto-replies → exit",
        category={
            "slug": "salons",
            "display_name": "Salons & Spas",
            "peer_stats": {
                "avg_rating": 4.3,
                "avg_review_count": 48,
                "avg_ctr": 0.028,
                "avg_views_30d": 2800,
            },
            "offer_catalog": [
                {"title": "Haircut @ ₹99"},
                {"title": "Keratin Treatment @ ₹999"},
                {"title": "Full Body Waxing @ ₹349"},
            ],
        },
        merchant={
            "merchant_id": "m_002_studio11",
            "category_slug": "salons",
            "identity": {
                "name": "Studio11 Family Salon",
                "owner_first_name": "Priya",
                "languages": ["en", "hi"],
            },
            "performance": {
                "window_days": 30,
                "views": 1850,
                "calls": 8,
                "directions": 22,
                "ctr": 0.018,
                "leads": 6,
                "delta_7d": {"views": -0.22, "calls": -0.40},
            },
        },
        trigger={
            "id": "trg_perf_dip_studio11",
            "scope": "merchant",
            "kind": "perf_dip",
            "source": "internal",
            "merchant_id": "m_002_studio11",
            "payload": {
                "context_version": 1,
                "metric": "calls",
                "drop_pct": 40,
                "window_days": 7,
            },
            "urgency": 3,
            "suppression_key": "perf_dip:m_002_studio11:2026-W17",
            "expires_at": "2026-05-07T00:00:00Z",
        },
        customer=None,
        merchant_replies=[
            MerchantReply(
                "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein aur sujhaav hamari team tak pahuncha deti hoon.",
                "first auto-reply — should detect and try once more",
            ),
            MerchantReply(
                "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein aur sujhaav hamari team tak pahuncha deti hoon.",
                "same auto-reply again — should exit",
            ),
        ],
        expected_win_criteria={"specificity": 4, "category_fit": 5, "merchant_fit": 4, "trigger_relevance": 5, "engagement_compulsion": 4},
    ),

    # ------------------------------------------------------------------
    # S3 — Pizza Junction: competitor opened, hostile then wait
    # ------------------------------------------------------------------
    Scenario(
        name="S3_pizza_competitor_hostile_wait",
        description="Restaurant, competitor_opened trigger, hostile then softens to wait",
        category={
            "slug": "restaurants",
            "display_name": "Restaurants",
            "peer_stats": {
                "avg_rating": 4.1,
                "avg_review_count": 110,
                "avg_ctr": 0.025,
                "avg_views_30d": 4500,
            },
            "offer_catalog": [
                {"title": "Lunch Combo @ ₹149"},
                {"title": "Family Meal Deal @ ₹499"},
            ],
        },
        merchant={
            "merchant_id": "m_003_pizza_junction",
            "category_slug": "restaurants",
            "identity": {
                "name": "Pizza Junction",
                "owner_first_name": "Rahul",
                "languages": ["en"],
            },
            "performance": {
                "window_days": 30,
                "views": 3900,
                "calls": 31,
                "directions": 88,
                "ctr": 0.024,
                "leads": 20,
                "delta_7d": {"views": 0.02, "calls": -0.08},
            },
        },
        trigger={
            "id": "trg_competitor_opened_pizza",
            "scope": "merchant",
            "kind": "competitor_opened",
            "source": "external",
            "merchant_id": "m_003_pizza_junction",
            "payload": {
                "competitor_name": "Slice Republic",
                "distance_km": 1.3,
                "rating": 4.2,
                "review_count": 5,
            },
            "urgency": 3,
            "suppression_key": "competitor:m_003:2026-W17",
            "expires_at": "2026-05-10T00:00:00Z",
        },
        customer=None,
        merchant_replies=[
            MerchantReply("Stop sending me these messages. Not interested.", "hostile — should exit cleanly"),
        ],
        expected_win_criteria={"specificity": 4, "category_fit": 4, "merchant_fit": 4, "trigger_relevance": 5, "engagement_compulsion": 4},
    ),

    # ------------------------------------------------------------------
    # S4 — Dr. Meera: customer recall (Priya), 6 turns, books slot
    # ------------------------------------------------------------------
    Scenario(
        name="S4_drmeera_recall_priya",
        description="Customer-facing recall trigger — Priya books slot via merchant_on_behalf",
        category={
            "slug": "dentists",
            "display_name": "Dental Clinics",
            "peer_stats": {"avg_rating": 4.4, "avg_ctr": 0.030},
            "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}],
        },
        merchant={
            "merchant_id": "m_001_drmeera",
            "category_slug": "dentists",
            "identity": {
                "name": "Dr. Meera's Dental Clinic",
                "owner_first_name": "Meera",
                "languages": ["en", "hi"],
            },
            "performance": {"ctr": 0.021, "views": 2410, "calls": 18},
        },
        trigger={
            "id": "trg_recall_priya",
            "scope": "customer",
            "kind": "recall_due",
            "source": "internal",
            "merchant_id": "m_001_drmeera",
            "customer_id": "c_001_priya",
            "payload": {
                "last_visit": "2025-11-04",
                "due_date": "2026-05-04",
                "services": ["cleaning"],
            },
            "urgency": 3,
            "suppression_key": "recall:c_001_priya:2026-05",
            "expires_at": "2026-05-15T00:00:00Z",
        },
        customer={
            "customer_id": "c_001_priya",
            "merchant_id": "m_001_drmeera",
            "identity": {
                "name": "Priya",
                "language_pref": "hi-en mix",
            },
        },
        merchant_replies=[
            MerchantReply("Hi, kaun bol raha hai?", "customer confusion — who is this?"),
            MerchantReply("Oh Dr. Meera clinic! Haan cleaning karani hai.", "positive response, genuine intent"),
            MerchantReply("Wednesday 6pm theek rahega.", "slot selection"),
            MerchantReply("Ok confirmed. ₹299 hi lagega na?", "price confirmation"),
            MerchantReply("Theek hai. Thanks!", "closure — conversation should wrap"),
        ],
        expected_win_criteria={"specificity": 5, "category_fit": 5, "merchant_fit": 6, "trigger_relevance": 6, "engagement_compulsion": 5},
    ),

    # ------------------------------------------------------------------
    # S5 — Gym merchant: milestone reached, wait then proceeds
    # ------------------------------------------------------------------
    Scenario(
        name="S5_gym_milestone_wait_proceed",
        description="Gym hits 100 reviews milestone, merchant asks for time then proceeds",
        category={
            "slug": "gyms",
            "display_name": "Gyms & Fitness",
            "peer_stats": {
                "avg_rating": 4.2,
                "avg_review_count": 85,
                "avg_ctr": 0.022,
                "avg_views_30d": 3100,
            },
            "offer_catalog": [
                {"title": "Monthly Membership @ ₹999"},
                {"title": "Personal Training Session @ ₹499"},
            ],
        },
        merchant={
            "merchant_id": "m_004_fitzone",
            "category_slug": "gyms",
            "identity": {
                "name": "FitZone Gym",
                "owner_first_name": "Vikram",
                "languages": ["en", "hi"],
            },
            "performance": {
                "window_days": 30,
                "views": 2900,
                "calls": 22,
                "ctr": 0.023,
                "leads": 15,
                "delta_7d": {"views": 0.15, "calls": 0.10},
            },
        },
        trigger={
            "id": "trg_milestone_fitzone",
            "scope": "merchant",
            "kind": "milestone_reached",
            "source": "internal",
            "merchant_id": "m_004_fitzone",
            "payload": {
                "milestone": "100_reviews",
                "current_count": 102,
                "avg_rating": 4.3,
            },
            "urgency": 2,
            "suppression_key": "milestone:m_004:100reviews",
            "expires_at": "2026-05-10T00:00:00Z",
        },
        customer=None,
        merchant_replies=[
            MerchantReply("Abhi busy hoon, baad mein baat karte hain.", "wait intent — should back off"),
            MerchantReply("Ok main ready hoon ab. Kya karna hai?", "returns ready — should resume with context"),
            MerchantReply("Social proof wala idea achha laga. Aage batao.", "engaged, wants to proceed"),
            MerchantReply("Haan kar do. YES.", "explicit yes — onboarding"),
            MerchantReply("Great, thanks Vera!", "closure"),
        ],
        expected_win_criteria={"specificity": 5, "category_fit": 5, "merchant_fit": 5, "trigger_relevance": 6, "engagement_compulsion": 5},
    ),
]


# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict) -> dict:
    url = f"{BOT_URL}{path}"
    resp = httpx.post(url, json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict:
    url = f"{BOT_URL}{path}"
    resp = httpx.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def push_context(scope: str, context_id: str, payload: dict) -> None:
    _post("/v1/context", {
        "scope": scope,
        "context_id": context_id,
        "version": 1,
        "payload": payload,
        "delivered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })


def fire_tick(trigger_id: str) -> list[dict]:
    result = _post("/v1/tick", {
        "now": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "available_triggers": [trigger_id],
    })
    return result.get("actions", [])


def send_reply(conversation_id: str, merchant_id: str, customer_id: str | None,
               message: str, turn_number: int) -> dict:
    return _post("/v1/reply", {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "from_role": "merchant",
        "message": message,
        "received_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "turn_number": turn_number,
    })


def teardown() -> None:
    _post("/v1/teardown", {})


# ---------------------------------------------------------------------------
# Conversation runner
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_number: int
    from_role: str        # "vera" | "merchant"
    body: str
    cta: str | None
    action: str | None    # "send" | "end" | "wait" | "noop"
    rationale: str
    scores: dict[str, int] = field(default_factory=dict)


@dataclass
class ConversationResult:
    scenario_name: str
    scenario_description: str
    category_slug: str
    merchant_id: str
    trigger_kind: str
    merchant_name: str
    turns: list[Turn]
    final_scores: dict[str, int]        # averaged across vera turns
    expected_scores: dict[str, int]
    passed: bool


def run_scenario(s: Scenario) -> ConversationResult:
    print(f"\n{'='*60}")
    print(f"  Running: {s.name}")
    print(f"  {s.description}")
    print(f"{'='*60}")

    turns: list[Turn] = []

    # --- Push context ---
    push_context("category", s.category["slug"], s.category)
    push_context("merchant", s.merchant["merchant_id"], s.merchant)
    push_context("trigger", s.trigger["id"], s.trigger)
    if s.customer:
        push_context("customer", s.customer["customer_id"], s.customer)

    merchant_name = (
        s.merchant.get("identity", {}).get("owner_first_name")
        or s.merchant.get("identity", {}).get("name")
        or "there"
    )

    # --- Turn 1: bot fires the first message via /v1/tick ---
    actions = fire_tick(s.trigger["id"])
    if not actions:
        print("  ⚠️  No action from /v1/tick — skipping scenario")
        return ConversationResult(
            scenario_name=s.name,
            scenario_description=s.description,
            category_slug=s.category["slug"],
            merchant_id=s.merchant["merchant_id"],
            trigger_kind=s.trigger["kind"],
            merchant_name=merchant_name,
            turns=[],
            final_scores={},
            expected_scores=s.expected_win_criteria,
            passed=False,
        )

    action = actions[0]
    conversation_id = action["conversation_id"]
    vera_body = action["body"]
    vera_cta = action["cta"]
    vera_rationale = action["rationale"]
    vera_scores = score_message(vera_body, vera_cta, s.category["slug"],
                                s.trigger["kind"], merchant_name)

    t1 = Turn(1, "vera", vera_body, vera_cta, "send", vera_rationale, vera_scores)
    turns.append(t1)

    print(f"\n  [Turn 1 — Vera]")
    print(f"    {vera_body}")
    print(f"    CTA: {vera_cta} | Score: {vera_scores['total']}/50")

    # --- Subsequent turns ---
    turn_number = 2
    for reply_idx, merchant_reply in enumerate(s.merchant_replies):
        # Merchant turn
        merchant_turn = Turn(
            turn_number, "merchant", merchant_reply.body, None, None,
            f"test: {merchant_reply.description}",
        )
        turns.append(merchant_turn)
        print(f"\n  [Turn {turn_number} — Merchant] ({merchant_reply.description})")
        print(f"    {merchant_reply.body}")
        turn_number += 1

        # Bot reply
        bot_response = send_reply(
            conversation_id=conversation_id,
            merchant_id=s.merchant["merchant_id"],
            customer_id=s.customer["customer_id"] if s.customer else None,
            message=merchant_reply.body,
            turn_number=turn_number,
        )

        bot_action = bot_response.get("action", "noop")
        bot_body = bot_response.get("body", "")
        bot_cta = bot_response.get("cta", "none")
        bot_rationale = bot_response.get("rationale", "")
        bot_scores: dict[str, int] = {}

        if bot_body:
            bot_scores = score_message(bot_body, bot_cta, s.category["slug"],
                                       s.trigger["kind"], merchant_name)

        vera_turn = Turn(turn_number, "vera", bot_body, bot_cta,
                         bot_action, bot_rationale, bot_scores)
        turns.append(vera_turn)

        print(f"\n  [Turn {turn_number} — Vera] (action={bot_action})")
        if bot_body:
            print(f"    {bot_body}")
            print(f"    CTA: {bot_cta} | Score: {bot_scores.get('total', 0)}/50")
        else:
            print(f"    (no body — action={bot_action})")

        turn_number += 1

        # Stop if bot ended or waited
        if bot_action in ("end", "wait"):
            print(f"  → Conversation {bot_action}ed at turn {turn_number - 1}")
            break

    # --- Aggregate vera scores ---
    vera_turns = [t for t in turns if t.from_role == "vera" and t.scores]
    if vera_turns:
        dims = ["specificity", "category_fit", "merchant_fit", "trigger_relevance", "engagement_compulsion"]
        final_scores = {
            d: round(sum(t.scores.get(d, 0) for t in vera_turns) / len(vera_turns))
            for d in dims
        }
        final_scores["total"] = sum(final_scores[d] for d in dims)
    else:
        final_scores = {}

    # --- Check against expected ---
    passed = all(
        final_scores.get(dim, 0) >= threshold
        for dim, threshold in s.expected_win_criteria.items()
    )

    return ConversationResult(
        scenario_name=s.name,
        scenario_description=s.description,
        category_slug=s.category["slug"],
        merchant_id=s.merchant["merchant_id"],
        trigger_kind=s.trigger["kind"],
        merchant_name=merchant_name,
        turns=turns,
        final_scores=final_scores,
        expected_scores=s.expected_win_criteria,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_jsonl(results: list[ConversationResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            for turn in result.turns:
                record = {
                    "scenario": result.scenario_name,
                    "category_slug": result.category_slug,
                    "merchant_id": result.merchant_id,
                    "trigger_kind": result.trigger_kind,
                    "turn_number": turn.turn_number,
                    "from_role": turn.from_role,
                    "body": turn.body,
                    "cta": turn.cta,
                    "action": turn.action,
                    "rationale": turn.rationale,
                    "scores": turn.scores,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n  Saved {path}")


def save_summary(results: list[ConversationResult], path: str) -> None:
    dims = ["specificity", "category_fit", "merchant_fit", "trigger_relevance", "engagement_compulsion"]
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("VERA BOT — WIN CRITERION SUMMARY")
    lines.append(f"Run at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 70)

    passed = sum(1 for r in results if r.passed)
    lines.append(f"\nScenarios run: {len(results)}  |  Passed: {passed}  |  Failed: {len(results)-passed}")
    lines.append("")

    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"{status}  {r.scenario_name}")
        lines.append(f"  {r.scenario_description}")
        lines.append(f"  Turns recorded: {len(r.turns)}")
        lines.append("")

        if r.final_scores:
            lines.append(f"  {'Dimension':<25} {'Got':>5}  {'Min':>5}  {'Pass?':>6}")
            lines.append(f"  {'-'*45}")
            for d in dims:
                got = r.final_scores.get(d, 0)
                minimum = r.expected_scores.get(d, 0)
                ok = "✅" if got >= minimum else "❌"
                lines.append(f"  {d:<25} {got:>5}  {minimum:>5}  {ok:>6}")
            lines.append(f"  {'TOTAL':<25} {r.final_scores.get('total',0):>5}  {sum(r.expected_scores.values()):>5}")
        else:
            lines.append("  (no vera turns scored)")

        lines.append("")

    # Overall dimension averages
    lines.append("=" * 70)
    lines.append("OVERALL DIMENSION AVERAGES")
    lines.append("")
    scored = [r for r in results if r.final_scores]
    if scored:
        for d in dims:
            avg = sum(r.final_scores.get(d, 0) for r in scored) / len(scored)
            lines.append(f"  {d:<25} {avg:.1f} / 10")
        total_avg = sum(r.final_scores.get("total", 0) for r in scored) / len(scored)
        lines.append(f"  {'TOTAL':<25} {total_avg:.1f} / 50")
    lines.append("")

    # Anti-patterns report
    lines.append("=" * 70)
    lines.append("ANTI-PATTERN REPORT")
    lines.append("")
    for r in results:
        vera_bodies = [t.body for t in r.turns if t.from_role == "vera" and t.body]
        for body in vera_bodies:
            for pattern, label in _ANTI_PATTERNS:
                if re.search(pattern, body.lower()):
                    lines.append(f"  ⚠️  {r.scenario_name}: [{label}] found in: {body[:80]}…")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Health check
    try:
        health = _get("/v1/healthz")
        print(f"Bot is up — uptime {health.get('uptime_seconds', '?')}s")
    except Exception as e:
        print(f"❌ Cannot reach bot at {BOT_URL}: {e}")
        print("   Make sure the bot is running: uvicorn bot:app --port 8080")
        sys.exit(1)

    results: list[ConversationResult] = []

    for scenario in SCENARIOS:
        # Clean state before each scenario
        teardown()
        time.sleep(0.3)

        try:
            result = run_scenario(scenario)
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ Scenario {scenario.name} crashed: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(0.5)  # brief pause between scenarios

    # Save outputs
    print("\n")
    save_jsonl(results, OUTPUT_FILE)
    save_summary(results, SUMMARY_FILE)

    # Print summary to stdout
    with open(SUMMARY_FILE) as f:
        print(f.read())


if __name__ == "__main__":
    main()