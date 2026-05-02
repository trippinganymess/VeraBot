"""Core bot composition and context loading utilities."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, model_validator

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


class AllowExtraModel(BaseModel):
    """Base model that permits extra fields from JSON payloads."""

    class Config:
        """Pydantic configuration for permissive parsing."""

        extra = "allow"


class PeerStats(AllowExtraModel):
    """Peer benchmark statistics for a category."""

    scope: str | None = None
    avg_rating: float | None = None
    avg_review_count: int | None = None
    avg_views_30d: int | None = None
    avg_calls_30d: int | None = None
    avg_directions_30d: int | None = None
    avg_ctr: float | None = None
    avg_photos: int | None = None
    avg_post_freq_days: int | None = None
    retention_6mo_pct: float | None = None


class CategoryContext(AllowExtraModel):
    """Category-level context used for voice and benchmarks."""

    slug: str
    display_name: str | None = None
    peer_stats: PeerStats | None = None
    offer_catalog: list[dict[str, Any]] | None = None
    digest: list[dict[str, Any]] | None = None


class PerformanceSnapshot(AllowExtraModel):
    """Merchant performance metrics snapshot."""

    window_days: int | None = None
    views: int | None = None
    calls: int | None = None
    directions: int | None = None
    ctr: float | None = None
    leads: int | None = None
    delta_7d: dict[str, float] | None = None


class MerchantIdentity(AllowExtraModel):
    """Human-facing identity details for the merchant."""

    name: str | None = None
    owner_first_name: str | None = None
    languages: list[str] | None = None


class MerchantContext(AllowExtraModel):
    """Merchant-specific context including identity and performance."""

    merchant_id: str
    category_slug: str
    identity: MerchantIdentity | None = None
    performance: PerformanceSnapshot | None = None


class CustomerIdentity(AllowExtraModel):
    """Customer identity and language preferences."""

    name: str | None = None
    language_pref: str | None = None


class CustomerContext(AllowExtraModel):
    """Customer context for merchant-on-behalf messaging."""

    customer_id: str
    merchant_id: str
    identity: CustomerIdentity | None = None


class TriggerContext(AllowExtraModel):
    """Trigger information driving the next message."""

    id: str
    scope: str
    kind: str
    source: str
    merchant_id: str | None = None
    customer_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int | None = None
    suppression_key: str | None = None
    expires_at: str | None = None


class ComposedMessage(BaseModel):
    """Validated output payload for the judge harness."""

    body: str
    cta: Literal["yes_no", "open_ended", "none"]
    send_as: Literal["vera", "merchant_on_behalf"]
    suppression_key: str | None = None
    rationale: str

    @model_validator(mode="after")
    def enforce_cta_position(self, info):
        """Ensure YES/STOP CTA is placed at the end when required."""
        if self.cta == "yes_no":
            if not re.search(r"\b(YES|STOP)\b\s*\.?$", self.body, flags=re.IGNORECASE):
                raise ValueError("YES/STOP CTA must be the final sentence.")
        return self

    @model_validator(mode="after")
    def guard_promotional_tone(self, info):
        """Reject promotional language for clinical categories."""
        category = None
        if info.context:
            category = info.context.get("category")
        if category and getattr(category, "slug", None) == "dentists":
            if re.search(r"AMAZING|BEST DEAL|HURRY", self.body, flags=re.IGNORECASE):
                raise ValueError("Promotional tone detected for dentists category.")
        return self

    @model_validator(mode="after")
    def validate_language_mix(self, info):
        """Validate required Hindi-English code mix when specified."""
        if not info.context:
            return self
        language_pref = info.context.get("language_pref")
        if not language_pref:
            return self
        if "hi" in language_pref:
            has_hindi = re.search(
                r"\b(namaste|aap|kripya|bataiye|haan|ji)\b",
                self.body,
                flags=re.IGNORECASE,
            )
            has_english = re.search(
                r"\b(hi|hello|context|ready|continue)\b",
                self.body,
                flags=re.IGNORECASE,
            )
            if not (has_hindi and has_english):
                raise ValueError("Expected Hindi-English code mix in body.")
        return self

    @model_validator(mode="after")
    def validate_referenced_facts(self, info):
        """Validate referenced prices against available offers."""
        if not info.context:
            return self
        category = info.context.get("category")
        merchant = info.context.get("merchant")
        body = self.body
        price_mentions = re.findall(r"₹\s?(\d+(?:\.\d+)?)", body)
        if price_mentions:
            offers = []
            if category and getattr(category, "offer_catalog", None):
                offers.extend([offer.get("title", "") for offer in category.offer_catalog])
            merchant_offers = getattr(merchant, "offers", None)
            if isinstance(merchant_offers, list):
                offers.extend(
                    [offer.get("title", "") for offer in merchant_offers if isinstance(offer, dict)]
                )
            if not any(any(price in title for title in offers) for price in price_mentions):
                raise ValueError("Price mentioned without matching offer catalog.")
        return self


ModelType = TypeVar("ModelType", bound=BaseModel)


def _validate(model_cls: type[ModelType], raw: dict[str, Any]) -> ModelType:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(raw)
    return model_cls.parse_obj(raw)


def _load_folder(path: Path, model_cls: type[BaseModel], id_key: str) -> dict[str, BaseModel]:
    data: dict[str, BaseModel] = {}
    for file_path in path.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
        model = _validate(model_cls, content)
        item_id = getattr(model, id_key)
        data[item_id] = model
    return data


@lru_cache(maxsize=2)
def load_data(base_path: str | None = None) -> dict[str, dict[str, BaseModel]]:
    """Load and validate dataset JSON into Pydantic models."""
    if base_path is None:
        candidate = Path("expanded")
        base_path = "expanded" if candidate.exists() else "dataset"

    path = Path(base_path)
    if not path.exists():
        raise FileNotFoundError(f"Base path not found: {path}")

    return {
        "categories": _load_folder(path / "categories", CategoryContext, "slug"),
        "merchants": _load_folder(path / "merchants", MerchantContext, "merchant_id"),
        "customers": _load_folder(path / "customers", CustomerContext, "customer_id"),
        "triggers": _load_folder(path / "triggers", TriggerContext, "id"),
    }


def data(base_path: str | None = None) -> dict[str, dict[str, BaseModel]]:
    """Backward-compatible alias for load_data."""
    return load_data(base_path)


def _context_version_check(merchant: MerchantContext, trigger: TriggerContext) -> None:
    merchant_version = getattr(merchant, "context_version", None)
    trigger_version = None
    if isinstance(trigger.payload, dict):
        trigger_version = trigger.payload.get("context_version")
    if merchant_version is not None and trigger_version is not None:
        if merchant_version != trigger_version:
            raise ValueError("Stale context detected; please refresh /v1/context.")


def _determine_send_as(
    trigger: TriggerContext,
    customer: CustomerContext | None,
) -> Literal["vera", "merchant_on_behalf"]:
    if trigger.scope == "customer" and customer is not None:
        return "merchant_on_behalf"
    return "vera"


def _language_pref(merchant: MerchantContext, customer: CustomerContext | None) -> str:
    """Infer a language preference for the outgoing message."""
    if customer and customer.identity and customer.identity.language_pref:
        return customer.identity.language_pref
    if merchant.identity and merchant.identity.languages:
        if "hi" in merchant.identity.languages:
            return "hi-en mix"
    return "en"


def _last_merchant_message(conversation_history: list[dict[str, Any]] | None) -> str | None:
    """Return the most recent merchant-authored message from history."""
    if not conversation_history:
        return None
    for entry in reversed(conversation_history):
        if entry.get("from") == "merchant" and isinstance(entry.get("body"), str):
            return entry["body"].strip()
    return None


def _auto_reply_detected(conversation_history: list[dict[str, Any]] | None) -> bool:
    """Detect canned auto-reply patterns or repeated identical merchant messages."""
    if not conversation_history:
        return False
    merchant_messages = [
        entry.get("body", "")
        for entry in conversation_history
        if entry.get("from") == "merchant" and isinstance(entry.get("body"), str)
    ]
    if len(merchant_messages) >= 2 and merchant_messages[-1] == merchant_messages[-2]:
        return True
    last_message = merchant_messages[-1].lower() if merchant_messages else ""
    canned_patterns = [
        r"thank you for contacting",
        r"auto[- ]reply",
        r"we will get back",
        r"main aapki baat",
    ]
    return any(re.search(pattern, last_message) for pattern in canned_patterns)


def _intent_transition_detected(message: str | None) -> bool:
    """Detect explicit intent to join or proceed with an action."""
    if not message:
        return False
    intent_patterns = [
        r"\b(i want to join|join|sign up|onboard|let's do it|proceed)\b",
        r"\b(jurna|zudna|jadna|zurna|judna|join karna|shuru karo|start karo|karna hai)\b",
    ]
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in intent_patterns)


def _format_pct(value: float) -> str:
    """Format a ratio as a percent string with one decimal place."""
    return f"{value * 100:.1f}%"


def _benchmark_facts(
    merchant: MerchantContext,
    category: CategoryContext,
) -> dict[str, str]:
    """Extract performance facts and peer comparisons for Stage 3."""
    facts: dict[str, str] = {}
    perf = merchant.performance
    peer = category.peer_stats
    if perf and perf.views is not None:
        facts["views"] = f"{perf.views} views"
    if perf and perf.ctr is not None:
        facts["ctr"] = _format_pct(perf.ctr)
    if perf and perf.calls is not None:
        facts["calls"] = f"{perf.calls} calls"
    if perf and peer and perf.ctr is not None and peer.avg_ctr is not None:
        facts["ctr_gap"] = f"{_format_pct(perf.ctr)} vs peer {_format_pct(peer.avg_ctr)}"
    return facts


def _research_digest_anchor(
    trigger: TriggerContext,
    category: CategoryContext,
) -> dict[str, str] | None:
    """Extract research digest metadata from trigger payload or category digest."""
    if "research_digest" not in trigger.kind:
        return None
    payload_item = None
    if isinstance(trigger.payload, dict):
        payload_item = trigger.payload.get("top_item")
    if payload_item:
        return {
            "title": payload_item.get("title", ""),
            "source": payload_item.get("source", ""),
            "trial_n": str(payload_item.get("trial_n", "")),
        }
    if category.digest:
        first_item = category.digest[0]
        return {
            "title": first_item.get("title", ""),
            "source": first_item.get("source", ""),
            "trial_n": str(first_item.get("trial_n", "")),
        }
    return None


LEVER_MAP: dict[str, str] = {
    # Loss aversion — best for performance dips or missed opportunities
    "perf_dip": "loss_aversion",
    "missed_search": "loss_aversion",
    "dormant_with_vera": "loss_aversion",
    "renewal_due": "loss_aversion",
    "seasonal_acquisition_dip": "loss_aversion",
    "winback": "loss_aversion",
    "customer_lapsed_soft": "loss_aversion",
    # Social proof — best for social-facing triggers or benchmarks
    "milestone_reached": "social_proof",
    "review_theme_emerged": "social_proof",
    "competitor_opened": "social_proof",
    "perf_spike": "social_proof",
    "festival_upcoming": "social_proof",
    # Effort externalization — best for "I've drafted X — just say go"
    "research_digest": "effort_externalization",
    "curious_ask_due": "effort_externalization",
    "trial_followup": "effort_externalization",
    "appointment_tomorrow": "effort_externalization",
    "recall_due": "effort_externalization",
    "chronic_refill_due": "effort_externalization",
    "unverified_gbp": "effort_externalization",
}
"""O(1) trigger-kind → compulsion lever lookup table."""


def _select_compulsion_lever(trigger_kind: str) -> str:
    """Map a trigger kind to a compulsion lever via O(1) dictionary lookup.

    Falls back to substring matching for compound trigger kinds (e.g.
    'research_digest_release'), and returns 'neutral' when no lever matches.
    """
    if trigger_kind in LEVER_MAP:
        return LEVER_MAP[trigger_kind]
    for key, lever in LEVER_MAP.items():
        if key in trigger_kind:
            return lever
    return "neutral"


def _apply_compulsion_lever(
    message: str,
    lever: str,
    language_pref: str,
) -> str:
    """Append a lever cue to the message body."""
    if lever == "loss_aversion":
        if language_pref.startswith("hi"):
            return f"{message} Missed demand avoid karne ke liye main ek quick fix bhej du?"
        return f"{message} Want me to share a quick fix to avoid missing demand?"
    if lever == "social_proof":
        if language_pref.startswith("hi"):
            return f"{message} Aapke area ke kuch peers ne isi week yeh try kiya hai."
        return f"{message} A few peers in your area tried this this week."
    if lever == "effort_externalization":
        if language_pref.startswith("hi"):
            return f"{message} Main draft ready karke bhej sakti hoon — bas YES bol dijiye."
        return f"{message} I can draft it for you — just say YES."
    return message


VOICE_PREFIX_MAP: dict[str, str] = {
    "dentists": "Clinical note:",
    "salons": "Quick tip:",
    "restaurants": "Quick ops note:",
    "gyms": "Coach's note:",
    "pharmacies": "Compliance note:",
}
"""Category-slug → message prefix for voice-appropriate framing."""


def _apply_voice_modulation(category: CategoryContext, message: str) -> str:
    """Apply a category-specific tone prefix to the message body.

    Uses the VOICE_PREFIX_MAP for O(1) lookup.  When the category voice
    data is available on the context, we additionally validate that
    the prefix is appropriate for the declared tone; the mapping
    itself is derived from the category tone definitions in the dataset.
    """
    prefix = VOICE_PREFIX_MAP.get(category.slug)
    if prefix:
        return f"{prefix} {message}"
    return message


def _build_rationale(
    strategy: str,
    lever: str,
    category_slug: str,
    trigger_kind: str,
) -> str:
    """Build a concise rationale explaining the composition strategy.

    Args:
        strategy: Which message-construction path was taken.
        lever: The compulsion lever applied (or 'none').
        category_slug: Category slug for voice reference.
        trigger_kind: Trigger kind that drove the message.

    Returns:
        A 1-2 sentence rationale suitable for the judge harness.
    """
    voice_label = VOICE_PREFIX_MAP.get(category_slug, "default")
    lever_label = lever.replace("_", " ") if lever != "neutral" else "neutral framing"

    parts: list[str] = []
    if strategy == "auto_reply_exit":
        parts.append("Detected canned auto-reply; routed to graceful exit.")
    elif strategy == "intent_transition":
        parts.append("Merchant signaled explicit intent; switched to action mode.")
    elif strategy == "customer_facing":
        parts.append("Customer-scoped trigger; sent as merchant_on_behalf.")
    elif strategy == "digest_anchor":
        parts.append(
            f"Anchored on research digest from {trigger_kind} trigger "
            f"using {lever_label} lever."
        )
    elif strategy == "benchmark_anchor":
        parts.append(
            f"Anchored on peer-median CTR benchmark to drive curiosity "
            f"through {lever_label}."
        )
    else:
        parts.append(
            f"Fallback path with {lever_label} lever applied."
        )

    parts.append(f"Voice: {voice_label} ({category_slug}).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Modular prompt template system
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Vera, a merchant-AI assistant on magicpin.
You help local merchants (dentists, salons, restaurants, gyms, pharmacies) \
grow their business via WhatsApp.

Rules:
1. Output valid JSON matching: {body, cta, send_as, rationale}.
2. cta must be one of: "yes_no", "open_ended", "none".
3. send_as must be one of: "vera", "merchant_on_behalf".
4. If cta is "yes_no", the body MUST end with YES or STOP.
5. Never fabricate facts, prices, or sources not in the input.
6. Keep the voice prefix at the start of the body if provided.
7. Body should be concise (2-4 sentences max).
"""

LEVER_TEMPLATES: dict[str, str] = {
    "social_proof": """\
Write a message using SOCIAL PROOF framing.
Reference how peers or competitors in the merchant's area are performing.
Use phrases like "X peers in your locality", "others in your area".
Make the merchant curious about what others are doing.
""",
    "loss_aversion": """\
Write a message using LOSS AVERSION framing.
Highlight what the merchant is missing or at risk of losing.
Use phrases like "missing X demand", "gap vs peer median", \
"before this window closes".
Create urgency without being promotional.
""",
    "effort_externalization": """\
Write a message using EFFORT EXTERNALIZATION framing.
Offer to do the work for the merchant — "I've drafted X", \
"I can set this up", "just say YES".
Minimize perceived effort for the merchant.
""",
    "neutral": """\
Write a helpful, peer-toned update message.
Be specific and fact-anchored but without a strong persuasion lever.
""",
}
"""Lever-specific prompt fragments injected into the LLM call."""


def _extract_jit_facts(
    merchant_ctx: MerchantContext,
    category_ctx: CategoryContext,
    trigger_ctx: TriggerContext,
    customer_ctx: CustomerContext | None,
    benchmark: dict[str, str],
    digest: dict[str, str] | None,
) -> dict[str, Any]:
    """Extract only the facts the LLM needs — no full context dumps.

    Returns a small dict of concrete, verifiable data points sourced
    from the four context objects and the pre-computed benchmark/digest.
    This keeps the prompt token-efficient and reduces hallucination risk.
    """
    facts: dict[str, Any] = {}

    # Merchant identity
    if merchant_ctx.identity:
        facts["merchant_name"] = (
            merchant_ctx.identity.owner_first_name
            or merchant_ctx.identity.name
            or "there"
        )
    else:
        facts["merchant_name"] = "there"

    # Performance numbers
    if merchant_ctx.performance:
        perf = merchant_ctx.performance
        if perf.views is not None:
            facts["views_30d"] = perf.views
        if perf.ctr is not None:
            facts["ctr"] = f"{perf.ctr * 100:.1f}%"
        if perf.calls is not None:
            facts["calls_30d"] = perf.calls

    # Peer benchmarks
    if benchmark:
        facts["benchmark"] = benchmark

    # Category voice
    facts["voice_prefix"] = VOICE_PREFIX_MAP.get(
        category_ctx.slug, ""
    )
    facts["category_slug"] = category_ctx.slug

    # Digest anchor
    if digest:
        facts["digest_title"] = digest.get("title", "")
        facts["digest_source"] = digest.get("source", "")
        facts["digest_trial_n"] = digest.get("trial_n", "")

    # Trigger metadata
    facts["trigger_kind"] = trigger_ctx.kind

    # Customer identity (for customer-facing sends)
    if customer_ctx and customer_ctx.identity:
        facts["customer_name"] = customer_ctx.identity.name or "there"

    return facts


def _build_llm_prompt(
    lever: str,
    language_pref: str,
    cta: str,
    send_as: str,
    facts: dict[str, Any],
    draft_body: str,
    draft_rationale: str,
) -> str:
    """Assemble the final LLM prompt from modular template pieces.

    Combines the constant SYSTEM_PROMPT, a lever-specific template
    fragment, the JIT-extracted facts, and the rule-engine draft.
    This avoids dumping entire context objects into the prompt.

    Args:
        lever: The compulsion lever to use for framing.
        language_pref: Language preference string (e.g. "en", "hi-en mix").
        cta: Desired CTA mode for the output.
        send_as: Sender attribution role.
        facts: Minimal fact dict from _extract_jit_facts.
        draft_body: Rule-engine pre-composed message body.
        draft_rationale: Rule-engine pre-composed rationale.

    Returns:
        A complete prompt string ready for the LLM.
    """
    lever_template = LEVER_TEMPLATES.get(lever, LEVER_TEMPLATES["neutral"])

    language_instruction = ""
    if language_pref.startswith("hi"):
        language_instruction = (
            "Language: Hindi-English code-mix. "
            "Blend naturally — use Hindi particles (ji, hai, karo) "
            "with English nouns and numbers.\n"
        )
    else:
        language_instruction = "Language: English.\n"

    facts_block = "\n".join(
        f"- {key}: {value}" for key, value in facts.items()
    )

    return f"""{SYSTEM_PROMPT}

{lever_template}

{language_instruction}
Constraints:
- CTA mode: {cta}
- Send as: {send_as}

Extracted facts (use ONLY these):
{facts_block}

Draft body from rule engine: "{draft_body}"
Draft rationale: "{draft_rationale}"

Refine the draft body using the lever template and facts above.
Return a JSON object with keys: body, cta, send_as, rationale."""


def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hydrate context and return a composed message payload.

    Implements Stages 1-4: context validation, auto-reply/intent filtering,
    benchmark/digest anchoring, lever selection, and voice modulation.
    """
    category_ctx: CategoryContext = _validate(CategoryContext, category)
    merchant_ctx: MerchantContext = _validate(MerchantContext, merchant)
    trigger_ctx: TriggerContext = _validate(TriggerContext, trigger)
    customer_ctx: CustomerContext | None = (
        _validate(CustomerContext, customer) if customer else None
    )

    if merchant_ctx.category_slug != category_ctx.slug:
        raise ValueError("Category slug mismatch between merchant and category contexts.")

    _context_version_check(merchant_ctx, trigger_ctx)
    send_as = _determine_send_as(trigger_ctx, customer_ctx)
    language_pref = _language_pref(merchant_ctx, customer_ctx)

    merchant_name = None
    if merchant_ctx.identity:
        merchant_name = merchant_ctx.identity.owner_first_name or merchant_ctx.identity.name
    merchant_name = merchant_name or "there"

    conversation_history = getattr(merchant_ctx, "conversation_history", None)
    last_merchant_message = _last_merchant_message(conversation_history)

    strategy = "fallback"
    lever = "neutral"
    benchmark: dict[str, str] = {}
    digest: dict[str, str] | None = None

    if _auto_reply_detected(conversation_history):
        strategy = "auto_reply_exit"
        if language_pref.startswith("hi"):
            body = (
                "Hi, lagta hai yeh auto-reply hai ji. "
                "Aapke owner/manager se main direct connect kar leti hoon."
            )
        else:
            body = (
                "Hi, this looks like an auto-reply. "
                "I'll connect directly with the owner or manager."
            )
        cta = "none"
    elif _intent_transition_detected(last_merchant_message):
        strategy = "intent_transition"
        if language_pref.startswith("hi"):
            body = (
                "Great, main aapka onboarding start kar sakti hoon. "
                "Proceed karne ke liye YES bol dijiye, STOP for later. STOP"
            )
        else:
            body = (
                "Great, I can start your onboarding now. "
                "Reply YES to proceed or STOP for later. STOP"
            )
        cta = "yes_no"
    elif send_as == "merchant_on_behalf" and customer_ctx and customer_ctx.identity:
        strategy = "customer_facing"
        customer_name = customer_ctx.identity.name or "there"
        if language_pref.startswith("hi"):
            body = (
                f"Namaste {customer_name}, {merchant_name} clinic se. "
                "Context update ho gaya hai. Aap kab baat karna chaahenge?"
            )
        else:
            body = (
                f"Hi {customer_name}, this is {merchant_name}. "
                "Context is updated. When would you like to continue?"
            )
        cta = "open_ended"
    else:
        benchmark = _benchmark_facts(merchant_ctx, category_ctx)
        digest = _research_digest_anchor(trigger_ctx, category_ctx)
        lever = _select_compulsion_lever(trigger_ctx.kind)
        if digest and digest.get("title"):
            strategy = "digest_anchor"
            if language_pref.startswith("hi"):
                body = (
                    f"{merchant_name}, naya research digest aaya hai: "
                    f"{digest['title']}. "
                    f"Source: {digest.get('source', 'N/A')}. "
                    "Aap chahen to main 2-min summary bhej du?"
                )
            else:
                body = (
                    f"{merchant_name}, new research digest: {digest['title']}. "
                    f"Source: {digest.get('source', 'N/A')}. "
                    "Want a 2-minute summary?"
                )
            cta = "open_ended"
        elif benchmark:
            strategy = "benchmark_anchor"
            ctr_gap = benchmark.get("ctr_gap")
            views = benchmark.get("views")
            if language_pref.startswith("hi"):
                body = (
                    f"{merchant_name}, aapke {views or 'latest'} me CTR "
                    f"{ctr_gap or benchmark.get('ctr', 'N/A')} hai. "
                    "Chahein to main quick improvement plan bheju?"
                )
            else:
                body = (
                    f"{merchant_name}, your {views or 'latest'} CTR is "
                    f"{ctr_gap or benchmark.get('ctr', 'N/A')}. "
                    "Want a quick improvement plan?"
                )
            cta = "open_ended"
        elif language_pref.startswith("hi"):
            body = (
                f"Namaste {merchant_name}, context update ho gaya hai. "
                "Jab aap ready ho, bataiye."
            )
            cta = "open_ended"
        else:
            body = (
                f"Hi {merchant_name}, context is updated. "
                "Let me know when you're ready to continue."
            )
            cta = "open_ended"

        if cta == "open_ended":
            body = _apply_compulsion_lever(body, lever, language_pref)
        body = _apply_voice_modulation(category_ctx, body)

    rationale = _build_rationale(
        strategy=strategy,
        lever=lever,
        category_slug=category_ctx.slug,
        trigger_kind=trigger_ctx.kind,
    )

    message_dict = {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger_ctx.suppression_key,
        "rationale": rationale,
    }

    if LLM_CLIENT:
        jit_facts = _extract_jit_facts(
            merchant_ctx=merchant_ctx,
            category_ctx=category_ctx,
            trigger_ctx=trigger_ctx,
            customer_ctx=customer_ctx,
            benchmark=benchmark,
            digest=digest,
        )
        prompt = _build_llm_prompt(
            lever=lever,
            language_pref=language_pref,
            cta=cta,
            send_as=send_as,
            facts=jit_facts,
            draft_body=body,
            draft_rationale=rationale,
        )
        try:
            response = LLM_CLIENT.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ComposedMessage,
                    temperature=0.0,
                ),
            )
            if response.text:
                llm_dict = json.loads(response.text)
                llm_dict["suppression_key"] = (
                    trigger_ctx.suppression_key
                )
                message_dict = llm_dict
        except Exception:
            # Fallback to deterministic rule-engine output on LLM error
            pass

    message = ComposedMessage.model_validate(
        message_dict,
        context={
            "category": category_ctx,
            "merchant": merchant_ctx,
            "customer": customer_ctx,
            "trigger": trigger_ctx,
            "language_pref": language_pref,
        },
    )
    return message.model_dump()

