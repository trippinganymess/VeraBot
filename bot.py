import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Literal, TypeVar, Type

from pydantic import BaseModel, Field, model_validator


class AllowExtraModel(BaseModel):
    class Config:
        extra = "allow"


class PeerStats(AllowExtraModel):
    scope: Optional[str] = None
    avg_rating: Optional[float] = None
    avg_review_count: Optional[int] = None
    avg_views_30d: Optional[int] = None
    avg_calls_30d: Optional[int] = None
    avg_directions_30d: Optional[int] = None
    avg_ctr: Optional[float] = None
    avg_photos: Optional[int] = None
    avg_post_freq_days: Optional[int] = None
    retention_6mo_pct: Optional[float] = None


class CategoryContext(AllowExtraModel):
    slug: str
    display_name: Optional[str] = None
    peer_stats: Optional[PeerStats] = None
    offer_catalog: Optional[list[Dict[str, Any]]] = None


class PerformanceSnapshot(AllowExtraModel):
    window_days: Optional[int] = None
    views: Optional[int] = None
    calls: Optional[int] = None
    directions: Optional[int] = None
    ctr: Optional[float] = None
    leads: Optional[int] = None
    delta_7d: Optional[Dict[str, float]] = None


class MerchantIdentity(AllowExtraModel):
    name: Optional[str] = None
    owner_first_name: Optional[str] = None
    languages: Optional[list[str]] = None


class MerchantContext(AllowExtraModel):
    merchant_id: str
    category_slug: str
    identity: Optional[MerchantIdentity] = None
    performance: Optional[PerformanceSnapshot] = None


class CustomerIdentity(AllowExtraModel):
    name: Optional[str] = None
    language_pref: Optional[str] = None


class CustomerContext(AllowExtraModel):
    customer_id: str
    merchant_id: str
    identity: Optional[CustomerIdentity] = None


class TriggerContext(AllowExtraModel):
    id: str
    scope: str
    kind: str
    source: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    urgency: Optional[int] = None
    suppression_key: Optional[str] = None
    expires_at: Optional[str] = None


class ComposedMessage(BaseModel):
    body: str
    cta: Literal["yes_no", "open_ended", "none"]
    send_as: Literal["vera", "merchant_on_behalf"]
    suppression_key: Optional[str] = None
    rationale: str

    @model_validator(mode="after")
    def enforce_cta_position(self, info):
        if self.cta == "yes_no":
            if not re.search(r"\b(YES|STOP)\b\s*\.?$", self.body, flags=re.IGNORECASE):
                raise ValueError("YES/STOP CTA must be the final sentence.")
        return self

    @model_validator(mode="after")
    def guard_promotional_tone(self, info):
        category = None
        if info.context:
            category = info.context.get("category")
        if category and getattr(category, "slug", None) == "dentists":
            if re.search(r"AMAZING|BEST DEAL|HURRY", self.body, flags=re.IGNORECASE):
                raise ValueError("Promotional tone detected for dentists category.")
        return self

    @model_validator(mode="after")
    def validate_language_mix(self, info):
        if not info.context:
            return self
        language_pref = info.context.get("language_pref")
        if not language_pref:
            return self
        if "hi" in language_pref:
            has_hindi = re.search(r"\b(namaste|aap|kripya|bataiye|haan|ji)\b", self.body, flags=re.IGNORECASE)
            has_english = re.search(r"\b(hi|hello|context|ready|continue)\b", self.body, flags=re.IGNORECASE)
            if not (has_hindi and has_english):
                raise ValueError("Expected Hindi-English code mix in body.")
        return self

    @model_validator(mode="after")
    def validate_referenced_facts(self, info):
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
                offers.extend([offer.get("title", "") for offer in merchant_offers if isinstance(offer, dict)])
            if not any(any(price in title for title in offers) for price in price_mentions):
                raise ValueError("Price mentioned without matching offer catalog.")
        return self


ModelType = TypeVar("ModelType", bound=BaseModel)


def _validate(model_cls: Type[ModelType], raw: Dict[str, Any]) -> ModelType:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(raw)
    return model_cls.parse_obj(raw)


def _load_folder(path: Path, model_cls: type[BaseModel], id_key: str) -> Dict[str, BaseModel]:
    data: Dict[str, BaseModel] = {}
    for file_path in path.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
        model = _validate(model_cls, content)
        item_id = getattr(model, id_key)
        data[item_id] = model
    return data


@lru_cache(maxsize=2)
def load_data(base_path: Optional[str] = None) -> Dict[str, Dict[str, BaseModel]]:
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


def data(base_path: Optional[str] = None) -> Dict[str, Dict[str, BaseModel]]:
    return load_data(base_path)


def _context_version_check(merchant: MerchantContext, trigger: TriggerContext) -> None:
    merchant_version = getattr(merchant, "context_version", None)
    trigger_version = None
    if isinstance(trigger.payload, dict):
        trigger_version = trigger.payload.get("context_version")
    if merchant_version is not None and trigger_version is not None:
        if merchant_version != trigger_version:
            raise ValueError("Stale context detected; please refresh /v1/context.")


def _determine_send_as(trigger: TriggerContext, customer: Optional[CustomerContext]) -> Literal["vera", "merchant_on_behalf"]:
    if trigger.scope == "customer" and customer is not None:
        return "merchant_on_behalf"
    return "vera"


def _language_pref(merchant: MerchantContext, customer: Optional[CustomerContext]) -> str:
    if customer and customer.identity and customer.identity.language_pref:
        return customer.identity.language_pref
    if merchant.identity and merchant.identity.languages:
        if "hi" in merchant.identity.languages:
            return "hi-en mix"
    return "en"


def compose(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Hydrate context and return a Stage 1 compliant message payload."""
    category_ctx: CategoryContext = _validate(CategoryContext, category)
    merchant_ctx: MerchantContext = _validate(MerchantContext, merchant)
    trigger_ctx: TriggerContext = _validate(TriggerContext, trigger)
    customer_ctx: Optional[CustomerContext] = _validate(CustomerContext, customer) if customer else None

    if merchant_ctx.category_slug != category_ctx.slug:
        raise ValueError("Category slug mismatch between merchant and category contexts.")

    _context_version_check(merchant_ctx, trigger_ctx)
    send_as = _determine_send_as(trigger_ctx, customer_ctx)
    language_pref = _language_pref(merchant_ctx, customer_ctx)

    merchant_name = None
    if merchant_ctx.identity:
        merchant_name = merchant_ctx.identity.owner_first_name or merchant_ctx.identity.name
    merchant_name = merchant_name or "there"

    if send_as == "merchant_on_behalf" and customer_ctx and customer_ctx.identity:
        customer_name = customer_ctx.identity.name or "there"
        if language_pref.startswith("hi"):
            body = f"Namaste {customer_name}, {merchant_name} clinic se. Context update ho gaya hai. Aap kab baat karna chaahenge?"
        else:
            body = f"Hi {customer_name}, this is {merchant_name}. Context is updated. When would you like to continue?"
        cta = "open_ended"
    else:
        if language_pref.startswith("hi"):
            body = f"Namaste {merchant_name}, context update ho gaya hai. Jab aap ready ho, bataiye."
        else:
            body = f"Hi {merchant_name}, context is updated. Let me know when you're ready to continue."
        cta = "open_ended"

    message = ComposedMessage.model_validate(
        {
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": trigger_ctx.suppression_key,
            "rationale": "Stage 1 hydration: verified category alignment and sender persona before any strategy is applied.",
        },
        context={
            "category": category_ctx,
            "merchant": merchant_ctx,
            "customer": customer_ctx,
            "trigger": trigger_ctx,
            "language_pref": language_pref,
        },
    )
    return message.model_dump()

