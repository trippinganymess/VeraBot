import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AllowExtraModel(BaseModel):
    class Config:
        extra = "allow"


class CategoryContext(AllowExtraModel):
    slug: str
    display_name: Optional[str] = None


class MerchantContext(AllowExtraModel):
    merchant_id: str
    category_slug: str


class CustomerContext(AllowExtraModel):
    customer_id: str
    merchant_id: str


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


def _validate(model_cls: type[BaseModel], raw: Dict[str, Any]) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(raw)
    return model_cls.model_validate(raw)


def _load_folder(path: Path, model_cls: type[BaseModel], id_key: str) -> Dict[str, BaseModel]:
    data: Dict[str, BaseModel] = {}
    for file_path in path.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as f:
            content = json.load(f)
        model = _validate(model_cls, content)
        item_id = getattr(model, id_key)
        data[item_id] = model
    return data


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

