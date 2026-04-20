from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


LIQUIDITY_MAP = {
    "high": "高",
    "medium": "中",
    "low": "低",
    "n/a": "不適用",
}

RISK_MAP = {
    "low": "低",
    "medium": "中",
    "high": "高",
}

PROFILE_MAP = {
    "conservative": "保守型",
    "balanced": "穩健型",
    "growth": "成長型",
}


@dataclass
class Document:
    doc_id: str
    title: str
    category: str
    content: str
    metadata: Dict[str, Any]


def _load_json(filename: str) -> Any:
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profiles() -> List[Dict[str, Any]]:
    return _load_json("customer_profiles.json")


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    profiles = load_profiles()
    for item in profiles:
        if item.get("id") == profile_id:
            return item
    return None


def list_profiles() -> List[Dict[str, Any]]:
    profiles = load_profiles()
    result = []
    for item in profiles:
        display_name = item.get("display_name")
        if not display_name:
            name = item.get("name", item.get("id", "示例客戶"))
            age = item.get("age")
            occupation = item.get("occupation", "")
            if age and occupation:
                display_name = f"{name}｜{age}歲{occupation}"
            else:
                display_name = name

        result.append(
            {
                "id": item.get("id", ""),
                "display_name": display_name,
                "monthly_income": item.get("monthly_income", 0),
                "monthly_expense": item.get("monthly_expense", 0),
                "cash_balance": item.get("cash_balance", 0),
                "investment_balance": item.get("investment_balance", 0),
                "risk_level": _to_profile_label(item.get("risk_level", "")),
                "insurance_status": item.get("insurance_status", "未提供"),
                "goals": item.get("goals", []),
                "notes": item.get("notes", ""),
                "spending_breakdown": item.get("spending_breakdown", {}),
            }
        )
    return result


def load_product_catalog() -> List[Dict[str, Any]]:
    return _load_json("product_catalog.json")


def load_knowledge_base() -> List[Dict[str, Any]]:
    return _load_json("knowledge_base.json")


def _to_profile_label(value: str) -> str:
    return PROFILE_MAP.get(value, value)


def _to_risk_label(value: str) -> str:
    return RISK_MAP.get(value, value)


def _to_liquidity_label(value: str) -> str:
    return LIQUIDITY_MAP.get(value, value)


def _safe_join(values: List[str], default: str = "無") -> str:
    if not values:
        return default
    return "、".join(values)


def _product_to_document(item: Dict[str, Any]) -> Document:
    best_for = item.get("best_for", [])
    avoid_if = item.get("avoid_if", [])
    eligible = item.get("eligible_risk_levels", [])

    content = (
        f"摘要：{item.get('summary', '未提供')}\n"
        f"適合：{_safe_join(best_for)}\n"
        f"不適合：{_safe_join(avoid_if)}\n"
        f"流動性：{_to_liquidity_label(item.get('liquidity', 'n/a'))}\n"
        f"風險：{_to_risk_label(item.get('risk', 'low'))}\n"
        f"可適用風險屬性：{_safe_join([_to_profile_label(x) for x in eligible], default='未限定')}"
    )

    return Document(
        doc_id=item.get("id", item.get("name", "product")),
        title=item.get("name", "未命名商品"),
        category=item.get("category", "product"),
        content=content,
        metadata={
            "type": "product",
            "name": item.get("name", ""),
            "summary": item.get("summary", ""),
            "best_for": best_for,
            "avoid_if": avoid_if,
            "liquidity": _to_liquidity_label(item.get("liquidity", "n/a")),
            "risk": _to_risk_label(item.get("risk", "low")),
            "eligible_risk_levels": [_to_profile_label(x) for x in eligible],
        },
    )


def _knowledge_to_document(item: Dict[str, Any]) -> Document:
    title = item.get("title", "未命名知識")
    category = item.get("category", "knowledge")
    content = item.get("content") or item.get("text") or item.get("body") or "未提供內容"

    return Document(
        doc_id=item.get("id", title),
        title=title,
        category=category,
        content=content,
        metadata={
            "type": "knowledge",
            "title": title,
            "category": category,
        },
    )


def build_documents() -> List[Document]:
    docs: List[Document] = []

    for item in load_product_catalog():
        docs.append(_product_to_document(item))

    for item in load_knowledge_base():
        docs.append(_knowledge_to_document(item))

    return docs


class TfidfRetriever:
    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents = documents or build_documents()
        self.vectorizer = TfidfVectorizer()
        self.corpus = [f"{doc.title}\n{doc.category}\n{doc.content}" for doc in self.documents]
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for idx, score in ranked:
            doc = self.documents[idx]
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "category": doc.category,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )
        return results


_DEFAULT_RETRIEVER: Optional[TfidfRetriever] = None


def get_retriever() -> TfidfRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = TfidfRetriever()
    return _DEFAULT_RETRIEVER


def search_documents(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    return get_retriever().search(query=query, top_k=top_k)