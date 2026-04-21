from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - optional dependency for offline demo
    class HumanMessage:  # type: ignore[override]
        def __init__(self, content: str) -> None:
            self.content = content

    class SystemMessage:  # type: ignore[override]
        def __init__(self, content: str) -> None:
            self.content = content

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency for offline demo
    LANGGRAPH_AVAILABLE = False
    END = "__end__"

    class _CompiledGraph:
        def __init__(self, state_graph: "StateGraph") -> None:
            self.state_graph = state_graph

        def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(initial_state)
            current = self.state_graph.entry_point
            while current and current != END:
                node_fn = self.state_graph.nodes[current]
                updates = node_fn(state) or {}
                state.update(updates)

                if current in self.state_graph.conditional_edges:
                    router, mapping = self.state_graph.conditional_edges[current]
                    route_key = router(state)
                    current = mapping[route_key]
                else:
                    next_nodes = self.state_graph.edges.get(current, [])
                    current = next_nodes[0] if next_nodes else END
            return state

    class StateGraph:  # type: ignore[override]
        def __init__(self, _state_schema: Any) -> None:
            self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
            self.edges: Dict[str, List[str]] = {}
            self.conditional_edges: Dict[str, Any] = {}
            self.entry_point: Optional[str] = None

        def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
            self.nodes[name] = fn

        def set_entry_point(self, name: str) -> None:
            self.entry_point = name

        def add_edge(self, source: str, target: str) -> None:
            self.edges.setdefault(source, []).append(target)

        def add_conditional_edges(self, source: str, router: Callable[[Dict[str, Any]], str], mapping: Dict[str, str]) -> None:
            self.conditional_edges[source] = (router, mapping)

        def compile(self) -> _CompiledGraph:
            return _CompiledGraph(self)

from llm_client import get_llm
from retriever import build_retriever, get_profile, load_profiles


Intent = Literal["budgeting", "goal_planning", "product_education", "insurance_review", "general"]
Route = Literal["continue", "blocked"]


class FinanceState(TypedDict, total=False):
    profile_id: str
    user_query: str
    profile: Dict[str, Any]
    profile_summary: str
    intent: Intent
    route: Route
    warnings: List[str]
    retrieved_docs: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    draft_response: str
    final_response: str
    sources: List[str]


def summarize_profile(profile: Dict[str, Any]) -> str:
    spending_lines = [f"{k}:{v:,}元" for k, v in profile.get("spending_breakdown", {}).items()]
    return (
        f"客戶：{profile['display_name']}\n"
        f"月收入：{profile['monthly_income']:,}元\n"
        f"月支出：{profile['monthly_expense']:,}元\n"
        f"現金餘額：{profile['cash_balance']:,}元\n"
        f"投資餘額：{profile['investment_balance']:,}元\n"
        f"貸款餘額：{profile['loan_balance']:,}元\n"
        f"風險屬性：{profile['risk_level']}\n"
        f"保險狀況：{profile['insurance_status']}\n"
        f"目標：{'；'.join(profile['goals'])}\n"
        f"已持有：{'、'.join(profile['products_owned'])}\n"
        f"支出結構：{'；'.join(spending_lines)}\n"
        f"備註：{profile['notes']}"
    )


FORBIDDEN_KEYWORDS = [
    "otp",
    "密碼",
    "cvv",
    "完整卡號",
    "身分證",
    "帳號給你",
    "幫我輸入",
    "代我轉帳",
]


INVESTMENT_KEYWORDS = ["基金", "股票", "投資", "報酬", "定期定額", "退休"]
INSURANCE_KEYWORDS = ["保單", "保險", "醫療", "壽險", "重大疾病", "保障"]
GOAL_KEYWORDS = ["存", "目標", "預備金", "旅遊", "教育金", "退休"]
BUDGET_KEYWORDS = ["花太多", "支出", "預算", "消費", "控制", "刷卡"]


PROMPT_SYSTEM = """
你是 CUBE Copilot 的示範版金融助理，任務是把複雜的金融資訊轉成客戶聽得懂、可執行、且合規的建議。

請嚴格遵守以下規則：
1. 只能根據提供的客戶資料與知識片段回覆，不可捏造商品、條件或保證報酬。
2. 不可要求或儲存完整卡號、OTP、網銀密碼、CVV、身分證影本等敏感資訊。
3. 若涉及投資或保險，須說明這只是 AI 助理的初步建議，最終仍須做適配性或保障檢視。
4. 回覆要用繁體中文、條列清楚、語氣專業但易懂。
5. 回覆格式固定為：
   - 財務觀察
   - 建議行動（最多 3 點）
   - 為何這樣建議
   - 風險提醒
   - 來源
""".strip()


def classify_intent(query: str) -> Intent:
    lowered = query.lower()
    if any(keyword in query for keyword in BUDGET_KEYWORDS):
        return "budgeting"
    if any(keyword in query for keyword in GOAL_KEYWORDS):
        return "goal_planning"
    if any(keyword in query for keyword in INSURANCE_KEYWORDS):
        return "insurance_review"
    if any(keyword in query for keyword in INVESTMENT_KEYWORDS) or "產品" in query or "推薦" in query:
        return "product_education"
    if "budget" in lowered or "save" in lowered:
        return "budgeting"
    return "general"


@lru_cache(maxsize=1)
def _get_retriever():
    return build_retriever()


@lru_cache(maxsize=1)
def _get_graph():
    graph = StateGraph(FinanceState)
    graph.add_node("load_profile", load_profile_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("load_profile")
    graph.add_edge("load_profile", "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"blocked": "finalize", "continue": "analyze"})
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "review")
    graph.add_edge("review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def load_profile_node(state: FinanceState) -> FinanceState:
    profile = get_profile(state["profile_id"])
    return {
        "profile": profile,
        "profile_summary": summarize_profile(profile),
        "intent": classify_intent(state["user_query"]),
        "warnings": [],
        "sources": [],
    }


def guardrail_node(state: FinanceState) -> FinanceState:
    query = state["user_query"].lower()
    warnings = list(state.get("warnings", []))

    if any(keyword in query for keyword in FORBIDDEN_KEYWORDS):
        return {
            "route": "blocked",
            "final_response": (
                "此問題涉及敏感資訊或高風險操作。\n\n"
                "建議：\n"
                "1. 不要在對話中提供完整卡號、OTP、密碼或 CVV。\n"
                "2. 請改走正式驗證流程或真人客服。\n"
                "3. 若有帳戶異常，請立即啟用安全機制與聯絡銀行。"
            ),
            "warnings": warnings + ["已觸發敏感資訊防護規則。"],
            "sources": ["敏感資訊處理政策"],
        }

    if any(keyword in state["user_query"] for keyword in INVESTMENT_KEYWORDS):
        warnings.append("此問題涉及投資，回覆僅做教育與初步引導，最終仍需適配性確認。")
    if any(keyword in state["user_query"] for keyword in INSURANCE_KEYWORDS):
        warnings.append("此問題涉及保險，回覆僅做缺口盤點與初步建議，最終仍需保障檢視。")

    return {"route": "continue", "warnings": warnings}


def analyze_node(state: FinanceState) -> FinanceState:
    profile = state["profile"]
    monthly_surplus = profile["monthly_income"] - profile["monthly_expense"]
    savings_rate = monthly_surplus / profile["monthly_income"] if profile["monthly_income"] else 0
    essential_expense = sum(
        value
        for key, value in profile.get("spending_breakdown", {}).items()
        if key in {"固定支出", "房貸", "家庭生活", "教育", "保險"}
    )
    emergency_fund_months = round(profile["cash_balance"] / max(essential_expense, 1), 1)
    spending_breakdown = profile["spending_breakdown"]
    variable_categories = {
        key: value
        for key, value in spending_breakdown.items()
        if key not in {"固定支出", "房貸", "保險"}
    }
    highest_category = max(variable_categories or spending_breakdown, key=(variable_categories or spending_breakdown).get)

    analysis = {
        "monthly_surplus": monthly_surplus,
        "savings_rate": round(savings_rate, 3),
        "emergency_fund_months": emergency_fund_months,
        "highest_category": highest_category,
        "essential_expense": essential_expense,
    }
    return {"analysis": analysis}


def retrieve_node(state: FinanceState) -> FinanceState:
    retriever = _get_retriever()
    profile = state["profile"]
    expanded_query = (
        f"{state['user_query']}\n"
        f"風險屬性：{profile['risk_level']}\n"
        f"目標：{'；'.join(profile['goals'])}\n"
        f"現金餘額：{profile['cash_balance']}\n"
        f"月結餘：{state['analysis']['monthly_surplus']}"
    )
    docs = retriever.search(expanded_query, top_k=4)
    return {
        "retrieved_docs": docs,
        "sources": [doc["title"] for doc in docs],
    }


def _render_context_docs(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "無"
    rendered = []
    for index, doc in enumerate(docs, start=1):
        rendered.append(f"[{index}] {doc['title']}\n{doc['content']}")
    return "\n\n".join(rendered)


def _heuristic_response(state: FinanceState) -> str:
    profile = state["profile"]
    analysis = state["analysis"]
    docs = state.get("retrieved_docs", [])
    suggestions: List[str] = []
    reasons: List[str] = []

    if state["intent"] in {"budgeting", "goal_planning"}:
        suggestions.append(f"先替 **{analysis['highest_category']}** 類別設定月上限，並把超支提醒做成固定通知。")
        reasons.append("若要在短時間內達成存錢目標，先管理波動較大的支出類別，效果通常最快。")

    if analysis["savings_rate"] < 0.2:
        suggestions.append("先把發薪日自動轉帳到儲蓄目標帳戶，讓存錢先發生。")
        reasons.append("你目前每月結餘偏緊，先提升儲蓄紀律，比直接追求高風險報酬更有效。")
    if analysis["emergency_fund_months"] < 3:
        suggestions.append("先補到至少 3 個月必要生活費的高流動性預備金。")
        reasons.append("目前可用現金對必要支出的覆蓋月數不足，流動性應優先於收益。")
    else:
        suggestions.append("將緊急預備金以外的短期閒置資金，分批放到高流動儲蓄或 3 個月定存。")
        reasons.append("你已有一定現金緩衝，可開始做短期資金效率管理。")

    if state["intent"] == "goal_planning":
        suggestions.append("把目標金額拆成每月固定存入值，並優先放在高流動性帳戶或自動存錢功能。")
        reasons.append("短期目標最重要的是可執行性與流動性，而不是追求較高波動報酬。")

    if state["intent"] == "insurance_review":
        suggestions.append("先進行家庭保障檢視，釐清保障缺口後再討論商品。")
        reasons.append("保險更適合先看責任與缺口，不是直接比價或直接推商品。")
    elif state["intent"] == "product_education" and profile["risk_level"] in {"balanced", "growth"}:
        suggestions.append("若目標在 3 年以上，可再評估穩健型定期定額方案。")
        reasons.append("你的風險屬性不是最保守，但仍需先確認目標期限與流動性。")
    elif state["intent"] == "product_education" and profile["risk_level"] == "conservative":
        suggestions.append("現階段先以高流動性儲蓄與目標管理為主，不建議直接從波動型商品開始。")
        reasons.append("你屬於保守型風險屬性，若沒有明確的長期目標與風險承受度確認，不宜直接進入中風險商品。")

    suggestions = suggestions[:3]
    reasons = reasons[:3]
    sources = [doc["title"] for doc in docs] or ["系統內建規則"]

    return (
        "### 財務觀察\n"
        f"- 你目前每月約有 **{analysis['monthly_surplus']:,} 元** 結餘，主要高支出類別是 **{analysis['highest_category']}**。\n"
        f"- 目前現金大約可支撐 **{analysis['emergency_fund_months']} 個月**必要支出。\n\n"
        "### 建議行動\n"
        + "\n".join(f"- {item}" for item in suggestions)
        + "\n\n### 為何這樣建議\n"
        + "\n".join(f"- {item}" for item in reasons)
        + "\n\n### 風險提醒\n"
        + "- 本回覆為 AI 助理示意，未使用真實銀行資料與正式 KYC 流程。\n"
        + "- 若涉及投資、保險或正式申辦，仍需經過適配性與人工覆核。\n\n"
        + "### 來源\n"
        + "- "
        + "\n- ".join(sources)
    )


def generate_node(state: FinanceState) -> FinanceState:
    llm = get_llm()
    if llm is None:
        return {"draft_response": _heuristic_response(state)}

    profile_summary = state["profile_summary"]
    analysis = state["analysis"]
    docs_block = _render_context_docs(state.get("retrieved_docs", []))
    user_prompt = f"""
客戶問題：{state['user_query']}

客戶摘要：
{profile_summary}

系統分析：
- 每月結餘：{analysis['monthly_surplus']:,} 元
- 儲蓄率：{analysis['savings_rate']:.1%}
- 緊急預備金月數：{analysis['emergency_fund_months']}
- 高支出類別：{analysis['highest_category']}

可用知識片段：
{docs_block}

請輸出專業、精簡、可執行的回覆，最多 400 字。
""".strip()

    response = llm.invoke([SystemMessage(content=PROMPT_SYSTEM), HumanMessage(content=user_prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    return {"draft_response": content if isinstance(content, str) else str(content)}


def review_node(state: FinanceState) -> FinanceState:
    profile = state["profile"]
    response = state["draft_response"].strip()
    warnings = list(state.get("warnings", []))

    unsuitable_phrase = None
    if profile["risk_level"] == "conservative" and "穩健型定期定額基金" in response:
        unsuitable_phrase = "保守型客戶不應直接推薦中風險商品，改以教育與引導為主。"
    if unsuitable_phrase:
        warnings.append(unsuitable_phrase)
        response += "\n\n### 補充提醒\n- 你目前屬於保守型風險屬性，若考慮投資型商品，應先完成正式適配性確認。"

    if "### 風險提醒" not in response:
        response += (
            "\n\n### 風險提醒\n"
            "- 本回覆為 AI 助理的初步建議。\n"
            "- 涉及投資、保險、貸款與正式申辦時，仍需正式流程與人工覆核。"
        )

    if "### 來源" not in response:
        response += "\n\n### 來源\n- " + "\n- ".join(state.get("sources", ["系統內建規則"]))

    return {"final_response": response, "warnings": warnings}


def finalize_node(state: FinanceState) -> FinanceState:
    if state.get("final_response"):
        return {
            "final_response": state["final_response"],
            "warnings": state.get("warnings", []),
            "sources": state.get("sources", []),
            "retrieved_docs": state.get("retrieved_docs", []),
            "analysis": state.get("analysis", {}),
        }

    return {
        "final_response": state.get("draft_response", "目前無法產生回覆。"),
        "warnings": state.get("warnings", []),
        "sources": state.get("sources", []),
        "retrieved_docs": state.get("retrieved_docs", []),
        "analysis": state.get("analysis", {}),
    }


def route_after_guardrail(state: FinanceState) -> str:
    return state.get("route", "continue")


def run_finance_copilot(profile_id: str, user_query: str) -> FinanceState:
    graph = _get_graph()
    result = graph.invoke({"profile_id": profile_id, "user_query": user_query})
    return result


def list_profiles() -> List[Dict[str, Any]]:
    return load_profiles()
