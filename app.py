from __future__ import annotations

import streamlit as st

from workflow import list_profiles, run_finance_copilot


st.set_page_config(page_title="CUBE Copilot Prototype", page_icon="💬", layout="wide")

st.title("CUBE Copilot｜生成式 AI 財務健康管家 Prototype")
st.caption("Demo 版：使用合成資料與示意商品，不連接真實銀行系統。")

profiles = list_profiles()
profile_options = {item["display_name"]: item for item in profiles}

sample_prompts = {
    "Amy｜28歲上班族": "我想在 6 個月內存到 10 萬，但最近餐飲和購物有點超支，該怎麼安排？",
    "Brian｜36歲雙薪家庭": "我有房貸跟小孩教育金需求，現在應該先加碼基金，還是先做保障檢視？",
    "Claire｜43歲高資產客戶": "我手上現金不少，想提高資金效率，但又不想影響流動性，可以怎麼分配？",
}

left, right = st.columns([1, 1.6], gap="large")

with left:
    selected_name = st.selectbox("選擇示例客戶", list(profile_options.keys()))
    profile = profile_options[selected_name]

    st.subheader("客戶概況")
    c1, c2 = st.columns(2)
    c1.metric("月收入", f"{profile['monthly_income']:,} 元")
    c2.metric("月支出", f"{profile['monthly_expense']:,} 元")
    c3, c4 = st.columns(2)
    c3.metric("現金餘額", f"{profile['cash_balance']:,} 元")
    c4.metric("投資餘額", f"{profile['investment_balance']:,} 元")
    st.write(f"**風險屬性：** {profile['risk_level']}")
    st.write(f"**保障現況：** {profile['insurance_status']}")
    st.write(f"**目標：** {'；'.join(profile['goals'])}")
    st.write(f"**備註：** {profile['notes']}")

    st.subheader("支出結構")
    st.json(profile["spending_breakdown"], expanded=False)

with right:
    st.subheader("請輸入客戶問題")
    question = st.text_area(
        "問題",
        value=sample_prompts[selected_name],
        height=140,
        label_visibility="collapsed",
        placeholder="例如：我這個月是不是花太多？我適合先做定存還是自動存錢？",
    )

    run_clicked = st.button("產生建議", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("AI 分析中..."):
            result = run_finance_copilot(profile_id=profile["id"], user_query=question)

        st.subheader("AI 回覆")
        st.markdown(result["final_response"])

        warnings = result.get("warnings", [])
        if warnings:
            st.warning("\n".join(warnings))

        sources = result.get("sources", [])
        if sources:
            st.caption("檢索來源：" + " / ".join(sources))

        with st.expander("查看檢索到的知識片段"):
            docs = result.get("retrieved_docs", [])
            if not docs:
                st.write("沒有檢索到外部片段，回覆使用系統內建規則。")
            else:
                for doc in docs:
                    st.markdown(f"**{doc['title']}**  | score={doc['score']}")
                    st.write(doc["content"])
                    st.divider()

st.divider()
st.markdown(
    """
**Demo 建議流程（2 分鐘）**
1. 選擇 Amy 範例客戶。
2. 用預設問題示範「支出分析 + 存錢目標」場景。
3. 再問一題：「我適合直接買基金嗎？」展示適配性提醒與風險控管。
4. 最後點開知識片段，說明 Prototype 採用 RAG + Guardrail + Human handoff 的設計。
"""
)
