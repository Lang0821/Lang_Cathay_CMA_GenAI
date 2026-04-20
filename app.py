from __future__ import annotations

import streamlit as st

from workflow import list_profiles, run_finance_copilot


st.set_page_config(page_title="CUBE Copilot", page_icon="💬", layout="wide")

st.title("CUBE Copilot｜生成式 AI 財務健康管家")

profiles = list_profiles()
profile_options = {item["display_name"]: item for item in profiles}

sample_prompts = {
    "Amy｜28歲上班族": "我想在 6 個月內存到 10 萬，但最近餐飲和購物有點超支，該怎麼安排？",
    "Brian｜36歲雙薪家庭": "我有房貸跟小孩教育金需求，現在應該先加碼基金，還是先做保障檢視？",
    "Claire｜43歲高資產客戶": "我手上現金不少，想提高資金效率，但又不想影響流動性，可以怎麼分配？",
}

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("示例客戶")
    selected_name = st.selectbox(
        "示例客戶",
        list(profile_options.keys()),
        label_visibility="collapsed",
    )
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
    st.write(f"**財務目標：** {'；'.join(profile['goals'])}")
    st.write(f"**備註：** {profile['notes']}")

    st.subheader("支出結構")
    breakdown = profile["spending_breakdown"]
    for k, v in breakdown.items():
        st.write(f"- {k}：{v:,} 元")

with right:
    st.subheader("問題輸入")
    question = st.text_area(
        "問題",
        value=sample_prompts[selected_name],
        height=140,
        label_visibility="collapsed",
        placeholder="例如：我想在半年內存到 10 萬，該怎麼安排？",
    )

    run_clicked = st.button("開始分析", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("分析中..."):
            result = run_finance_copilot(profile_id=profile["id"], user_query=question)

        st.subheader("分析結果")
        st.markdown(result["final_response"])

        warnings = result.get("warnings", [])
        if warnings:
            st.warning("\n".join(warnings))

        sources = result.get("sources", [])
        if sources:
            st.caption("資料來源：" + " / ".join(sources))

        with st.expander("參考依據"):
            docs = result.get("retrieved_docs", [])
            if not docs:
                st.write("本次回覆未引用外部知識片段。")
            else:
                for doc in docs:
                    st.markdown(f"**{doc['title']}**")
                    st.write(doc["content"])
                    st.divider()