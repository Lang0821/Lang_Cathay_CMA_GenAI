# CUBE Copilot Prototype

CUBE Copilot是一個面向金融客戶的「生成式AI財務健康管家」示範版。

這個prototype的設計目標，不是直接取代理專或保險顧問，而是把原本分散的帳務資訊、商品知識與 FAQ，轉成客戶看得懂、願意採取行動的對話式建議，並在高風險場景下保留適配性檢核與真人轉接。

## 1. 為什麼選這個場景

金融客戶最常見的問題不是「找不到功能」，而是：
- 看得到帳務，但不知道下一步要做什麼。
- 想存錢、理財或補保障，卻不清楚先後順序。
- 面對金融商品說明時，理解成本高，容易延後決策。

因此，我把場景定義成：

> 以 GenAI 幫客戶做「財務健康分析 + 下一步建議 + 商品教育 + 真人轉接」。

## 2. Prototype 功能

本 demo 提供四個能力：

1. **財務健康摘要**：根據客戶收入、支出、現金、投資與保險狀態，生成簡短分析。
2. **目標導向建議**：例如 6 個月存到 10 萬、教育金規劃、現金管理。
3. **商品教育與適配性提醒**：用合成商品資料示範如何推薦下一步，但不直接做高風險銷售。
4. **Guardrail / Handoff**：當問題涉及敏感資料或高風險決策時，觸發提醒或建議轉接真人。

## 3. 技術設計

### 架構

- **前端**：Streamlit
- **流程編排**：LangGraph
- **LLM**：OpenAI API（可透過 `langchain-openai` 接入）
- **檢索**：輕量版 TF-IDF Retriever（demo 用）
- **資料**：本地 JSON 合成資料

### LangGraph Workflow

```text
load_profile
  -> guardrail
    -> analyze
      -> retrieve
        -> generate
          -> review
            -> finalize
```

### 每個節點做什麼

- `load_profile`：載入示例客戶資料並做摘要。
- `guardrail`：攔截敏感資料、標註投資/保險風險提醒。
- `analyze`：計算月結餘、儲蓄率、緊急預備金月數、最高支出類別。
- `retrieve`：從知識庫與示意商品說明檢索相關資訊。
- `generate`：透過 LLM 產生自然語言建議。
- `review`：補上適配性、風險與來源說明。
- `finalize`：輸出最終回覆。

## 4. 專案結構

```text
prototype/
├── app.py
├── workflow.py
├── retriever.py
├── llm_client.py
├── requirements.txt
├── .env.example
├── README.md
└── data/
    ├── customer_profiles.json
    ├── product_catalog.json
    └── knowledge_base.json
```

## 5. 安裝與執行

### Step 1. 建立虛擬環境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 請改用 .venv\Scripts\activate
```

### Step 2. 安裝套件

```bash
pip install -r requirements.txt
```

### Step 3. 設定環境變數

```bash
cp .env.example .env
```

請將 `.env` 中的 `OPENAI_API_KEY` 改成你的金鑰，並視需要調整 `OPENAI_MODEL`。

> 若未提供 API Key，系統會自動切換成 heuristic fallback 模式，仍可完整跑 demo 流程，但不會呼叫 LLM API。

### Step 4. 啟動 Demo

```bash
streamlit run app.py
```

若現場只想快速用終端機展示，也可執行：

```bash
python demo_cli.py --profile young_professional --question "我想在 6 個月內存到 10 萬，但最近餐飲和購物有點超支，該怎麼安排？"
```

## 6. 2 分鐘 Demo 建議腳本

### Demo 1：支出分析與存錢

選擇 `Amy｜28歲上班族`，輸入：

```text
我想在 6 個月內存到 10 萬，但最近餐飲和購物有點超支，該怎麼安排？
```

預期看到：
- 月結餘與高支出類別分析
- 自動存錢 / 高流動性預備金 / 短期定存的順序建議
- 風險提醒與檢索來源

### Demo 2：適配性提醒

再輸入：

```text
我適合直接買基金嗎？
```

預期看到：
- 先做風險屬性與目標期限判斷
- 若客戶偏保守，系統不直接硬推中風險商品
- 顯示「此為 AI 初步建議，仍需正式適配性確認」

## 7. 這個 Prototype 對應商業提案的哪一段

這份 prototype 對應到簡報中「CUBE Copilot 財務健康管家」的 MVP：

- **對客價值**：把複雜資料翻成下一步行動。
- **商業價值**：提高 app 互動率、產品導流與低價值客服分流。
- **技術價值**：驗證 RAG、Guardrail、Explainability、Human handoff 能否在金融場景協同運作。

## 8. 限制與下一步

### 目前限制

- 使用合成資料，不連接真實客戶資料。
- 使用輕量檢索器，不是正式企業級向量檢索。
- 未串接真實交易、KYC、產品申辦、CRM 與審計日誌。

### 下一步

1. 接真實但去識別化的交易與產品資料。
2. 導入企業級向量資料庫與權限控管。
3. 加入線上評測儀表板（命中率、幻覺率、轉換率、客服轉接率）。
4. 把真人轉接摘要整合到 RM / 客服工作台。

## 9. 安全聲明

- 本專案僅供面試 demo 使用。
- 所有客戶資料、商品資料與規則皆為合成或示意內容。
- 不應直接用於真實投資、保險、貸款或交易決策。
