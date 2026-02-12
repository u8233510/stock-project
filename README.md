# stock-project

## 直接貼上的 `config.json`（X1/X2/X3 + R1）

```json
{
  "storage": {
    "sqlite_path": "data/stock.db"
  },
  "universe": [
    {"stock_id": "2330", "name": "台積電"},
    {"stock_id": "2317", "name": "鴻海"},
    {"stock_id": "2454", "name": "聯發科"}
  ],
  "datasets": {
    "enabled": [
      "ohlcv",
      "institutional",
      "branch",
      "margin_short",
      "day_trading",
      "holding_shares",
      "securities_lending",
      "per_pbr",
      "month_revenue",
      "financial_statements",
      "dividend",
      "industry_chain",
      "market_value"
    ]
  },
  "llm": {
    "api_key": "YOUR_NVIDIA_API_KEY",
    "models": {
      "fundamental": "meta/llama-3.3-70b-instruct",
      "branch": "deepseek-ai/deepseek-v3.2",
      "chip": "meta/llama-3.3-70b-instruct",
      "prediction": "meta/llama-3.1-8b-instruct"
    },
    "rag": {
      "enabled": true,
      "embedding_model": "nvidia/nv-embed-v1",
      "top_k": 5
    }
  }
}
```

## 這份設定對應你的問題（X1 / X2 / X3）

- 基本面分析：`llm.models.fundamental`（X1）
- 分點分析：`llm.models.branch`（X2）
- 全方位籌碼診斷：`llm.models.chip`（X3）
- 公司資料搜尋/RAG：`llm.rag.embedding_model`（R1）

> 若某一欄沒填，系統會回退到舊版 `llm.model`（單模型模式）。

## CMD 模式如何確認模型可用（Windows）

### 0) 先設定 API Key（CMD）

```bat
set NVIDIA_API_KEY=YOUR_NVIDIA_API_KEY
```

### 1) 先確認你的 Key 看得到模型清單

```bat
curl -X GET "https://integrate.api.nvidia.com/v1/models" ^
  -H "Authorization: Bearer %NVIDIA_API_KEY%" ^
  -H "accept: application/json"
```

看到 `"data":[{"id":"..."}]` 代表 Key 正常。

### 2) 單一模型健康檢查（Chat）

把 `MODEL_ID` 換成你要測的模型，例如 `meta/llama-3.3-70b-instruct`：

```bat
set MODEL_ID=meta/llama-3.3-70b-instruct
echo {"model":"%MODEL_ID%","messages":[{"role":"user","content":"請只回覆 OK"}],"temperature":0,"max_tokens":16} > nim_chat_test.json
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" ^
  -H "Authorization: Bearer %NVIDIA_API_KEY%" ^
  -H "Content-Type: application/json" ^
  --data @nim_chat_test.json
```

回傳有 `choices[0].message.content` 就代表該模型可用。

### 3) 針對 RAG 的 embedding 模型檢查

```bat
set EMBED_MODEL=nvidia/nv-embed-v1
echo {"model":"%EMBED_MODEL%","input":"台積電是做什麼的？"} > nim_embed_test.json
curl -X POST "https://integrate.api.nvidia.com/v1/embeddings" ^
  -H "Authorization: Bearer %NVIDIA_API_KEY%" ^
  -H "Content-Type: application/json" ^
  --data @nim_embed_test.json
```

回傳有 `data[0].embedding` 就代表 embedding 模型可用。

### 4) 快速對照到本專案（X1/X2/X3 + R1）

- X1 基本面：先測 `llm.models.fundamental`
- X2 分點：先測 `llm.models.branch`
- X3 籌碼：先測 `llm.models.chip`
- R1 檢索：先測 `llm.rag.embedding_model`

只要上述各模型都能回應，你的 `config.json` 對應流程就能正常跑。

## 三層分析流程（你的目前需求）

1. **第一層：基本面分析（先過濾）**
   - 看營收趨勢、毛利率、ROE、負債比等。
   - 先刪掉體質不穩定標的，留下觀察池。

2. **第二層：分點分析（看資金行為）**
   - 看主力分點買賣超連續性、成本帶、是否異常集中。
   - 確認「誰在買、是否持續買、買在什麼位置」。

3. **第三層：全方位籌碼診斷（做風險評級）**
   - 整合融資融券、三大法人、量能結構與分點訊號。
   - 目的不是只決定買不買，而是決定倉位和節奏。

## 你列出的 NVIDIA 可用模型「有沒有幫助」？

有幫助，而且很重要；只是**不是全部都適合你現在的三層分析**。

### 目前最適合你的（優先）

- `meta/llama-3.3-70b-instruct`：三層流程通用主模型（穩定、泛用）。
- `deepseek-ai/deepseek-v3.2`：推理整合能力強，適合關鍵日重跑。
- `meta/llama-3.1-8b-instruct`：低成本日常批次。
- `nvidia/nv-embed-v1`、`baai/bge-m3`：做公司資料檢索/RAG 的 embedding。

### 你之前看到的「較完整候選清單」也有用（但屬候選，不是必選）

- 分析模型候選：`mistralai/mistral-large`、`mistralai/mistral-large-2-instruct`、`nvidia/llama-3.3-nemotron-super-49b-v1.5`、`qwen/qwen3-next-80b-a3b-instruct`。
- 低成本 baseline 候選：`meta/llama-3.1-8b-instruct`、`mistralai/mistral-7b-instruct-v0.3`。
- 檢索 embedding 候選：`baai/bge-m3`、`nvidia/nv-embed-v1`、`nvidia/nv-embedqa-e5-v5`、`snowflake/arctic-embed-l`。

> 原則：先固定 1~2 個主模型跑穩，再拿候選模型做 A/B 測試；不是把清單全部一起上。

### 先不用投入的（目前情境）

- 多數 `vision` / `multimodal` 模型：你目前是文字型分析流程。
- `code` 專用模型：除非你要大量生成程式碼策略。
- `guard` / safety 模型：偏內容安全治理，不是主要分析模型。

### 你的實際配置建議

- 日常：`meta/llama-3.1-8b-instruct`
- 每週深度 + 關鍵事件：`meta/llama-3.3-70b-instruct` 或 `deepseek-ai/deepseek-v3.2`
- 若做公司資料查詢：加 `nvidia/nv-embed-v1`（或 `baai/bge-m3`）

## 你上面列的那些（OpenRouter / Perplexity / DDG）現在不用嗎？

不用。你目前這版設定是 **NVIDIA-only**，所以只要維護以下四塊：

- `storage`
- `universe`
- `datasets.enabled`
- `llm`（`api_key` + `model`）

以下項目在你目前流程都可以不設定、不使用：

- `openrouter_api_key`
- `perplexity_api_key`
- `search.provider`
- `openrouter_then_rss` / `openrouter_then_ddg` / `ddg` 相關流程

## 一個模型用全部，還是分模型？

### 建議起步：先用一個模型全包

- 模型：`meta/llama-3.3-70b-instruct`
- 適合你現在：先把三層流程跑穩，流程簡單、維護最少。

### 成本優化（第二階段）

- 每日批次：`meta/llama-3.1-8b-instruct`
- 關鍵日/重點個股：`meta/llama-3.3-70b-instruct` 或 `deepseek-ai/deepseek-v3.2`

### 如果要做公司資料搜尋（RAG）

- Embedding：`nvidia/nv-embed-v1` 或 `baai/bge-m3`
- 生成回答：`meta/llama-3.3-70b-instruct` 或 `deepseek-ai/deepseek-v3.2`

## 免費 / 低成本方案（以你目前只有 NVIDIA key）

1. 先固定用小模型跑日常：`meta/llama-3.1-8b-instruct`
2. 每週只對觀察池做一次大模型深度分析。
3. 只有在「訊號衝突 / 財報公布 / 法說後」才升級大模型重跑。

## 最終建議（直接照做）

- 第 1~2 週：`meta/llama-3.3-70b-instruct` 一個模型跑全部模組。
- 第 3 週起：改成雙模型（8B 日常 + 70B/DeepSeek 關鍵重跑）。
- `universe` 先放 3~10 檔，穩定後再擴大。
