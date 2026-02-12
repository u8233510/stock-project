# stock-project

## `config.json` 設定範例

以下提供可直接使用的 `config.json` 範本（請把 API key 換成自己的）：

```json
{
  "storage": {
    "sqlite_path": "data/stock.db"
  },
  "universe": [
    {"stock_id": "2330", "name": "台積電"},
    {"stock_id": "2303", "name": "聯電"}
  ],
  "datasets": {
    "enabled": [
      "ohlcv",
      "per_pbr",
      "month_revenue",
      "financial_statements",
      "dividend",
      "industry_chain"
    ]
  },
  "llm": {
    "api_key": "",
    "model": "meta/llama-3.1-70b-instruct"
  },
  "search": {
    "provider": "google_then_rss",

    "google_api_key": "YOUR_GOOGLE_CSE_API_KEY",
    "google_cse_id": "YOUR_GOOGLE_CSE_ID",
    "google_daily_free_limit": 100,
    "google_queries_per_run": 2,

    "enable_ddg": false,

    "openrouter_api_key": "",
    "openrouter_model": "perplexity/sonar",
    "openrouter_site_url": "https://localhost",
    "openrouter_app_name": "stock-project",

    "puter_api_key": "",
    "puter_model": "perplexity/sonar",

    "perplexity_api_key": "",
    "perplexity_model": "sonar"
  }
}
```

---

## `search.provider` 如何選

- `google_then_rss`（預設建議）  
  Google CSE（有額度保護）+ Google News RSS + Wikipedia。
- `openrouter`  
  使用 OpenRouter 產生外部摘要（需 `openrouter_api_key`）。
- `google_then_ddg`  
  Google + DDG（僅在你想加 DDG 時）。
- `google`  
  只用 Google CSE（若額度到上限仍會顯示警告）。
- `perplexity`  
  使用 Perplexity API（需 key）。
- `puter`  
  使用 Puter 作為可選 provider（需 `puter_api_key`，建議先自行小流量測試）。

---

## 快速建議

1. 先用 `google_then_rss` 跑起來（免費且相對穩定）。
2. 有 OpenRouter key 再改成 `openrouter`；若你要試 Puter 可改成 `puter`。
3. 若 DDG 品質不穩，保持 `enable_ddg: false`。

---

## 為什麼 Perplexity「可免費用」，但程式仍建議走 API？

你說得對：Perplexity 網站有免費使用額度。  
但在這個專案情境（後端自動化抓資料）有幾個差異：

1. **網站免費 ≠ 後端可程式化呼叫**  
   網頁版免費通常給人手動互動，不保證可穩定用於後端批次流程。
2. **API 才有可控的請求/回應格式**  
   後端需要固定 JSON 結構、錯誤碼、超時處理與重試策略，API 比較可維運。
3. **可追蹤與配額管理**  
   需要記錄每次呼叫、錯誤、成本與限額，API 管理比較清楚。

### 不想用 Perplexity API 也可以

本專案已提供不用 Perplexity API 的路線：
- `google_then_rss`（預設）：Google CSE + Google News RSS + Wikipedia
- `google_then_ddg`：若你想額外加 DDG
- `openrouter`：若你有 OpenRouter key，可改用 OpenRouter（其中也可選 Perplexity 模型）

也就是說：**不是一定要 Perplexity API**，只是「若要用 Perplexity 作為後端 provider」，API 會是較穩定可控的做法。
