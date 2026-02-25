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

## 分點活動異常偵測指南

你有分點進出（含價格、買量、賣量）時，可用以下框架判斷是否異常。
下面把「六步驟」拆成可直接執行的版本（每步都包含目的、做法、輸出）。

### 步驟 1：定義異常類型（先定義問題，不然後面會失焦）

**目的**：先決定你要抓哪種異常，避免把所有波動都當成異常。

**做法**：把異常分為 4 類，後續都用這四類來對齊。
- 量異常：`net_vol = buy - sell`、`gross_vol = buy + sell` 明顯偏離歷史。
- 價異常：分點均價偏離收盤/VWAP 且方向一致。
- 集中度異常：少數分點成交佔比突然上升。
- 持續性異常：同一分點連續多日單邊買或賣。

**輸出**：一份「異常字典」（哪幾種異常、各自判準），供團隊統一使用。

### 步驟 2：先做日級特徵（把明細資料轉成可比較的指標）

**目的**：把逐筆或原始進出資料，整理成每天可評分的特徵。

**做法**：在「股票 × 分點 × 日期」層級彙總。
- `net_vol`、`gross_vol`、`buy_ratio`、`avg_price`
- `vol_share = gross_vol / 該股當日總量`
- `price_impact = (avg_price - close) / close`

**輸出**：每個分點每天一列特徵，可直接拿去算 z-score 或模型。

### 步驟 3：建立滾動基準（比較的是「自己過去」，不是別人）

**目的**：每個分點習性不同，要用自己的歷史行為當基準。

**做法**：用 20/60 日滾動視窗。
- `z_net_20`、`z_gross_20`、`z_price_impact_20`
- 可加 robust 指標（MAD/IQR）降低極端值干擾。

**輸出**：標準化後的異常程度（例如 z 值），方便跨日期比較。

### 步驟 4：三層偵測法（先可解釋，再補模型）

**目的**：兼顧可解釋性與涵蓋率，降低漏網與誤報。

**做法**：
1. 規則層：`abs(z_net_20)>=3`、`z_gross_20>=3`、`vol_share>=10~15%`。
2. 統計層：EWMA 或 median+MAD，讓最近資料權重更高。
3. 模型層：Isolation Forest / LOF 抓多變數複合異常。

**輸出**：每筆事件的「規則命中 + 統計分數 + 模型分數」。

### 步驟 5：做 anomaly score（0~100，方便排序與派工）

**目的**：把多種訊號合併成單一分數，讓你能每天排優先級。

**做法**：
- 權重範例：量能異常 40%、價格偏離 25%、集中度 20%、持續性 15%。
- 分級：`>=80` 重大、`60~79` 中度、`<60` 一般。

**輸出**：每日異常排行榜（先看分數最高的前 10~20 筆）。

### 步驟 6：交易前過濾（把「統計異常」變成「可交易訊號」）

**目的**：降低假訊號，避免因事件噪音誤判。

**做法**：
- 低流動性剔除。
- 重大事件日（財報/除權息）分開看。
- 一定要回測事件後 1/3/5/10 日報酬與勝率。

**輸出**：最終可追蹤清單（有交易意義的異常），而非純統計異常。

**實務建議**：
- 第 1 週先上線「規則層 + 分數排序」。
- 第 2~3 週再加模型層。
- 每月調一次閾值與權重（用最近 3~6 個月資料）。

### 可立即上線的兩個警示

- **主力進貨警示**：`z_net_20 > 3` + `vol_share > 10%` + 連續 2 日淨買超。
- **主力出貨警示**：`z_net_20 < -3` + `vol_share > 10%` + 均價顯著低於收盤。

先把這兩個警示做成每日清單，再逐步加入模型層，通常就能抓到 70% 以上值得人工複核的異常事件。


## 最後你會得到什麼結果？（白話版）

如果照上面六步驟做完，你每天會拿到 3 份可以直接用的輸出：

1. **異常分點清單（可排序）**
   - 內容：股票、日期、分點、`anomaly_score`、異常類型（量/價/集中/持續）。
   - 用途：先看前 10~20 筆最高分事件，快速聚焦。

2. **可交易追蹤清單（過濾後）**
   - 內容：從異常清單再過濾流動性、事件日、方向一致性後留下的名單。
   - 用途：這份才是盤前/盤中實際要追蹤的候選。

3. **策略驗證報表（每週更新）**
   - 內容：事件後 1/3/5/10 日的平均報酬、勝率、最大回撤。
   - 用途：判斷規則是否真的有 edge，決定要不要調整閾值。

### 你可以把它想成這樣

- 六步驟前半（1~4）：是在做「偵測器」。
- 第 5 步：是在做「優先級排序」。
- 第 6 步：是在做「交易可用性過濾」。

所以最後不是只得到一句「有異常」，而是會得到：
**「哪一檔、哪個分點、哪一天、異常多嚴重、要不要追蹤、過去這種訊號勝率多少」**。

這樣你每天就能用固定流程產出名單，不用靠主觀逐筆看盤。

## 程式碼：分點異常偵測模組

已提供可直接使用的程式碼：`utility/branch_anomaly_detection.py`

```python
import pandas as pd
from utility.branch_anomaly_detection import build_anomaly_outputs

raw_df = pd.read_csv("your_branch_data.csv")
# 需要欄位：stock_id, date, branch_id, price, buy, sell, close

anomaly_events, watchlist, weekly_report = build_anomaly_outputs(raw_df)

print(anomaly_events.head())
print(watchlist.head())
print(weekly_report.head())
```

輸出說明：
- `anomaly_events`：異常事件排行榜（含 anomaly_score）。
- `watchlist`：可交易追蹤清單（規則命中 + 分數門檻）。
- `weekly_report`：每週事件統計摘要。

### 與 `app.py` 整合方式

目前已整合到主選單：
- 進入 `🚨 分點異常偵測` 即可在介面中選擇股票、日期區間與門檻，直接產出：
  1. 異常事件排行榜
  2. 可交易追蹤清單
  3. 每週摘要

## AI 籌碼決策系統（點 + 面）

你要同時做到兩件事：
- **點（監控）**：持續追蹤「贏家分點」
- **面（挖掘）**：從海量明細資料自動找出可回測的新策略

這可落地成一個雙層架構：

### 第一層：影子跟單（自動化追蹤贏家分點）

重點不是誰買最多，而是「誰在對的時間買」。

1. **動態贏家評級（Winner Rating）**
   - 對每個分點做滾動回測（20/60/120 日）
   - 核心分數建議：
     ```text
     winner_rating = 0.30*hit_rate + 0.30*pnl_ratio + 0.20*sharpe + 0.20*timing_score
     ```
   - `hit_rate`：訊號後 N 日正報酬比率
   - `pnl_ratio`：平均獲利 / 平均虧損
   - `sharpe`：風險調整後績效
   - `timing_score`：買入位置是否接近波段起漲區

2. **贏家型態分群（短線/長線）**
   - 長線價值型：持有期長、回補慢、偏基本面事件
   - 短線交易型：週轉快、集中出手、事件日反應快
   - 作法：用持有天數、周轉率、回吐率做 clustering（KMeans/階層分群）

3. **多因子高優先級警示**
   - 例：
     - 高勝率長線分點「首次回補」
     - + 近期獲利了結分點再度回流
     - + 股價仍在成本帶上緣
   - 同時成立就觸發 A 級通知；單一條件成立僅 B/C 級觀察

4. **潛伏型贏家偵測（異常行為）**
   - 定義「低頻交易但高命中」分點
   - 若平常交易少、但每次進場都接近起漲點，標記為潛伏型名單

### 第二層：策略挖掘引擎（Feature Engineering + 回測）

AI 先提案，策略引擎再驗證，避免直接讓模型決定買賣。

1. **籌碼集中度特徵**
   - 以 Entropy / HHI 量化集中速度：
     ```text
     HHI = sum((branch_share_i)^2)
     Entropy = -sum(branch_share_i * log(branch_share_i))
     ```
   - 範例規則：10 日內 HHI 快速上升 + 價格波動收斂，視為蓄勢訊號

2. **拆單意圖分析（大單拆小單）**
   - 偵測連續小單節奏（例如固定時間間隔、固定小張數）
   - 特徵可用：單筆張數分布、下單間隔、自相關、同價位重複掛單
   - 若低位階出現高一致性拆單，列為高權重特徵

3. **關聯分點網絡（Network Analysis）**
   - 建立分點共現圖（同日同向買入、跨標的同步）
   - 觀察社群結構與核心節點（community / centrality）
   - 用來辨識可能的策略聯盟或共同操盤行為

4. **統一回測驗證（必要）**
   - 指標至少包含：年化報酬、Sharpe、MDD、勝率、卡瑪比
   - 必做：交易成本、滑價、流動性門檻
   - 必做：多市場狀態檢驗（多頭/空頭/盤整）

### 本專案落地對應（現有模組）

- `modules/branch_analysis.py`：產生分點日特徵（淨買超、連續性、成本帶）
- `modules/branch_anomaly.py`：潛伏型/異常行為偵測
- `utility/branch_anomaly_detection.py`：異常事件與觀察清單輸出
- `modules/prediction.py`：將「觸發規則 + 風險條件」轉成可讀的策略摘要

### 建議的每日批次流程（收盤後）

1. 更新日資料（分點、OHLCV、法人、券資）
2. 重算 Winner Rating 與分點分群標籤
3. 計算集中度/拆單/網絡特徵
4. 產生 A/B/C 級警示與候選策略
5. 對候選策略跑快速回測並更新策略排行榜

### MVP 建議時程（4 週）

- 第 1 週：完成 Winner Rating + 分點分群
- 第 2 週：完成 Entropy/HHI 與拆單偵測特徵
- 第 3 週：完成分點網絡特徵 + 候選規則生成
- 第 4 週：完成回測儀表板與紙上交易驗證

## 程式碼：贏家分點追蹤 + 策略挖掘工具

已提供可直接使用的程式碼：`utility/winner_branch_ai_system.py`

```python
import pandas as pd
from utility.winner_branch_ai_system import build_winner_branch_outputs

raw_df = pd.read_csv("your_branch_data.csv")
# 需要欄位：stock_id, date, branch_id, price, buy, sell, close

winner_rating, daily_alerts, concentration, strategy_candidates = build_winner_branch_outputs(raw_df)

print(winner_rating.head())
print(daily_alerts.head())
print(strategy_candidates.head())
```

輸出說明：
- `winner_rating`：分點勝率/盈虧比/Sharpe/時機分數整合後的贏家評級。
- `daily_alerts`：依 A/B/C 級輸出的每日影子跟單警示。
- `concentration`：含 `HHI`、`Entropy`、壓縮度等集中度特徵。
- `strategy_candidates`：由集中度 + 壓縮條件產生的候選策略事件。

### 這支 `winner_branch_ai_system.py` 掛在哪裡跑？

已整合在 `app.py` 側邊選單：
- 進入 **`🧠 AI 贏家分點追蹤`** 即可執行。

執行方式：
```bash
streamlit run app.py
```

流程：選股票與日期區間 → 點「執行贏家分點計算」→ 取得
1. Winner Rating
2. Daily Alerts（A/B/C）
3. Strategy Candidates（HHI/Entropy/壓縮）

### Q: 一定要用 AI 模型訓練嗎？

不一定，建議分兩階段：

- **第一階段（贏家導向）**：可先不訓練模型，先跑規則與績效追蹤
  - 分點資料聚合
  - 分點績效回溯（勝率/盈虧比/Sharpe）
  - Dashboard 每日掃描「頂級贏家」今日買入標的

- **第二階段（模型導向）**：當你要擴充到策略挖掘，再做監督式學習
  - 標註正樣本（例如未來 20 日曾上漲超過 8%）
  - 以 `hhi / entropy / avg_buy_cost / buy_continuity / retail_exit_ratio` 等特徵訓練 XGBoost
  - 做持有天數、停損比例參數掃描（proxy backtest）

本專案已提供第二階段工具：`utility/winner_branch_ml.py`，並整合在 `🧠 AI 贏家分點追蹤` 頁面中可直接產生訓練資料與嘗試訓練。

### 欄位中文化

`🧠 AI 贏家分點追蹤` 頁面目前已將第一階段與第二階段輸出欄位顯示為中文（含下載 CSV 欄位）。
