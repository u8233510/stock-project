import streamlit as st
import pandas as pd
import database
import requests
from duckduckgo_search import DDGS



def _normalize_secret(value):
    """去除常見貼上污染（空白/換行/BOM）。"""
    if value is None:
        return ""
    return str(value).replace("﻿", "").strip()


def _google_error_hint(err_msg):
    """將 Google API 常見錯誤轉成可操作建議。"""
    msg = (err_msg or "").lower()
    if "api key not valid" in msg or "invalid" in msg:
        return "請確認：1) API key 屬於同一個 GCP 專案；2) 已啟用 Custom Search JSON API；3) key 沒有被 HTTP referrer/IP 限制擋住此執行環境。"
    if "referer" in msg or "ip" in msg or "not allowed" in msg:
        return "此金鑰限制不符（HTTP referrer/IP）。若在本機 Python 後端呼叫，請移除 referrer 限制或改用允許該來源的金鑰。"
    if "quota" in msg or "rate limit" in msg:
        return "已達配額上限，請檢查 GCP Quotas/計費設定。"
    if "access not configured" in msg or "has not been used" in msg:
        return "尚未啟用 Custom Search JSON API，請到 Google Cloud Console 啟用後再試。"
    return "請檢查 API key 是否正確、Custom Search JSON API 是否啟用、以及 key 限制是否允許目前執行環境。"

PUTER_JS_SNIPPET = """<script src="https://js.puter.com/v2/"></script>
<script>
async function runPuterDemo() {
  try {
    const response = await puter.ai.chat(
      "量子運算的最新進展是什麼？",
      { model: "perplexity/sonar" }
    );
    console.log(response);
  } catch (err) {
    console.error("Puter 呼叫失敗:", err);
  }
}
runPuterDemo();
</script>
"""


def _google_search(query, cfg, max_results=5):
    """透過 Google Custom Search 取得搜尋摘要。"""
    search_cfg = cfg.get("search", {})
    api_key = _normalize_secret(search_cfg.get("google_api_key"))
    cse_id = _normalize_secret(search_cfg.get("google_cse_id"))
    if not api_key or not cse_id:
        return [], "Google 未設定 google_api_key 或 google_cse_id。"

    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": min(max_results, 10),
                "hl": "zh-TW"
            },
            timeout=20,
        )
        data = resp.json()
        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
            hint = _google_error_hint(err_msg)
            return [], f"Google 搜尋失敗：{err_msg}｜建議：{hint}"

        items = data.get("items", []) if isinstance(data, dict) else []
        records = [
            {
                "source": "Google",
                "title": i.get("title", ""),
                "snippet": i.get("snippet", ""),
                "url": i.get("link", ""),
            }
            for i in items
        ]
        if not records:
            return [], "Google 已連線，但此查詢無結果。"
        return records, None
    except Exception as exc:
        return [], f"Google 搜尋例外：{str(exc)}"


def _perplexity_search(query, cfg):
    """透過 Perplexity API 取得外部資訊摘要。"""
    search_cfg = cfg.get("search", {})
    api_key = _normalize_secret(search_cfg.get("perplexity_api_key"))
    model = search_cfg.get("perplexity_model", "sonar")
    if not api_key:
        return [], "Perplexity 未設定 perplexity_api_key。"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是研究助理，請根據最新網路公開資訊整理重點，並附上來源網址。",
            },
            {
                "role": "user",
                "content": f"請針對以下主題整理 5 點重點，格式為一行一點，且每點附來源網址：{query}",
            },
        ],
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
            return [], f"Perplexity 搜尋失敗：{err_msg}"

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(data, dict) else ""
        if not content:
            return [], "Perplexity 已連線，但無回傳內容。"
        return [{"source": "Perplexity", "title": "摘要", "snippet": content, "url": ""}], None
    except Exception as exc:
        return [], f"Perplexity 搜尋例外：{str(exc)}"


def _ddg_search(query, max_results=5, source="DuckDuckGo"):
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "source": source,
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                )
            if not results:
                return [], f"{source} 查詢無結果。"
            return results, None
    except Exception as exc:
        return [], f"{source} 搜尋例外：{str(exc)}"


def _build_external_context(stock_name, sid, cfg):
    """蒐集外部資訊（Perplexity / Google / DDG / 社群網站搜尋）。"""
    base_query = f"{stock_name} {sid} 核心產品 產業地位 最新新聞"

    records = []
    warnings = []

    per_records, per_warn = _perplexity_search(base_query, cfg)
    records.extend(per_records)
    if per_warn:
        warnings.append(per_warn)

    google_records, google_warn = _google_search(base_query, cfg, max_results=5)
    records.extend(google_records)
    if google_warn:
        warnings.append(google_warn)

    ddg_records, ddg_warn = _ddg_search(base_query, max_results=5, source="DuckDuckGo")
    records.extend(ddg_records)
    if ddg_warn:
        warnings.append(ddg_warn)

    # 社群/輿情（以公開可索引頁面為主，非登入資料）
    social_queries = [
        (f"site:x.com OR site:twitter.com {stock_name} {sid}", "X/Twitter"),
        (f"site:facebook.com {stock_name} {sid}", "Facebook"),
        (f"site:instagram.com {stock_name} {sid}", "Instagram"),
    ]
    for query, source in social_queries:
        social_records, social_warn = _ddg_search(query, max_results=3, source=source)
        records.extend(social_records)
        if social_warn:
            warnings.append(social_warn)

    if not records:
        warnings.insert(0, "目前未取得外部來源。請先檢查下方各來源診斷訊息。")
        return "", warnings

    source_counts = {}
    for rec in records:
        src = rec.get("source", "來源")
        source_counts[src] = source_counts.get(src, 0) + 1
    summary = "、".join([f"{k}:{v}" for k, v in source_counts.items()])
    warnings.insert(0, f"外部來源抓取成功（{summary}）。")

    lines = []
    for rec in records[:20]:
        url = rec.get("url", "")
        url_text = f"（{url}）" if url else ""
        lines.append(f"- [{rec.get('source', '來源')}] {rec.get('title', '')}: {rec.get('snippet', '')} {url_text}")
    return "\n".join(lines), warnings


def _build_fundamental_prompt(stock_name, sid, search_ctx, metrics):
    """建立固定章節格式的基本面分析 Prompt。"""
    return f"""
請你扮演台股資深基本面分析師，針對 {stock_name}（{sid}）撰寫報告。

【重要規則】
1) 嚴格使用以下固定格式與標題順序，不要增減章節。
2) 若資料不足，請明確寫「未提供」或「資料不足」，禁止虛構。
3) 所有數值盡量引用我提供的資料；若引用新聞，僅能使用「搜尋事實摘要」。
4) 以繁體中文輸出。

【固定輸出格式】
## 公司簡介
（公司定位、核心產品/服務、主要市場）

## 財務分析
（整體獲利能力與近況，2-4 句）

### 財務指標
- EPS：
- ROE（股東權益報酬率）：
- ROA（資產報酬率）：
- 營收成長率：
- 毛利率：

## 營收分析
（營收趨勢、可能驅動因子）

## 毛利率分析
（毛利率水準與可能原因；若無資料請寫未提供）

## 現金流量分析
（現金流量狀況與品質；若無資料請寫未提供）

## 投資評價
- 短期評價：
- 中期評價：
- 長期評價：
- 目標價格：

## 風險評估
- 市場風險：
- 財務風險：
- 法規/政策風險：

## 結論
（總結投資觀點與關鍵追蹤指標）

【可用資料】
- 搜尋事實摘要：{search_ctx if search_ctx else '未提供'}
- 最新季度 EPS：{metrics.get('latest_eps', '未提供')}
- 上季 EPS：{metrics.get('prev_eps', '未提供')}
- 近 12 月最新營收（億元）：{metrics.get('latest_revenue', '未提供')}
- 近 12 月最舊營收（億元）：{metrics.get('oldest_revenue', '未提供')}
- 估算營收成長率（最新 vs 最舊）：{metrics.get('revenue_growth', '未提供')}
""".strip()

# 1. 核心 AI 呼叫工具 (保持穩定，未更動)
def _call_nim_fundamental(cfg, prompt):
    llm_cfg = cfg.get("llm", {})
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_cfg.get('api_key')}", "Content-Type": "application/json"}
    payload = {
        "model": llm_cfg.get("model"),
        "messages": [
            {"role": "system", "content": "你是一位專業的證券分析師。請優先參考連網搜尋到的事實，結合財務數據給出具體的投資評價，嚴禁虛構公司業務。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.json()["choices"][0]["message"]["content"]

def show_fundamental_analysis():
    st.markdown("### 💎 基本面數據全覽與 AI 診斷")
    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}
    
    selected_stock = st.selectbox("請選擇分析標的", list(stock_options.keys()))
    sid = stock_options[selected_stock]
    
    # 統一基準日
    def_s = pd.to_datetime("today") - pd.Timedelta(days=60)
    def_e = pd.to_datetime("today")
    f_range = st.date_input("數據觀察基準日", value=[def_s, def_e])
    end_d = f_range[1] if len(f_range) == 2 else def_e

    st.divider()

    # --- 1. 數據抓取與格式化 ---
    
    # (1) 每月營收：除以 1000 並加上千分位 ","
    rev_query = f"SELECT date as '日期', revenue FROM stock_month_revenue_monthly WHERE stock_id='{sid}' ORDER BY date DESC LIMIT 12"
    rev_df = pd.read_sql(rev_query, conn)
    if not rev_df.empty:
        # ✅ 修正關鍵：統一欄位名稱為 '營收(億)'，並加上千分位
        rev_df['營收(億)'] = (rev_df['revenue'] / 100000000).apply(lambda x: f"{x:,.2f}")
        rev_df = rev_df[['日期', '營收(億)']]
    
    # (2+4) 每季獲利與 EPS：安全轉置處理
    profit_raw = pd.read_sql(f"SELECT date, type, value FROM stock_financial_statements WHERE stock_id='{sid}' AND type IN ('EPS', 'Net Profit') ORDER BY date DESC LIMIT 16", conn)
    if not profit_raw.empty:
        profit_df = profit_raw.pivot(index='date', columns='type', values='value').reset_index()
        # 安全重命名
        rename_map = {'date': '季度', 'EPS': 'EPS', 'Net Profit': '每季獲利'}
        profit_df = profit_df.rename(columns={k: v for k, v in rename_map.items() if k in profit_df.columns})
        # 加上千分位 (每季獲利)
        if '每季獲利' in profit_df.columns:
            profit_df['每季獲利'] = profit_df['每季獲利'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
    else:
        profit_df = pd.DataFrame()

    # (3+5) 本益比與殖利率
    val_query = f"SELECT date as '日期', PER as '本益比', dividend_yield as '殖利率%' FROM stock_per_pbr_daily WHERE stock_id='{sid}' AND date <= '{end_d}' ORDER BY date DESC LIMIT 10"
    valuation_df = pd.read_sql(val_query, conn)

    # (6+7) 現金股利與股息分配
    div_query = f"SELECT year as '年份', CashEarningsDistribution as '現金股利', StockEarningsDistribution as '股票股利' FROM stock_dividend WHERE stock_id='{sid}' ORDER BY year DESC LIMIT 5"
    div_df = pd.read_sql(div_query, conn)

    # --- 2. 表格化顯示分頁 ---
    tab1, tab2, tab3 = st.tabs(["📈 營收與獲利詳情", "💰 股利與估值看板", "🔍 AI 聯網趨勢報告"])

    with tab1:
        st.write("#### 1. 每月營收 (單位：百萬元)")
        st.dataframe(rev_df, use_container_width=True, hide_index=True)
        
        st.write("#### 2 & 4. 每季獲利與 EPS 歷程")
        if not profit_df.empty:
            # 確保僅顯示現有欄位
            cols_to_show = [c for c in ['季度', '每季獲利', 'EPS'] if c in profit_df.columns]
            st.dataframe(profit_df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("尚無獲利數據。")

    with tab2:
        st.write("#### 3 & 5. 本益比與殖利率變動")
        st.dataframe(valuation_df, use_container_width=True, hide_index=True)
        
        st.write("#### 6 & 7. 歷年股利分配 (現金與股票)")
        if not div_df.empty:
            st.table(div_df)
        else:
            st.info("尚無股利歷史數據。")

    with tab3:
        st.info("💡 外部資訊來源說明：目前後端已整合 Google / Perplexity / DDG / 社群公開頁面，並顯示各來源診斷訊息。若 Google 回報 key 無效，通常是 API 啟用或金鑰限制問題。")
        with st.expander("Puter.js 免 API Key 使用方式（前端範例）", expanded=False):
            st.markdown("可以直接這樣寫，但建議用 `async/await + try/catch`（如下）較容易除錯。")
            st.code(PUTER_JS_SNIPPET, language="html")
            st.markdown("支援模型示例：`perplexity/sonar`、`perplexity/sonar-pro`、`perplexity/sonar-deep-research`、`perplexity/sonar-reasoning-pro`。")

        # ✅ 保留聯網搜尋邏輯
        if st.button(f"🚀 啟動 {selected_stock} 聯網事實分析", use_container_width=True):
            with st.spinner("正在搜尋最新產業地位與市場新聞..."):
                search_ctx, search_warnings = _build_external_context(selected_stock, sid, cfg)
                if search_warnings:
                    for w in search_warnings:
                        if w.startswith("外部來源抓取成功"):
                            st.success(w)
                        else:
                            st.warning(w)
                
                # 獲取 AI 參考數據
                latest_eps = profit_df['EPS'].iloc[0] if ('EPS' in profit_df.columns and not profit_df.empty) else "未提供"
                prev_eps = profit_df['EPS'].iloc[1] if ('EPS' in profit_df.columns and len(profit_df) > 1) else "未提供"

                latest_revenue = rev_df['營收(億)'].iloc[0] if not rev_df.empty else "未提供"
                oldest_revenue = rev_df['營收(億)'].iloc[-1] if not rev_df.empty else "未提供"

                revenue_growth = "未提供"
                if not rev_df.empty and len(rev_df) > 1:
                    rev_num = pd.to_numeric(rev_df['營收(億)'].str.replace(',', ''), errors='coerce')
                    latest_rev_num = rev_num.iloc[0]
                    oldest_rev_num = rev_num.iloc[-1]
                    if pd.notnull(latest_rev_num) and pd.notnull(oldest_rev_num) and oldest_rev_num != 0:
                        revenue_growth = f"{((latest_rev_num - oldest_rev_num) / oldest_rev_num) * 100:.2f}%"

                metrics = {
                    "latest_eps": latest_eps,
                    "prev_eps": prev_eps,
                    "latest_revenue": latest_revenue,
                    "oldest_revenue": oldest_revenue,
                    "revenue_growth": revenue_growth
                }

                prompt = _build_fundamental_prompt(selected_stock, sid, search_ctx, metrics)
                st.markdown(_call_nim_fundamental(cfg, prompt))

    conn.close()
