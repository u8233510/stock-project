import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from duckduckgo_search import DDGS

import database
from modules.llm_model_selector import get_llm_model


def _normalize_secret(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value).replace(",", "").replace("%", "").strip()
    if txt in {"", "N/A", "None", "nan", "--"}:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _safe_read_sql(conn, sql: str, params: Tuple | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def _mask_secret(value: Any, keep: int = 4) -> str:
    raw = _normalize_secret(value)
    if not raw:
        return "(未設定)"
    if len(raw) <= keep:
        return "*" * len(raw)
    return f"{'*' * (len(raw) - keep)}{raw[-keep:]}"


def _call_nim(cfg: Dict[str, Any], task: str, system_prompt: str, user_prompt: str) -> str:
    llm_cfg = cfg.get("llm", {})
    api_key = _normalize_secret(llm_cfg.get("api_key"))
    if not api_key:
        raise ValueError("llm.api_key 未設定")

    model = get_llm_model(cfg, task, "meta/llama-3.1-70b-instruct")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", json=payload, headers=headers, timeout=90)
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
        raise RuntimeError(msg)

    text = data.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(data, dict) else ""
    if not text:
        raise RuntimeError("LLM 未回傳內容")
    return text


def _ddg_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    try:
        out: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=limit):
                out.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                        "source": "DuckDuckGo",
                    }
                )
        return out
    except Exception:
        return []


def _perplexity_search(cfg: Dict[str, Any], query: str) -> List[Dict[str, str]]:
    search_cfg = cfg.get("search", {})
    api_key = _normalize_secret(search_cfg.get("perplexity_api_key"))
    model = search_cfg.get("perplexity_model", "sonar")
    if not api_key:
        return []

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是研究助理，回覆重點並盡量附上來源。"},
            {"role": "user", "content": f"整理以下主題的最新重點，列點輸出：{query}"},
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers, timeout=45)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            return []
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(data, dict) else ""
        if not content:
            return []
        return [{"title": "Perplexity 摘要", "snippet": content, "url": "", "source": "Perplexity"}]
    except Exception:
        return []


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(text).lower()))


def _simple_rag(query: str, documents: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
    q_tokens = _tokenize(query)
    scored: List[Tuple[int, Dict[str, str]]] = []
    for doc in documents:
        d_tokens = _tokenize(f"{doc.get('title', '')} {doc.get('snippet', '')}")
        score = len(q_tokens & d_tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k] if s > 0] or documents[:top_k]


def _collect_external_context(cfg: Dict[str, Any], stock_label: str, sid: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    queries = [
        f"{stock_label} {sid} 法說會 財報 展望",
        f"{stock_label} {sid} 分點 籌碼",
        f"{stock_label} {sid} 產業 景氣 上游 下游",
    ]
    docs: List[Dict[str, str]] = []
    for q in queries:
        docs.extend(_ddg_search(q, limit=4))

    if not docs:
        warnings.append("外部搜尋未取得結果（DDG）。")

    # Optional search-LLM as an additional source
    px_docs = _perplexity_search(cfg, f"{stock_label} {sid} 最新消息與風險")
    if px_docs:
        docs.extend(px_docs)
    else:
        warnings.append("Perplexity 未啟用或無回傳，已略過。")

    rag_docs = _simple_rag(f"{stock_label} {sid} 基本面 分點 籌碼 預測", docs, top_k=8)
    lines = []
    for i, d in enumerate(rag_docs, start=1):
        title = d.get("title", "")
        snippet = d.get("snippet", "")
        url = d.get("url", "")
        lines.append(f"[{i}] {title}\n{snippet}\n來源: {url}")

    return "\n\n".join(lines), warnings


def _load_fundamental_data(conn, sid: str) -> Dict[str, Any]:
    rev = _safe_read_sql(
        conn,
        """
        SELECT date, revenue
        FROM stock_month_revenue_monthly
        WHERE stock_id=?
        ORDER BY date DESC
        LIMIT 12
        """,
        (sid,),
    )
    fin = _safe_read_sql(
        conn,
        """
        SELECT date, type, value
        FROM stock_financial_statements
        WHERE stock_id=?
        ORDER BY date DESC
        LIMIT 300
        """,
        (sid,),
    )
    val = _safe_read_sql(
        conn,
        """
        SELECT date, PER, dividend_yield
        FROM stock_per_pbr_daily
        WHERE stock_id=?
        ORDER BY date DESC
        LIMIT 20
        """,
        (sid,),
    )

    eps = None
    roe = None
    gross_margin = None
    operating_cf = None

    if not fin.empty:
        fin2 = fin.copy()
        fin2["value"] = fin2["value"].apply(_to_float)

        def _pick(types: List[str]) -> float | None:
            rows = fin2[fin2["type"].isin(types)]
            if rows.empty:
                return None
            return rows.iloc[0]["value"]

        eps = _pick(["EPS", "每股盈餘", "基本每股盈餘（元）"])
        roe = _pick(["ROE", "ROE(%)", "Return on Equity", "權益報酬率"])
        gross_margin = _pick(["Gross Margin", "Gross Margin(%)", "毛利率", "營業毛利率(%)"])
        operating_cf = _pick(["Operating Cash Flow", "營業現金流", "營業活動之淨現金流入（流出）"])

    rev_growth = None
    if not rev.empty and len(rev) > 1:
        newest = _to_float(rev.iloc[0]["revenue"])
        oldest = _to_float(rev.iloc[-1]["revenue"])
        if newest is not None and oldest not in {None, 0}:
            rev_growth = (newest - oldest) / oldest * 100

    latest_per = _to_float(val.iloc[0]["PER"]) if not val.empty else None
    latest_yield = _to_float(val.iloc[0]["dividend_yield"]) if not val.empty else None

    return {
        "rev": rev,
        "fin": fin,
        "val": val,
        "eps": eps,
        "roe": roe,
        "gross_margin": gross_margin,
        "operating_cf": operating_cf,
        "rev_growth": rev_growth,
        "per": latest_per,
        "yield": latest_yield,
    }


def _load_branch_data(conn, sid: str, start_d, end_d) -> pd.DataFrame:
    return _safe_read_sql(
        conn,
        """
        SELECT date, securities_trader, buy, sell, price
        FROM branch_price_daily
        WHERE stock_id=? AND date >= ? AND date <= ?
        """,
        (sid, str(start_d), str(end_d)),
    )


def _build_branch_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"status": "無資料"}

    g = (
        df.groupby("securities_trader", as_index=False)
        .agg(net_volume=("buy", "sum"), sell_volume=("sell", "sum"), avg_price=("price", "mean"))
    )
    g["net"] = g["net_volume"] - g["sell_volume"]
    g = g.sort_values("net", ascending=False)

    top_buy = g.head(5)[["securities_trader", "net", "avg_price"]]
    top_sell = g.tail(5)[["securities_trader", "net", "avg_price"]]

    return {
        "status": "ok",
        "top_buy": top_buy.to_dict(orient="records"),
        "top_sell": top_sell.to_dict(orient="records"),
        "total_net": float(g["net"].sum()),
    }


def _load_chip_data(conn, sid: str, start_d, end_d) -> Dict[str, Any]:
    ohlcv = _safe_read_sql(
        conn,
        """
        SELECT date, close, volume
        FROM stock_ohlcv_daily
        WHERE stock_id=? AND date >= ? AND date <= ?
        ORDER BY date DESC
        """,
        (sid, str(start_d), str(end_d)),
    )
    branch = _load_branch_data(conn, sid, start_d, end_d)

    latest_close = _to_float(ohlcv.iloc[0]["close"]) if not ohlcv.empty else None
    avg_volume = _to_float(ohlcv["volume"].mean()) if not ohlcv.empty else None

    net = None
    concentration = None
    if not branch.empty:
        gp = branch.groupby("securities_trader", as_index=False).agg(buy=("buy", "sum"), sell=("sell", "sum"))
        gp["net"] = gp["buy"] - gp["sell"]
        total_buy = gp["buy"].sum()
        top5_buy = gp.sort_values("buy", ascending=False).head(5)["buy"].sum()
        concentration = (top5_buy / total_buy * 100) if total_buy else None
        net = gp["net"].sum()

    return {
        "latest_close": latest_close,
        "avg_volume": avg_volume,
        "net": net,
        "concentration": concentration,
    }


def _build_prediction_input(fund: Dict[str, Any], chip: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    if (fund.get("rev_growth") or 0) > 0:
        score += 1
    if (fund.get("eps") or 0) > 0:
        score += 1
    if (fund.get("roe") or 0) > 8:
        score += 1
    if (chip.get("net") or 0) > 0:
        score += 1
    if (chip.get("concentration") or 0) < 35:
        score += 1

    if score >= 4:
        regime = "偏多"
    elif score <= 1:
        regime = "偏空"
    else:
        regime = "區間"

    return {"score": score, "regime": regime}


def _fallback_report(stock_label: str, sid: str, section: str, data: Dict[str, Any], context: str) -> str:
    return (
        f"### {section}（規則化報告）\n"
        f"標的: {stock_label} ({sid})\n\n"
        f"- 核心資料: `{data}`\n"
        f"- 外部資訊摘要（RAG 節選）:\n{context[:1200]}"
    )


def _render_section_with_llm(cfg: Dict[str, Any], task: str, section_title: str, stock_label: str, sid: str, data: Dict[str, Any], context: str):
    llm_enabled = bool(_normalize_secret(cfg.get("llm", {}).get("api_key")))
    prompt = f"""
你現在要撰寫「{section_title}」段落，標的是 {stock_label}({sid})。
請基於提供資料給出：
1) 目前狀態判讀
2) 2~3 個關鍵風險
3) 2~3 個後續追蹤指標
4) 最後一句操作建議（非投資保證）

【資料】
{data}

【外部資訊（RAG 節選）】
{context}
""".strip()
    if llm_enabled:
        try:
            text = _call_nim(
                cfg,
                task=task,
                system_prompt="你是台股研究員，嚴禁虛構，優先引用給定資料。",
                user_prompt=prompt,
            )
            st.markdown(text)
            return
        except Exception as exc:
            st.warning(f"{section_title} LLM 失敗，改用規則化輸出：{exc}")

    st.markdown(_fallback_report(stock_label, sid, section_title, data, context))


def show_fundamental_analysis():
    st.markdown("### 💎 四段式整合分析（基本面 / 分點 / 籌碼 / 預測）")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s["stock_id"] for s in universe}
    if not stock_options:
        st.error("universe 未設定，請先在設定檔配置標的。")
        conn.close()
        return

    c1, c2 = st.columns([2, 2])
    with c1:
        selected_label = st.selectbox("分析標的", list(stock_options.keys()))
        sid = stock_options[selected_label]
    with c2:
        def_s = pd.to_datetime("today") - pd.Timedelta(days=90)
        def_e = pd.to_datetime("today")
        date_range = st.date_input("分析區間", value=[def_s, def_e])

    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.warning("請選擇完整日期區間。")
        conn.close()
        return

    start_d, end_d = pd.to_datetime(date_range[0]).date(), pd.to_datetime(date_range[1]).date()

    st.caption(
        "模型策略：基本面/預測使用較強推理 LLM；分點/籌碼優先結構化資料計算；"
        "外部消息使用搜尋引擎 +（可選）Perplexity，再經簡易 RAG 節選。"
    )
    st.info(
        f"LLM Key: {_mask_secret(cfg.get('llm', {}).get('api_key'))} | "
        f"Perplexity Key: {_mask_secret(cfg.get('search', {}).get('perplexity_api_key'))}"
    )

    run = st.button("🚀 產生四段式分析", use_container_width=True)

    if run:
        with st.spinner("蒐集資料與建模中..."):
            context, warnings = _collect_external_context(cfg, selected_label, sid)
            for w in warnings:
                st.warning(w)

            fundamental = _load_fundamental_data(conn, sid)
            branch_df = _load_branch_data(conn, sid, start_d, end_d)
            branch = _build_branch_summary(branch_df)
            chip = _load_chip_data(conn, sid, start_d, end_d)
            prediction = _build_prediction_input(fundamental, chip)

        t1, t2, t3, t4 = st.tabs(["📘 基本面分析", "🏦 分點分析", "🧩 籌碼分析", "🔮 預測"])

        with t1:
            st.dataframe(fundamental.get("rev", pd.DataFrame()).head(12), use_container_width=True, hide_index=True)
            _render_section_with_llm(cfg, "fundamental", "基本面分析", selected_label, sid, {
                "eps": fundamental.get("eps"),
                "roe": fundamental.get("roe"),
                "gross_margin": fundamental.get("gross_margin"),
                "operating_cf": fundamental.get("operating_cf"),
                "rev_growth_%": fundamental.get("rev_growth"),
                "per": fundamental.get("per"),
                "dividend_yield": fundamental.get("yield"),
            }, context)

        with t2:
            st.dataframe(branch_df.head(30), use_container_width=True, hide_index=True)
            _render_section_with_llm(cfg, "branch", "分點分析", selected_label, sid, branch, context)

        with t3:
            st.json(chip)
            _render_section_with_llm(cfg, "chip", "籌碼分析", selected_label, sid, chip, context)

        with t4:
            st.json(prediction)
            _render_section_with_llm(cfg, "prediction", "預測", selected_label, sid, {
                "prediction": prediction,
                "fundamental": {
                    "rev_growth_%": fundamental.get("rev_growth"),
                    "eps": fundamental.get("eps"),
                    "roe": fundamental.get("roe"),
                },
                "chip": chip,
            }, context)

    conn.close()
