from datetime import datetime, timedelta, timezone
import re
from typing import Any

import requests
import streamlit as st
from ddgs import DDGS

import database
from modules.llm_model_selector import get_llm_model


def _search_news(ddgs: DDGS, query: str, timelimit: str) -> list[dict]:
    """相容不同 ddgs 版本的 news 參數命名。"""
    common_kwargs = {
        "region": "wt-wt",
        "safesearch": "off",
        "timelimit": timelimit,
        "max_results": 10,
    }

    attempts = [
        lambda: ddgs.news(query=query, **common_kwargs),
        lambda: ddgs.news(query, **common_kwargs),
        lambda: ddgs.news(keywords=query, **common_kwargs),
    ]

    last_exc = None
    for attempt in attempts:
        try:
            return list(attempt())
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    return []


def _is_relevant_news(item: dict, sid: str, sname: str) -> bool:
    """以股票代碼/名稱做基礎相關性過濾，降低無關新聞。"""
    text = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("body", "")),
            str(item.get("snippet", "")),
            str(item.get("url", "")),
            str(item.get("href", "")),
            str(item.get("link", "")),
        ]
    ).lower()

    sid_txt = str(sid).strip().lower()
    sname_txt = str(sname).strip().lower()

    sid_hit = sid_txt and sid_txt in text
    name_hit = sname_txt and sname_txt in text
    return bool(sid_hit or name_hit)


def _parse_relative_date(text: str):
    now = datetime.now(timezone.utc)
    raw = str(text).strip().lower()

    m = re.search(r"(\d+)\s*(minute|minutes|min|mins)\s*ago", raw)
    if m:
        return now - timedelta(minutes=int(m.group(1)))

    m = re.search(r"(\d+)\s*(hour|hours|hr|hrs)\s*ago", raw)
    if m:
        return now - timedelta(hours=int(m.group(1)))

    m = re.search(r"(\d+)\s*(day|days)\s*ago", raw)
    if m:
        return now - timedelta(days=int(m.group(1)))

    m = re.search(r"(\d+)\s*(week|weeks)\s*ago", raw)
    if m:
        return now - timedelta(weeks=int(m.group(1)))

    m = re.search(r"(\d+)\s*分鐘前", raw)
    if m:
        return now - timedelta(minutes=int(m.group(1)))

    m = re.search(r"(\d+)\s*小時前", raw)
    if m:
        return now - timedelta(hours=int(m.group(1)))

    m = re.search(r"(\d+)\s*天前", raw)
    if m:
        return now - timedelta(days=int(m.group(1)))

    m = re.search(r"(\d+)\s*週前", raw)
    if m:
        return now - timedelta(weeks=int(m.group(1)))

    return None


def _parse_news_date(date_str: str):
    if not date_str:
        return datetime.min.replace(tzinfo=timezone.utc)

    txt = str(date_str).strip()
    normalized = txt.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%b %d, %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    rel = _parse_relative_date(txt)
    if rel is not None:
        return rel

    return datetime.min.replace(tzinfo=timezone.utc)


def _news_sort_key(item: dict) -> float:
    """回傳可排序數值，避免不同來源日期異常造成排序例外。"""
    try:
        dt = _parse_news_date(str(item.get("date", "")))
        if isinstance(dt, datetime):
            return dt.timestamp()
    except Exception:
        pass
    return 0.0


def _build_queries(sid: str, sname: str) -> list[str]:
    return [
        f"{sname} {sid} 台股 新聞",
        f"{sname} {sid} 股票 新聞",
        f"{sname} 股票 新聞",
        f"{sid} 股票 新聞",
    ]


def _fetch_dgs_news(sid: str, sname: str, timelimit: str = "y") -> list[dict]:
    queries = _build_queries(sid, sname)
    news_list: list[dict[str, Any]] = []

    with DDGS() as ddgs:
        best_fallback = []
        for query in queries:
            fetched = _search_news(ddgs, query, timelimit)
            if fetched and not best_fallback:
                best_fallback = fetched

            relevant = [n for n in fetched if _is_relevant_news(n, sid, sname)]
            if relevant:
                news_list = relevant
                break

        if not news_list:
            news_list = best_fallback

    return sorted(news_list, key=_news_sort_key, reverse=True)[:10]


def _fetch_serper_news(sid: str, sname: str, api_key: str) -> list[dict]:
    queries = _build_queries(sid, sname)
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    best_fallback: list[dict[str, Any]] = []
    for query in queries:
        payload = {"q": query, "num": 10, "gl": "tw", "hl": "zh-tw", "tbs": "qdr:y"}
        resp = requests.post("https://google.serper.dev/news", headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        fetched = data.get("news", []) if isinstance(data, dict) else []
        if fetched and not best_fallback:
            best_fallback = fetched

        relevant = [n for n in fetched if _is_relevant_news(n, sid, sname)]
        if relevant:
            best_fallback = relevant
            break

    normalized = [
        {
            "title": item.get("title", ""),
            "body": item.get("snippet", "") or item.get("body", ""),
            "url": item.get("link", "") or item.get("url", ""),
            "source": item.get("source", "SERPER"),
            "date": item.get("date", ""),
        }
        for item in best_fallback
    ]

    return sorted(normalized, key=_news_sort_key, reverse=True)[:10]


def _render_news_list(news_list: list[dict], source_label: str):
    if not news_list:
        st.warning("目前找不到相關新聞（已嘗試多組關鍵字與近一年範圍），請稍後再試。")
        return

    for news in news_list[:10]:
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                source = news.get("source", source_label)
                date_str = news.get("date", "")
                if date_str:
                    dt = _parse_news_date(date_str)
                    if dt != datetime.min.replace(tzinfo=timezone.utc):
                        st.caption(f"📅 {dt.astimezone(timezone.utc).strftime('%m/%d %H:%M')}")
                    else:
                        st.caption(date_str)
                st.info(f"📍 {source}")

            with col2:
                title = news.get("title", "(無標題)")
                url = news.get("url") or news.get("href", "") or news.get("link", "")
                snippet = news.get("body", "點擊標題查看完整內容...")
                if url:
                    st.markdown(f"#### [{title}]({url})")
                else:
                    st.markdown(f"#### {title}")
                st.write(f"{snippet[:150]}...")

            st.divider()


def _summarize_news(cfg: dict, sid: str, sname: str, source_label: str, news_list: list[dict]) -> str:
    if not news_list:
        return "目前沒有可供總結的新聞內容。"

    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    api_key = llm_cfg.get("api_key", "")
    if not api_key:
        return "⚠️ 尚未設定 LLM API Key（llm.api_key），目前無法產生新聞總結。"

    news_lines = []
    for idx, item in enumerate(news_list[:10], start=1):
        title = str(item.get("title", "(無標題)")).strip()
        snippet = str(item.get("body", "")).strip()[:220]
        date_str = str(item.get("date", "")).strip()
        source = str(item.get("source", source_label)).strip()
        url = str(item.get("url") or item.get("href", "") or item.get("link", "")).strip()
        news_lines.append(
            f"{idx}. [{source}] {title}\n日期：{date_str or '未知'}\n摘要：{snippet or '（無摘要）'}\n連結：{url or '（無連結）'}"
        )

    prompt = (
        f"請以繁體中文總結 {sname}（{sid}）的 {source_label} 新聞，並輸出：\n"
        "1) 三點重點\n"
        "2) 對股價可能的偏多/偏空影響（短期）\n"
        "3) 需要追蹤的風險事件\n"
        "內容請精簡、避免杜撰，若資訊不足請明確標示。\n\n"
        f"新聞資料：\n{chr(10).join(news_lines)}"
    )

    payload = {
        "model": get_llm_model(cfg, "fundamental"),
        "messages": [
            {"role": "system", "content": "你是專業台股研究助理，僅能根據給定新聞進行整理，不可捏造。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _render_summary_button(cfg: dict, sid: str, sname: str, source_label: str, news_list: list[dict], key: str):
    if st.button(f"🧠 總結 {source_label} 新聞", use_container_width=True, key=key):
        with st.spinner(f"正在整理 {source_label} 新聞重點..."):
            try:
                summary = _summarize_news(cfg, sid, sname, source_label, news_list)
                st.markdown(summary)
            except Exception as e:
                st.error(f"{source_label} 新聞總結失敗：{str(e)}")


def render_stock_news(sid: str, sname: str, cfg: dict | None = None, serper_api_key: str | None = None):
    """顯示 DGS 與 SERPER 兩種來源新聞（最新到最舊，最多 10 筆）。"""
    st.subheader(f"🌐 {sname} ({sid}) 最新相關新聞")

    cfg = cfg or database.load_config()
    tab_dgs, tab_serper = st.tabs(["DGS", "SERPER"])

    with tab_dgs:
        try:
            with st.spinner("DGS 正在搜尋最新動態..."):
                dgs_news = _fetch_dgs_news(sid, sname, timelimit="y")
            _render_news_list(dgs_news, "DGS")
            _render_summary_button(cfg, sid, sname, "DGS", dgs_news, key=f"sum_dgs_{sid}")
        except Exception as e:
            st.error(f"DGS 搜尋新聞時發生錯誤：{str(e)}")

    with tab_serper:
        if not serper_api_key:
            st.warning("未設定 SERPER API Key（search.serper_api_key），此分頁無法查詢。")
            return
        try:
            with st.spinner("SERPER 正在搜尋最新動態..."):
                serper_news = _fetch_serper_news(sid, sname, serper_api_key)
            _render_news_list(serper_news, "SERPER")
            _render_summary_button(cfg, sid, sname, "SERPER", serper_news, key=f"sum_serper_{sid}")
        except Exception as e:
            st.error(f"SERPER 搜尋新聞時發生錯誤：{str(e)}")


def show_fundamental_analysis():
    """保持與 app.py 相容的入口函數。"""
    st.markdown("### 💎 基本面分析（新聞）")

    cfg = database.load_config()
    universe = cfg.get("universe", [])
    if not universe:
        st.error("universe 未設定，請先在設定檔配置標的。")
        return

    stock_options = {f"{s['stock_id']} {s['name']}": (s["stock_id"], s["name"]) for s in universe}
    selected_label = st.selectbox("選擇股票", list(stock_options.keys()))
    sid, sname = stock_options[selected_label]
    serper_api_key = cfg.get("search", {}).get("serper_api_key", "")

    if st.button("🔍 搜尋最新新聞", use_container_width=True):
        render_stock_news(sid, sname, cfg=cfg, serper_api_key=serper_api_key)


if __name__ == "__main__":
    st.set_page_config(page_title="股票新聞搜尋", layout="wide")
    render_stock_news("2233", "宇隆")
