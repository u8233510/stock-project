import streamlit as st
import pandas as pd
import database
import requests
import math
from duckduckgo_search import DDGS
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from modules.llm_model_selector import get_llm_model


TRUSTED_SOURCE_PATTERNS = {
    "A": [
        "mops.twse.com.tw",
        "twse.com.tw",
        "tpex.org.tw",
        "sec.gov",
        "investor",
        "ir.",
    ],
    "B": [
        "reuters.com",
        "bloomberg.com",
        "cnbc.com",
        "wsj.com",
        "moneydj.com",
        "cnyes.com",
        "udn.com",
    ],
}

TW_US_ADR_MAPPING = {
    "2330": "TSM",  # 台積電
    "2303": "UMC",  # 聯電
}



def _normalize_secret(value):
    """去除常見貼上污染（空白/換行/BOM）。"""
    if value is None:
        return ""
    return str(value).replace("﻿", "").strip()


def _mask_secret(value, keep=4):
    """遮罩敏感資訊，避免完整金鑰外露。"""
    val = _normalize_secret(value)
    if not val:
        return "(未設定)"
    if len(val) <= keep:
        return "*" * len(val)
    return f"{'*' * (len(val) - keep)}{val[-keep:]}"


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


def _external_cache_get(query, max_age_minutes=120):
    cache = st.session_state.setdefault("external_search_cache", {})
    item = cache.get(query)
    if not item:
        return None
    now_ts = pd.Timestamp.utcnow().timestamp()
    if now_ts - item.get("ts", 0) > max_age_minutes * 60:
        return None
    return item.get("records", [])


def _external_cache_set(query, records):
    cache = st.session_state.setdefault("external_search_cache", {})
    cache[query] = {"ts": pd.Timestamp.utcnow().timestamp(), "records": records}


def _google_news_rss_search(query, max_results=4):
    """免費補強來源：Google News RSS（不需付費，不用 Search Console）。"""
    try:
        rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        resp = requests.get(rss_url, timeout=20)
        if resp.status_code >= 400:
            return [], f"Google News RSS 失敗：HTTP {resp.status_code}"

        root = ET.fromstring(resp.text)
        items = root.findall('.//item')
        records = []
        for item in items[:max_results]:
            link = (item.findtext('link') or '').strip()
            records.append(
                {
                    "source": "GoogleNewsRSS",
                    "title": (item.findtext('title') or '').strip(),
                    "snippet": (item.findtext('description') or '').strip(),
                    "url": link,
                    "tier": _classify_source_tier(link),
                }
            )

        if not records:
            return [], "Google News RSS 查詢無結果。"
        return records, None
    except Exception as exc:
        return [], f"Google News RSS 例外：{str(exc)}"


def _wikipedia_summary_search(stock_name, sid):
    """免費補強來源：Wikipedia 摘要（公司簡介/產業線索）。"""
    candidates = [
        f"{stock_name}",
        f"{stock_name} {sid}",
    ]
    for q in candidates:
        try:
            url = "https://zh.wikipedia.org/api/rest_v1/page/summary/" + quote_plus(q)
            resp = requests.get(url, timeout=20)
            if resp.status_code >= 400:
                continue
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            title = data.get("title", "") if isinstance(data, dict) else ""
            extract = data.get("extract", "") if isinstance(data, dict) else ""
            page = data.get("content_urls", {}).get("desktop", {}).get("page", "") if isinstance(data, dict) else ""
            if title and extract:
                return [{
                    "source": "Wikipedia",
                    "title": title,
                    "snippet": extract,
                    "url": page,
                    "tier": "B",
                }], None
        except Exception:
            continue
    return [], "Wikipedia 摘要無結果。"


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



def _puter_search(query, cfg):
    """透過 Puter API 取得外部資訊摘要（可選 provider，預設不啟用）。"""
    search_cfg = cfg.get("search", {})
    api_key = _normalize_secret(search_cfg.get("puter_api_key"))
    model = search_cfg.get("puter_model", "perplexity/sonar")
    if not api_key:
        return [], "Puter 未設定 puter_api_key。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是研究助理，請根據公開網路資訊整理重點並附來源網址。"},
            {"role": "user", "content": f"請針對以下主題整理 5 點重點，每點需附來源網址：{query}"},
        ],
        "temperature": 0.1,
    }

    # 注意：Puter 官方介面可能調整，故此 provider 預設為可選。
    try:
        resp = requests.post(
            "https://api.puter.com/v2/ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
            return [], f"Puter 搜尋失敗：{err_msg}"

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(data, dict) else ""
        if not content:
            return [], "Puter 已連線，但無回傳內容。"
        return [{"source": "Puter", "title": "摘要", "snippet": content, "url": "", "tier": "B"}], None
    except Exception as exc:
        return [], f"Puter 搜尋例外：{str(exc)}"


def _openrouter_search(query, cfg):
    """透過 OpenRouter 取得外部資訊摘要（可使用你的 OpenRouter Key）。"""
    search_cfg = cfg.get("search", {})
    api_key = _normalize_secret(search_cfg.get("openrouter_api_key"))
    model = search_cfg.get("openrouter_model", "perplexity/sonar")
    if not api_key:
        return [], "OpenRouter 未設定 openrouter_api_key。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": search_cfg.get("openrouter_site_url", "https://localhost"),
        "X-Title": search_cfg.get("openrouter_app_name", "stock-project"),
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是研究助理，請根據公開網路資訊整理重點，附上來源網址，禁止虛構。",
            },
            {
                "role": "user",
                "content": f"請針對以下主題整理 5 點重點，每點需附可點擊網址：{query}",
            },
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.status_code >= 400:
            err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
            return [], f"OpenRouter 搜尋失敗：{err_msg}"

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(data, dict) else ""
        if not content:
            return [], "OpenRouter 已連線，但無回傳內容。"
        return [{"source": "OpenRouter", "title": "摘要", "snippet": content, "url": "", "tier": "B"}], None
    except Exception as exc:
        return [], f"OpenRouter 搜尋例外：{str(exc)}"


def _openrouter_connectivity_check(cfg):
    """快速檢查 OpenRouter 是否可由目前環境成功呼叫。"""
    records, warning = _openrouter_search("台股 今日重點新聞", cfg)
    if warning:
        return False, warning
    return bool(records), "OpenRouter 連線檢查成功。"

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
                        "tier": _classify_source_tier(r.get("href", "")),
                    }
                )
            if not results:
                return [], f"{source} 查詢無結果。"
            return results, None
    except Exception as exc:
        return [], f"{source} 搜尋例外：{str(exc)}"


def _classify_source_tier(url):
    """將來源分級：A(官方/監管)、B(主流媒體)、C(其他)。"""
    url_text = (url or "").lower()
    if not url_text:
        return "C"

    for tier, patterns in TRUSTED_SOURCE_PATTERNS.items():
        if any(p in url_text for p in patterns):
            return tier
    return "C"


def _resolve_us_mapping(stock_id, stock_name):
    """Layer B 起點：先做高可信白名單對應，再保留後續擴充空間。"""
    sid = str(stock_id).strip()
    ticker = TW_US_ADR_MAPPING.get(sid)
    if ticker:
        return {
            "ticker": ticker,
            "mapping_type": "direct_adr",
            "confidence": 0.98,
            "evidence": ["manual_mapping_table"],
        }

    return {
        "ticker": "",
        "mapping_type": "none",
        "confidence": 0.0,
        "evidence": [f"no_mapping_for_{stock_name}_{sid}"],
    }


def _build_search_queries(stock_name, sid):
    """建立多角度查詢，讓免費聯網摘要更接近可搜尋 LLM 的效果。"""
    base = f"{stock_name} {sid}"
    return [
        (f"{base} 公司簡介 核心產品 產業地位", "公司定位"),
        (f"{base} 最新新聞 訂單 客戶", "最新動態"),
        (f"{base} 法說會 財測 資本支出 毛利率", "經營展望"),
        (f"{base} 風險 匯率 原物料 地緣政治", "風險事件"),
    ]


def _dedup_records(records):
    seen = set()
    deduped = []
    for rec in records:
        key = (rec.get("source", ""), rec.get("title", ""), rec.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped


def _resolve_rag_config(cfg):
    """讀取 RAG 設定；優先使用 llm.rag，並相容舊錯置到 search.rag 的情況。"""
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    search_cfg = cfg.get("search", {}) if isinstance(cfg, dict) else {}

    llm_rag = llm_cfg.get("rag", {}) if isinstance(llm_cfg, dict) else {}
    search_rag = search_cfg.get("rag", {}) if isinstance(search_cfg, dict) else {}
    rag_cfg = llm_rag if llm_rag else search_rag

    llm_api_key = _normalize_secret(llm_cfg.get("api_key")) if isinstance(llm_cfg, dict) else ""
    search_api_key = _normalize_secret(search_cfg.get("api_key")) if isinstance(search_cfg, dict) else ""

    return {
        "enabled": str(rag_cfg.get("enabled", "false")).lower() == "true" if isinstance(rag_cfg, dict) else False,
        "embedding_model": (rag_cfg.get("embedding_model") if isinstance(rag_cfg, dict) else None) or "nvidia/nv-embed-v1",
        "top_k": int((rag_cfg.get("top_k") if isinstance(rag_cfg, dict) else 8) or 8),
        "api_key": llm_api_key or search_api_key,
    }


def _embed_texts_nim(api_key, model, texts):
    """呼叫 NVIDIA Embeddings API 取得向量。"""
    if not api_key or not texts:
        return []

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers=headers,
        json=payload,
        timeout=30,
    )

    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        err_msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}") if isinstance(data, dict) else f"HTTP {resp.status_code}"
        raise RuntimeError(f"Embedding API 失敗：{err_msg}")

    vectors = []
    for row in data.get("data", []) if isinstance(data, dict) else []:
        vec = row.get("embedding") if isinstance(row, dict) else None
        if isinstance(vec, list):
            vectors.append(vec)
    return vectors


def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def _apply_rag_rerank(stock_name, sid, records, cfg):
    """對外部搜集結果做 embedding 相似度排序，取 top-k。"""
    rag_cfg = _resolve_rag_config(cfg)
    if not rag_cfg["enabled"]:
        return records, None
    if not records:
        return records, "RAG 已啟用，但目前沒有可重排序的外部資料。"
    if not rag_cfg["api_key"]:
        return records, "RAG 已啟用但缺少 API key（請設定 llm.api_key 或 search.api_key）。"

    top_k = max(1, min(rag_cfg["top_k"], len(records)))
    query = f"{stock_name} {sid} 基本面 公司定位 新聞 風險"
    doc_texts = [f"{r.get('title', '')}\n{r.get('snippet', '')}\n{r.get('url', '')}" for r in records]

    try:
        q_vecs = _embed_texts_nim(rag_cfg["api_key"], rag_cfg["embedding_model"], [query])
        d_vecs = _embed_texts_nim(rag_cfg["api_key"], rag_cfg["embedding_model"], doc_texts)
        if not q_vecs or len(d_vecs) != len(records):
            return records, "RAG 重排序略過：embedding 回傳不完整。"

        q_vec = q_vecs[0]
        scored = []
        for rec, d_vec in zip(records, d_vecs):
            sim = _cosine_similarity(q_vec, d_vec)
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [rec for _, rec in scored[:top_k]]
        return reranked, f"RAG 重排序已啟用：模型 {rag_cfg['embedding_model']}，保留 Top-{top_k}。"
    except Exception as exc:
        return records, f"RAG 重排序失敗，回退原始搜尋結果：{str(exc)}"


def _build_external_context(stock_name, sid, cfg):
    """蒐集外部資訊（可配置付費/免費來源 + 社群網站搜尋）。"""
    search_cfg = cfg.get("search", {})
    preferred_provider = str(search_cfg.get("provider", "openrouter_then_rss")).lower().strip()

    records = []
    warnings = []
    topic_queries = _build_search_queries(stock_name, sid)

    openrouter_queries_used = 0
    openrouter_query_budget = int(search_cfg.get("openrouter_queries_per_run", 2) or 2)
    use_ddg = str(search_cfg.get("enable_ddg", "false")).lower() == "true"

    for query, tag in topic_queries:
        if preferred_provider in {"openrouter", "openrouter_then_rss", "openrouter_then_ddg"} and openrouter_queries_used < openrouter_query_budget:
            cache_key = f"or::{query}"
            cached = _external_cache_get(cache_key)
            if cached is not None:
                records.extend(cached)
            else:
                or_records, or_warn = _openrouter_search(query, cfg)
                records.extend(or_records)
                if or_records:
                    _external_cache_set(cache_key, or_records)
                if or_warn:
                    warnings.append(f"[{tag}] {or_warn}")
            openrouter_queries_used += 1
        elif preferred_provider == "perplexity":
            pplx_records, pplx_warn = _perplexity_search(query, cfg)
            records.extend(pplx_records)
            if pplx_warn:
                warnings.append(f"[{tag}] {pplx_warn}")
        elif preferred_provider == "puter":
            put_records, put_warn = _puter_search(query, cfg)
            records.extend(put_records)
            if put_warn:
                warnings.append(f"[{tag}] {put_warn}")

        rss_records, rss_warn = _google_news_rss_search(query, max_results=2)
        records.extend(rss_records)
        if rss_warn:
            warnings.append(f"[{tag}] {rss_warn}")

        if use_ddg or preferred_provider == "openrouter_then_ddg":
            ddg_records, ddg_warn = _ddg_search(query, max_results=2, source=f"DuckDuckGo/{tag}")
            records.extend(ddg_records)
            if ddg_warn:
                warnings.append(f"[{tag}] {ddg_warn}")

    wiki_records, wiki_warn = _wikipedia_summary_search(stock_name, sid)
    records.extend(wiki_records)
    if wiki_warn:
        warnings.append(wiki_warn)

    # 社群/輿情（可選，避免 DDG 品質差時引入噪音）
    if use_ddg:
        social_queries = [
            (f"site:x.com OR site:twitter.com {stock_name} {sid}", "X/Twitter"),
            (f"site:facebook.com {stock_name} {sid}", "Facebook"),
            (f"site:instagram.com {stock_name} {sid}", "Instagram"),
        ]
        for query, source in social_queries:
            social_records, social_warn = _ddg_search(query, max_results=2, source=source)
            records.extend(social_records)
            if social_warn:
                warnings.append(social_warn)

    records = _dedup_records(records)
    records, rag_warning = _apply_rag_rerank(stock_name, sid, records, cfg)
    if rag_warning:
        warnings.insert(0, rag_warning)

    if not records:
        warnings.insert(0, "目前未取得外部來源。請先檢查下方各來源診斷訊息。")
        return "", warnings

    source_counts = {}
    tier_counts = {"A": 0, "B": 0, "C": 0}
    for rec in records:
        src = rec.get("source", "來源")
        source_counts[src] = source_counts.get(src, 0) + 1
        tier = rec.get("tier", "C")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    summary = "、".join([f"{k}:{v}" for k, v in source_counts.items()])
    tier_summary = "、".join([f"{k}:{v}" for k, v in tier_counts.items() if v > 0])
    warnings.insert(0, f"外部來源抓取成功（{summary}；來源分級 {tier_summary}）。已停用 Google Custom Search JSON API，改採 OpenRouter + Google News RSS/Wikipedia。")

    mapping_info = _resolve_us_mapping(sid, stock_name)
    if mapping_info["mapping_type"] == "direct_adr":
        warnings.insert(1, f"美股對應：{sid} → {mapping_info['ticker']}（direct_adr, confidence={mapping_info['confidence']:.2f}）")
    else:
        warnings.insert(1, f"美股對應：目前無白名單 ADR 對應（{sid}），後續可由外部結構化來源補強。")

    lines = []
    for rec in records[:24]:
        url = rec.get("url", "")
        url_text = f"（{url}）" if url else ""
        lines.append(
            f"- [{rec.get('source', '來源')}/Tier-{rec.get('tier', 'C')}] "
            f"{rec.get('title', '')}: {rec.get('snippet', '')} {url_text}"
        )
    return "\n".join(lines), warnings


def _fmt_metric(value, fallback="未提供"):
    if value is None or value == "":
        return fallback
    return str(value)


def _to_float(value):
    if value is None:
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "未提供", "N/A", "nan"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _free_score_label(score):
    if score >= 2:
        return "偏多"
    if score <= -2:
        return "偏保守"
    return "中性"


def _latest_metric_value(financial_df, metric_names):
    """從財報明細中取出指定指標的最新值。"""
    if financial_df.empty:
        return None

    norm_names = {str(n).strip().lower() for n in metric_names}
    matched = financial_df[financial_df["type"].astype(str).str.strip().str.lower().isin(norm_names)]
    if matched.empty:
        return None
    return matched.iloc[0]["value"]


def _fmt_percent(value):
    val = _to_float(value)
    if val is None:
        return "未提供"
    return f"{val:.2f}%"


def _compute_data_quality(metrics):
    required = [
        "latest_eps",
        "prev_eps",
        "latest_revenue",
        "oldest_revenue",
        "revenue_growth",
        "roe",
        "roa",
        "gross_margin",
        "operating_cf",
    ]
    available = sum(1 for key in required if _to_float(metrics.get(key)) is not None)
    ratio = available / len(required)

    if ratio >= 0.8:
        return "高", ratio
    if ratio >= 0.5:
        return "中", ratio
    return "低", ratio


def _build_free_fundamental_report(stock_name, sid, search_ctx, metrics):
    latest_eps = _to_float(metrics.get("latest_eps"))
    prev_eps = _to_float(metrics.get("prev_eps"))
    revenue_growth = _to_float(metrics.get("revenue_growth"))

    eps_trend = "資料不足"
    if latest_eps is not None and prev_eps is not None:
        eps_trend = "成長" if latest_eps > prev_eps else ("下滑" if latest_eps < prev_eps else "持平")

    score = 0
    if latest_eps is not None:
        score += 1 if latest_eps > 0 else -1
    if revenue_growth is not None:
        score += 1 if revenue_growth > 0 else -1
    if latest_eps is not None and prev_eps is not None and latest_eps > prev_eps:
        score += 1

    risk_note = "市場景氣循環與產業競爭可能影響營收與獲利。"
    if latest_eps is not None and latest_eps < 0:
        risk_note = "目前 EPS 為負，需優先關注虧損收斂與現金流品質。"
    elif revenue_growth is not None and revenue_growth < 0:
        risk_note = "近期營收成長率為負，需留意需求放緩或產品組合變化。"

    ext_note = "未取得外部新聞摘要。"
    if search_ctx:
        ext_note = "已納入 OpenRouter / RSS / Wikipedia 等外部摘要，並交叉對照資料庫數據。"

    data_quality_level, data_quality_ratio = _compute_data_quality(metrics)

    return f"""
## 公司簡介
{stock_name}（{sid}）為台股上市櫃公司，本報告採用內部資料庫財報欄位與免費外部搜尋摘要進行整理。

## 財務分析
目前觀察到 EPS 趨勢為「{eps_trend}」，整體財務動能判讀為「{_free_score_label(score)}」。
{ext_note}
分析重點以「獲利連續性（EPS）＋成長方向（營收）＋資產品質（ROE/ROA/現金流）」三軸交叉判讀。

### 財務指標
- EPS：{_fmt_metric(metrics.get('latest_eps'))}
- ROE（股東權益報酬率）：{_fmt_percent(metrics.get('roe'))}
- ROA（資產報酬率）：{_fmt_percent(metrics.get('roa'))}
- 營收成長率：{_fmt_metric(metrics.get('revenue_growth'))}
- 毛利率：{_fmt_percent(metrics.get('gross_margin'))}

## 營收分析
近 12 月營收由 { _fmt_metric(metrics.get('oldest_revenue')) } 億變化至 { _fmt_metric(metrics.get('latest_revenue')) } 億，成長率為 { _fmt_metric(metrics.get('revenue_growth')) }。
若成長率轉弱，通常代表終端需求、產品價格或出貨節奏承壓。

## 毛利率分析
目前資料庫未提供可直接計算的最新毛利率欄位，建議後續補齊季報毛利率以提升判讀精度。

## 現金流量分析
營業現金流（Operating Cash Flow）：{_fmt_metric(metrics.get('operating_cf'))}。
若營收與獲利成長但現金流未同步改善，需留意應收帳款、庫存與資本支出壓力。

## 投資評價
- 短期評價：{_free_score_label(score)}（以營收與 EPS 最新變化為主）
- 中期評價：中性偏基本面驗證（需追蹤連續 2~3 季 EPS 與營收是否同向）
- 長期評價：取決於產品競爭力、資本支出效率、自由現金流與景氣循環位置
- 目標價格：資料不足（免費版不產生目標價）

## 風險評估
- 市場風險：受總體景氣、利率與資金面影響
- 財務風險：{risk_note}
- 法規/政策風險：需留意產業政策、出口管制與會計準則變動

## 結論
本次為「強化版免費 AI 基本面分析」，以可驗證數據做規則化摘要；若啟用 LLM 可再進一步做脈絡整合。
目前資料完整度評估：{data_quality_level}（{data_quality_ratio:.0%}）。
建議後續持續追蹤：EPS 連續性、營收年增率轉折、現金流品質，以及重大新聞事件對訂單與毛利率的影響。
""".strip()


def _build_fundamental_prompt(stock_name, sid, search_ctx, metrics):
    """建立固定章節格式的基本面分析 Prompt。"""
    return f"""
請你扮演台股資深基本面分析師，針對 {stock_name}（{sid}）撰寫報告。

【重要規則】
1) 嚴格使用以下固定格式與標題順序，不要增減章節。
2) 若資料不足，請明確寫「未提供」或「資料不足」，禁止虛構。
3) 所有數值盡量引用我提供的資料；若引用新聞，僅能使用「搜尋事實摘要」。
4) 若有外部事件，請在句尾加上對應來源網址（可多筆）。
5) 以繁體中文輸出。

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
- ROE：{metrics.get('roe', '未提供')}
- ROA：{metrics.get('roa', '未提供')}
- 毛利率：{metrics.get('gross_margin', '未提供')}
- 營業現金流：{metrics.get('operating_cf', '未提供')}
""".strip()

# 1. 核心 AI 呼叫工具 (保持穩定，未更動)
def _call_nim_fundamental(cfg, prompt):
    llm_cfg = cfg.get("llm", {})
    api_key = _normalize_secret(llm_cfg.get("api_key"))
    model_name = get_llm_model(cfg, "fundamental", "meta/llama-3.1-70b-instruct")
    if not api_key:
        raise ValueError("llm.api_key 未設定。")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一位專業的證券分析師。請優先參考連網搜尋到的事實，結合財務數據給出具體的投資評價，嚴禁虛構公司業務。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        data = resp.json()
    except Exception:
        data = {}

    if resp.status_code >= 400:
        err_msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
        raise RuntimeError(err_msg or f"NIM API 呼叫失敗（HTTP {resp.status_code}）。")

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(data, dict) else ""
    if not content:
        raise RuntimeError("NIM API 未回傳可用內容。")
    return content

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
    metric_candidates = [
        "EPS", "Net Profit", "ROE", "ROE(%)", "Return on Equity",
        "ROA", "ROA(%)", "Return on Assets",
        "Gross Margin", "Gross Margin(%)", "毛利率",
        "Operating Cash Flow", "營業活動之淨現金流入（流出）", "營業現金流",
    ]
    metric_filter = ", ".join([f"'{m}'" for m in metric_candidates])
    profit_raw = pd.read_sql(
        f"SELECT date, type, value FROM stock_financial_statements WHERE stock_id='{sid}' AND type IN ({metric_filter}) ORDER BY date DESC LIMIT 200",
        conn,
    )
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
        llm_cfg = cfg.get("llm", {})
        llm_available = bool(_normalize_secret(llm_cfg.get("api_key")))

        use_llm = st.toggle(
            "啟用 LLM 強化分析（可選）",
            value=llm_available,
            help="若已設定 llm.api_key，建議開啟；未啟用時系統將使用免費規則化摘要。",
        )
        model_name = st.text_input(
            "LLM 模型（NVIDIA NIM）",
            value=get_llm_model(cfg, "fundamental", "meta/llama-3.1-70b-instruct"),
            disabled=not use_llm,
        )

        if use_llm and not llm_available:
            st.warning("目前未設定 llm.api_key，將自動回退到免費規則化報告。")

        if llm_available:
            st.success(f"✅ 已偵測到 llm.api_key（{_mask_secret(llm_cfg.get('api_key'))}），可直接使用 {model_name} 進行強化分析。")
        st.info("💡 改善建議：系統會先做多查詢聯網蒐集，再交給 LLM 整合；效果會比只靠模型記憶好。")
        st.caption("此頁支援純 NVIDIA LLM 分析；若未設定 llm.api_key，系統會自動使用免費規則化報告。")

        run_btn_label = "🚀 啟動 AI 基本面分析（LLM 強化）" if use_llm else "🚀 啟動 AI 基本面分析（免費規則化）"
        # ✅ 保留聯網搜尋邏輯
        if st.button(run_btn_label, use_container_width=True):
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
                    "revenue_growth": revenue_growth,
                    "roe": _latest_metric_value(profit_raw, ["ROE", "Return on Equity", "ROE(%)"]),
                    "roa": _latest_metric_value(profit_raw, ["ROA", "Return on Assets", "ROA(%)"]),
                    "gross_margin": _latest_metric_value(profit_raw, ["Gross Margin", "Gross Margin(%)", "毛利率"]),
                    "operating_cf": _latest_metric_value(profit_raw, ["Operating Cash Flow", "營業活動之淨現金流入（流出）", "營業現金流"]),
                }

                if use_llm and llm_available:
                    cfg.setdefault("llm", {}).setdefault("models", {})["fundamental"] = model_name
                    prompt = _build_fundamental_prompt(selected_stock, sid, search_ctx, metrics)
                    try:
                        ai_report = _call_nim_fundamental(cfg, prompt)
                        st.markdown(ai_report)
                    except Exception as exc:
                        st.error(f"LLM 呼叫失敗，改用免費規則化報告：{str(exc)}")
                        st.markdown(_build_free_fundamental_report(selected_stock, sid, search_ctx, metrics))
                else:
                    st.markdown(_build_free_fundamental_report(selected_stock, sid, search_ctx, metrics))

    conn.close()
