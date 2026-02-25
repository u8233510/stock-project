import numpy as np
import pandas as pd
import requests
import streamlit as st

import database
from modules.llm_model_selector import get_llm_model
from utility.branch_anomaly_detection import prepare_branch_daily_features
from utility.chip_strategy_ai import ChipStrategyAI, ChipStrategyConfig
from utility.winner_branch_ai_system import WinnerBranchConfig


DISPLAY_COLUMN_MAP = {
    "stock_id": "股票代號\nStock ID",
    "date": "日期\nDate",
    "branch_id": "分點代碼\nBranch ID",
    "branch_name": "分點名稱\nBranch Name",
    "hit_rate": "命中率\nHit Rate",
    "avg_alpha": "平均超額報酬\nAverage Alpha",
    "pnl_ratio": "盈虧比\nPnL Ratio",
    "sharpe": "夏普值\nSharpe Ratio",
    "timing_score": "時機分數\nTiming Score",
    "holding_power": "持股續航力\nHolding Power",
    "risk_penalty": "風險懲罰\nRisk Penalty",
    "winner_rating": "贏家評分\nWinner Rating",
    "winner_type": "贏家類型\nWinner Type",
    "net_vol": "淨買賣超\nNet Volume",
    "vol_share": "成交量占比\nVolume Share",
    "alert_level": "警示等級\nAlert Level",
    "alert_level_desc": "警示說明\nAlert Description",
    "reason": "觸發原因\nTrigger Reason",
    "hhi": "HHI 集中度\nHHI Concentration",
    "entropy": "熵值\nEntropy",
    "hhi_delta_10": "HHI 10日變化\n10D HHI Delta",
    "price_compression": "價格壓縮度\nPrice Compression",
    "rule_hhi_rise": "規則：HHI上升\nRule: HHI Rise",
    "rule_compression": "規則：價格壓縮\nRule: Price Compression",
    "strategy_candidate": "策略候選\nStrategy Candidate",
    "candidate_score": "候選分數\nCandidate Score",
    "rule_id": "規則代號\nRule ID",
    "rule_desc": "規則說明\nRule Description",
    "support": "樣本數\nSupport",
    "win_rate": "勝率\nWin Rate",
    "avg_return": "平均報酬\nAverage Return",
    "expectancy": "期望值\nExpectancy",
    "stability": "穩定度\nStability",
    "trigger_reason": "觸發原因\nTrigger Reason",
    "risk_level": "風險分級\nRisk Level",
    "position_suggestion": "建議倉位\nPosition Suggestion",
    "recent_win_rate": "近期勝率\nRecent Win Rate",
    "baseline_win_rate": "基準勝率\nBaseline Win Rate",
    "hit_rate_drop": "命中率下滑\nHit-Rate Drop",
    "recent_expectancy": "近期期望值\nRecent Expectancy",
    "pause_strategy": "是否暫停\nPause Strategy",
    "status": "狀態\nStatus",
}

ALERT_LEVEL_DESC = {
    "A": "A 級（最強）",
    "B": "B 級（中強）",
    "C": "C 級（觀察）",
}

RATING_FOCUS_COLUMNS = [
    "date",
    "branch_id",
    "branch_name",
    "winner_type",
    "winner_rating",
    "hit_rate",
    "avg_alpha",
    "pnl_ratio",
    "sharpe",
    "timing_score",
    "holding_power",
    "risk_penalty",
]


def _to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    if "stock_id" in display_df.columns:
        display_df = display_df.drop(columns=["stock_id"])

    if "date" in display_df.columns:
        parsed_date = pd.to_datetime(display_df["date"], errors="coerce")
        display_df["date"] = parsed_date.dt.strftime("%Y-%m-%d").where(parsed_date.notna(), display_df["date"])

    return display_df.rename(columns={k: v for k, v in DISPLAY_COLUMN_MAP.items() if k in display_df.columns})


def _render_winner_rating_table(rating_df: pd.DataFrame, branch_lookup: pd.DataFrame):
    if rating_df.empty:
        st.info("目前沒有可顯示的分點評級資料。")
        return

    merged = rating_df.merge(branch_lookup, on="branch_id", how="left")
    merged["branch_name"] = merged["branch_name"].fillna("-")

    sort_cols = [
        "winner_rating",
        "hit_rate",
        "avg_alpha",
        "pnl_ratio",
        "sharpe",
        "timing_score",
        "holding_power",
    ]
    available_sort_cols = [c for c in sort_cols if c in merged.columns]
    if available_sort_cols:
        merged = merged.sort_values(available_sort_cols, ascending=[False] * len(available_sort_cols))

    st.caption("同一張表整合全部重點欄位，可依 Rating 篩選要顯示的分點。")
    st.caption("Top N：只保留評分最高前 N 筆；Rating 門檻：保留所有評分 >= 指定分數的分點。")

    filter_mode = st.radio(
        "顯示分點篩選方式",
        options=["全部分點", "Top N（依 Rating）", "Rating 門檻"],
        horizontal=True,
        key="winner_rating_filter_mode",
    )

    if filter_mode == "Top N（依 Rating）":
        max_rows = max(1, len(merged))
        top_n = st.slider(
            "顯示前 N 名分點",
            min_value=1,
            max_value=max_rows,
            value=min(20, max_rows),
            step=1,
            key="winner_rating_top_n",
        )
        display_df = merged.head(top_n)
    elif filter_mode == "Rating 門檻":
        threshold = st.slider(
            "最低 Winner Rating",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.5,
            key="winner_rating_threshold",
        )
        display_df = merged[merged["winner_rating"] >= threshold]
    else:
        display_df = merged

    selected_branches = st.multiselect(
        "進一步指定要顯示的分點（可複選）",
        options=[str(v) for v in display_df["branch_id"].astype(str).unique()],
        default=[],
        help="不選則顯示篩選後的全部分點。",
        key="winner_rating_branch_multiselect",
    )
    if selected_branches:
        display_df = display_df[display_df["branch_id"].astype(str).isin(selected_branches)]

    summary_cols = [c for c in RATING_FOCUS_COLUMNS if c in display_df.columns]
    final_df = display_df[summary_cols].copy() if summary_cols else display_df.copy()
    st.dataframe(_to_display_df(final_df), use_container_width=True, hide_index=True)




def _format_percentage_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    return out


def _format_decimal_columns(df: pd.DataFrame, columns: list[str], ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(ndigits)
    return out


def _build_rules_display(rules_df: pd.DataFrame) -> pd.DataFrame:
    if rules_df.empty:
        return rules_df
    cols = ["rule_id", "rule_desc", "support", "win_rate", "avg_return", "expectancy", "stability"]
    out = rules_df[[c for c in cols if c in rules_df.columns]].copy()
    out = _format_decimal_columns(out, ["expectancy"], ndigits=4)
    out = _format_percentage_columns(out, ["win_rate", "avg_return", "stability"])
    return out


def _build_today_display(today_df: pd.DataFrame) -> pd.DataFrame:
    if today_df.empty:
        return today_df
    cols = ["date", "rule_id", "trigger_reason", "risk_level", "position_suggestion"]
    return today_df[[c for c in cols if c in today_df.columns]].copy()


def _build_monitor_display(monitor_df: pd.DataFrame) -> pd.DataFrame:
    if monitor_df.empty:
        return monitor_df
    cols = [
        "rule_id",
        "baseline_win_rate",
        "recent_win_rate",
        "hit_rate_drop",
        "recent_expectancy",
        "pause_strategy",
        "status",
    ]
    out = monitor_df[[c for c in cols if c in monitor_df.columns]].copy()
    out = _format_percentage_columns(out, ["baseline_win_rate", "recent_win_rate", "hit_rate_drop"])
    out = _format_decimal_columns(out, ["recent_expectancy"], ndigits=4)
    return out

def _load_branch_raw(conn, sid: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
    SELECT
        b.stock_id,
        b.date,
        b.securities_trader_id AS branch_id,
        b.securities_trader AS branch_name,
        b.price,
        b.buy,
        b.sell,
        o.close
    FROM branch_price_daily b
    LEFT JOIN stock_ohlcv_daily o
      ON o.stock_id = b.stock_id
     AND o.date = b.date
    WHERE b.stock_id = ?
      AND b.date BETWEEN ? AND ?
    ORDER BY b.date ASC
    """
    return pd.read_sql(query, conn, params=(sid, start_date, end_date))


def _calc_expectancy(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    win_rate = (ret > 0).mean()
    avg_win = ret[ret > 0].mean() if (ret > 0).any() else 0.0
    avg_loss = abs(ret[ret <= 0].mean()) if (ret <= 0).any() else 0.0
    return float(win_rate * avg_win - (1.0 - win_rate) * avg_loss)


def _call_nim(cfg: dict, messages: list[dict]) -> str:
    llm_cfg = cfg.get("llm", {})
    api_key = llm_cfg.get("api_key")
    if not api_key:
        raise ValueError("尚未設定 llm.api_key，請先於 config.json 設定 NVIDIA API Key。")

    payload = {
        "model": get_llm_model(cfg, "chip"),
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _build_ai_strategy_brief(
    cfg: dict,
    sid: str,
    rules_df: pd.DataFrame,
    today_df: pd.DataFrame,
    monitor_df: pd.DataFrame,
) -> str:
    rules_csv = rules_df.head(12).to_csv(index=False)
    today_csv = today_df.head(20).to_csv(index=False)
    monitor_csv = monitor_df.head(20).to_csv(index=False)

    prompt = (
        "你是台股分點策略分析助理。"
        "請根據下列三張表，輸出『可執行結論』，格式如下：\n"
        "1) 今天可執行規則（最多3條）\n"
        "2) 今天觸發清單的風險與倉位建議（保守/中性/積極）\n"
        "3) 應暫停的規則與原因\n"
        "4) 明天開盤前要再檢查的3件事\n"
        "請盡量量化並引用欄位數值，不要空泛。\n\n"
        f"股票: {sid}\n"
        "[規則清單]\n"
        f"{rules_csv}\n"
        "[今天觸發清單]\n"
        f"{today_csv}\n"
        "[失效監控表]\n"
        f"{monitor_csv}"
    )
    return _call_nim(cfg, [{"role": "user", "content": prompt}])


def _build_grounded_strategy_tables(raw_df: pd.DataFrame, min_support: int, recent_weeks: int):
    features = prepare_branch_daily_features(raw_df)
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    day_close = (
        raw_df[["stock_id", "date", "close"]]
        .dropna()
        .drop_duplicates(subset=["stock_id", "date"], keep="last")
        .sort_values(["stock_id", "date"])
    )
    day_close["date"] = pd.to_datetime(day_close["date"])
    day_close["next_ret_1d"] = day_close.groupby("stock_id")["close"].shift(-1) / day_close["close"] - 1.0

    f = features.merge(day_close[["stock_id", "date", "next_ret_1d"]], on=["stock_id", "date"], how="left")

    q70 = float(f["vol_share"].quantile(0.70))
    q85 = float(f["vol_share"].quantile(0.85))

    rule_defs = [
        (
            "R1",
            f"主力偏買且成交量占比 >= {q70:.2%}",
            (f["net_vol"] > 0) & (f["vol_share"] >= q70),
        ),
        (
            "R2",
            "主力偏買且均價接近收盤（追價風險較低）",
            (f["net_vol"] > 0) & (f["price_impact"].abs() <= 0.01),
        ),
        (
            "R3",
            f"分點高度集中（成交量占比 >= {q85:.2%}）且偏買",
            (f["net_vol"] > 0) & (f["vol_share"] >= q85),
        ),
    ]

    rule_rows = []
    monitor_rows = []
    today_rows = []

    latest_date = pd.to_datetime(f["date"]).max()
    recent_cutoff = latest_date - pd.Timedelta(days=recent_weeks * 7)

    for rule_id, rule_desc, cond in rule_defs:
        sample = f[cond & f["next_ret_1d"].notna()].copy().sort_values("date")
        support = int(len(sample))
        if support < min_support:
            continue

        mid = support // 2
        first_half = sample.iloc[:mid] if mid > 0 else sample
        second_half = sample.iloc[mid:] if mid > 0 else sample
        first_wr = float((first_half["next_ret_1d"] > 0).mean()) if not first_half.empty else 0.0
        second_wr = float((second_half["next_ret_1d"] > 0).mean()) if not second_half.empty else 0.0

        win_rate = float((sample["next_ret_1d"] > 0).mean())
        avg_return = float(sample["next_ret_1d"].mean())
        expectancy = _calc_expectancy(sample["next_ret_1d"])
        stability = float(max(0.0, 1.0 - abs(second_wr - first_wr)))

        rule_rows.append(
            {
                "rule_id": rule_id,
                "rule_desc": rule_desc,
                "support": support,
                "win_rate": round(win_rate, 4),
                "avg_return": round(avg_return, 4),
                "expectancy": round(expectancy, 4),
                "stability": round(stability, 4),
            }
        )

        recent = sample[sample["date"] >= recent_cutoff]
        baseline = sample[sample["date"] < recent_cutoff]
        recent_wr = float((recent["next_ret_1d"] > 0).mean()) if not recent.empty else win_rate
        base_wr = float((baseline["next_ret_1d"] > 0).mean()) if not baseline.empty else win_rate
        hit_rate_drop = base_wr - recent_wr
        recent_expectancy = _calc_expectancy(recent["next_ret_1d"]) if not recent.empty else expectancy
        pause_strategy = (hit_rate_drop >= 0.15) or (recent_expectancy < 0)

        monitor_rows.append(
            {
                "rule_id": rule_id,
                "baseline_win_rate": round(base_wr, 4),
                "recent_win_rate": round(recent_wr, 4),
                "hit_rate_drop": round(hit_rate_drop, 4),
                "recent_expectancy": round(recent_expectancy, 4),
                "pause_strategy": bool(pause_strategy),
                "status": "暫停" if pause_strategy else "啟用",
            }
        )

        today_trigger = f[(f["date"] == latest_date) & cond]
        if not today_trigger.empty:
            branch_list = ", ".join(today_trigger["branch_id"].astype(str).unique()[:6])
            risk_level = "低" if (expectancy > 0 and stability >= 0.7) else "中" if expectancy > 0 else "高"
            position = "1.0x" if risk_level == "低" else "0.5x" if risk_level == "中" else "0.0x"
            today_rows.append(
                {
                    "date": latest_date,
                    "rule_id": rule_id,
                    "trigger_reason": f"{rule_desc}；觸發分點：{branch_list}",
                    "risk_level": risk_level,
                    "position_suggestion": position,
                }
            )

    rules_df = pd.DataFrame(rule_rows).sort_values(["expectancy", "stability"], ascending=[False, False]) if rule_rows else pd.DataFrame()
    today_df = pd.DataFrame(today_rows).sort_values(["risk_level", "rule_id"]) if today_rows else pd.DataFrame()
    monitor_df = pd.DataFrame(monitor_rows).sort_values(["pause_strategy", "hit_rate_drop"], ascending=[False, False]) if monitor_rows else pd.DataFrame()

    return rules_df, today_df, monitor_df


def show_winner_branch_system():
    st.header("🧠 AI 贏家分點追蹤與落地策略")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])

    if not universe:
        st.error("設定檔沒有 universe，請先在 config.json 設定 stock_id。")
        conn.close()
        return

    stock_options = {f"{s['stock_id']} {s.get('name', '')}".strip(): s["stock_id"] for s in universe}

    c1, c2, c3 = st.columns([1.7, 1.8, 1.5])
    with c1:
        selected_stock = st.selectbox("分析標的", list(stock_options.keys()))
        sid = stock_options[selected_stock]
    with c2:
        default_end = pd.to_datetime("today").date()
        default_start = (pd.to_datetime("today") - pd.Timedelta(days=180)).date()
        d_range = st.date_input("分析區間", value=[default_start, default_end])
        if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
            start_d = pd.to_datetime(d_range[0]).date().isoformat()
            end_d = pd.to_datetime(d_range[1]).date().isoformat()
        else:
            start_d = default_start.isoformat()
            end_d = default_end.isoformat()
    with c3:
        st.caption("策略輸出設定")
        min_support = st.slider("最小樣本數", min_value=10, max_value=120, value=30, step=5)
        recent_weeks = st.slider("失效監控最近週數", min_value=2, max_value=12, value=4, step=1)

    raw_df = _load_branch_raw(conn, sid, start_d, end_d)
    if raw_df.empty:
        st.warning("指定區間沒有分點資料，請調整日期。")
        conn.close()
        return

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df["branch_id"] = raw_df["branch_id"].astype(str)

    st.caption("此頁面已整合：需求1分點追蹤 + 新版落地策略三張表（規則、今日觸發、失效監控）。")

    top_quantile = st.slider("Top quantile（贏家門檻）", min_value=0.60, max_value=0.95, value=0.80, step=0.01)
    hhi_rise_window = st.slider("HHI 變化視窗（日）", min_value=5, max_value=30, value=10, step=1)
    compression_window = st.slider("價格壓縮視窗（日）", min_value=3, max_value=15, value=5, step=1)
    compression_threshold = st.slider("價格壓縮門檻", min_value=0.001, max_value=0.03, value=0.01, step=0.001)

    wb_cfg = WinnerBranchConfig(
        top_quantile=top_quantile,
        hhi_rise_window=hhi_rise_window,
        compression_window=compression_window,
        compression_threshold=compression_threshold,
    )

    st.divider()
    st.subheader("🎯 需求 1：贏家分點追蹤")

    if "winner_track_cache" not in st.session_state:
        st.session_state["winner_track_cache"] = None

    if st.button("🚀 執行需求1：贏家分點追蹤", use_container_width=True):
        branch_lookup = (
            raw_df[["branch_id", "branch_name"]]
            .dropna(subset=["branch_id"])
            .astype({"branch_id": str})
            .drop_duplicates(subset=["branch_id"], keep="last")
        )

        chip = ChipStrategyAI.from_dataframe(
            raw_df,
            start_date=start_d,
            end_date=end_d,
            config=ChipStrategyConfig(winner_cfg=wb_cfg),
        )
        track = chip.track_winner_branches()
        track["winner_rating"]["branch_id"] = track["winner_rating"]["branch_id"].astype(str)
        st.session_state["winner_track_cache"] = {
            "sid": sid,
            "start_d": start_d,
            "end_d": end_d,
            "track": track,
            "branch_lookup": branch_lookup,
        }

    cache = st.session_state.get("winner_track_cache")
    if cache and cache.get("sid") == sid and cache.get("start_d") == start_d and cache.get("end_d") == end_d:
        track = cache["track"]
        branch_lookup = cache["branch_lookup"]

        st.subheader("🏆 Winner Rating（分點評級）")
        _render_winner_rating_table(track["winner_rating"], branch_lookup)

        st.subheader("🔔 Daily Alerts（A/B/C）")
        alerts_df = track["daily_alerts"].copy()
        alerts_df["branch_id"] = alerts_df["branch_id"].astype(str)
        daily_alerts_display = alerts_df.merge(branch_lookup, on="branch_id", how="left")
        daily_alerts_display["branch_name"] = daily_alerts_display["branch_name"].fillna("-")
        daily_alerts_display["alert_level_desc"] = daily_alerts_display["alert_level"].map(ALERT_LEVEL_DESC).fillna("-")
        st.dataframe(_to_display_df(daily_alerts_display), use_container_width=True, hide_index=True)

        st.subheader("🧪 Strategy Candidates（集中度策略候選）")
        cand = track["strategy_candidates"]
        if "strategy_candidate" in cand.columns:
            cand = cand[cand["strategy_candidate"] == True]
        st.dataframe(_to_display_df(cand), use_container_width=True, hide_index=True)

        with st.expander("查看 Concentration Features 原始輸出"):
            st.dataframe(_to_display_df(track["concentration"]), use_container_width=True, hide_index=True)
    else:
        st.info("請先點擊「執行需求1：贏家分點追蹤」。")

    st.divider()
    st.subheader("🧭 新版落地策略（先跑給你看）")
    st.caption("已移除舊版需求2（XGBoost + 參數掃描），改為固定輸出三張落地表。")

    if st.button("⚙️ 產生落地策略三張表", use_container_width=True):
        rules_df, today_df, monitor_df = _build_grounded_strategy_tables(raw_df, min_support=min_support, recent_weeks=recent_weeks)
        st.session_state["grounded_strategy_cache"] = {
            "sid": sid,
            "start_d": start_d,
            "end_d": end_d,
            "rules": rules_df,
            "today": today_df,
            "monitor": monitor_df,
        }

    grounded = st.session_state.get("grounded_strategy_cache")
    if grounded and grounded.get("sid") == sid and grounded.get("start_d") == start_d and grounded.get("end_d") == end_d:
        rules_df = grounded["rules"]
        today_df = grounded["today"]
        monitor_df = grounded["monitor"]

        # readability-focused layout: KPIs + tabs + formatted columns
        rules_display = _build_rules_display(rules_df)
        today_display = _build_today_display(today_df)
        monitor_display = _build_monitor_display(monitor_df)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("規則數", int(len(rules_df)))
        with k2:
            st.metric("今日觸發", int(len(today_df)))
        with k3:
            pause_count = int(monitor_df["pause_strategy"].sum()) if "pause_strategy" in monitor_df.columns and not monitor_df.empty else 0
            st.metric("建議暫停規則", pause_count)
        with k4:
            top_expectancy = float(rules_df["expectancy"].max()) if "expectancy" in rules_df.columns and not rules_df.empty else 0.0
            st.metric("最高期望值", f"{top_expectancy:.4f}")

        tab1, tab2, tab3 = st.tabs([
            "1) 規則清單",
            "2) 今天觸發",
            "3) 失效監控",
        ])

        with tab1:
            st.caption("已將勝率/平均報酬/穩定度轉為百分比，並固定欄位順序。")
            if rules_display.empty:
                st.warning("規則樣本不足，請降低最小樣本數或拉長日期區間。")
            else:
                st.dataframe(_to_display_df(rules_display), use_container_width=True, hide_index=True, height=420)

        with tab2:
            st.caption("優先查看 risk_level 與 position_suggestion。")
            if today_display.empty:
                st.info("今天沒有規則被觸發。")
            else:
                st.dataframe(_to_display_df(today_display), use_container_width=True, hide_index=True, height=360)

        with tab3:
            st.caption("命中率下滑與近期期望值會直接影響暫停建議。")
            if monitor_display.empty:
                st.info("目前沒有可監控規則（可能是樣本不足）。")
            else:
                st.dataframe(_to_display_df(monitor_display), use_container_width=True, hide_index=True, height=360)

        st.markdown("**🤖 AI 策略解讀（使用 NVIDIA 模型）**")
        st.caption("這段會呼叫你在 config.json 設定的 llm.models.chip（或 llm.model）產生落地執行摘要。")
        if st.button("🤖 產生 AI 解讀建議", use_container_width=True):
            with st.spinner("AI 分析三張表中，請稍候..."):
                try:
                    ai_text = _build_ai_strategy_brief(cfg, sid, rules_df, today_df, monitor_df)
                    st.markdown(ai_text)
                except Exception as e:
                    st.error(f"AI 分析失敗：{e}")
    else:
        st.info("請點擊「產生落地策略三張表」。")

    conn.close()
