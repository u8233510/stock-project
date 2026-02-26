import streamlit as st
import pandas as pd

import database
from utility.branch_anomaly_detection import BranchAnomalyConfig, build_anomaly_outputs


COLUMN_LABELS = {
    "week": "週別",
    "start_date": "起始日",
    "end_date": "結束日",
    "observed_days": "觀察天數",
    "event_days": "異常事件天數",
    "avg_score": "平均異常分數",
    "peak_score": "區間最高分",
    "stock_id": "股票代號",
    "branch_id": "分點代號",
    "branch_name": "分點名稱",
    "anomaly_score": "異常分數",
    "anomaly_level": "異常等級",
    "z_net_20": "淨買賣量 Z 分數(20日)",
    "z_gross_20": "總成交量 Z 分數(20日)",
    "vol_share": "分點成交占比",
    "flag_accumulation": "疑似吸籌",
    "flag_distribution": "疑似出貨",
    "buy_event_days": "偏買天數",
    "sell_event_days": "偏賣天數",
    "net_flow_ratio": "淨流向占比",
    "dominant_action": "行為判讀",
    "stealth_accum_days": "偷偷進貨天數",
    "stealth_dist_days": "偷偷在賣天數",
    "supply_to_market_days": "供給籌碼天數",
    "absorb_from_market_days": "承接籌碼天數",
    "events": "事件筆數",
    "avg_vol_share": "平均成交占比",
    "adj_avg_score": "穩健異常分數",
    "adj_avg_vol_share": "穩健成交占比",
    "short_category": "短線分類",
    "long_category": "長線分類",
    "short_score": "短線分數",
    "long_score": "長線分數",
}

LEVEL_LABELS = {
    "normal": "一般",
    "medium": "中度",
    "major": "高度",
}

ACTION_ORDER = ["偷偷進貨", "偷偷在賣", "供給籌碼", "承接籌碼", "偏買", "偏賣", "中性"]

NINE_CATEGORY_ORDER = [
    "主力攻擊",
    "主力倒貨",
    "主力進貨",
    "主力出貨",
    "偷偷買",
    "偷偷賣",
    "當沖主導",
    "隔日沖主導",
    "無明顯主力活動",
]


def _localize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "anomaly_level" in out.columns:
        out["anomaly_level"] = out["anomaly_level"].map(LEVEL_LABELS).fillna(out["anomaly_level"])
    return out.rename(columns=COLUMN_LABELS)


def _format_main_table(df: pd.DataFrame):
    localized = _localize_table(df)
    formatters = {
        "異常分數": "{:.2f}",
        "平均異常分數": "{:.2f}",
        "區間最高分": "{:.2f}",
        "淨買賣量 Z 分數(20日)": "{:.2f}",
        "總成交量 Z 分數(20日)": "{:.2f}",
        "分點成交占比": "{:.2%}",
        "淨流向占比": "{:.2%}",
    }
    return localized.style.format({k: v for k, v in formatters.items() if k in localized.columns})


def _render_column_guide() -> None:
    with st.expander("欄位說明（重新設計版）", expanded=False):
        st.markdown(
            """
            **先看三個欄位就好：**
            1. **異常分數**：越高代表該分點在區間內越不尋常。  
            2. **行為判讀**：直接告訴你是 `偷偷進貨 / 偷偷在賣 / 供給籌碼 / 承接籌碼 / 偏買 / 偏賣 / 中性`。  
            3. **淨流向占比**：區間淨買賣量 ÷ 區間總成交量，判斷買賣力道。  

            其他欄位可用於進一步確認：
            - **偏買天數 / 偏賣天數**：方向持續性。
            - **偷偷進貨天數 / 偷偷在賣天數**：低占比但有方向性。
            - **供給籌碼天數 / 承接籌碼天數**：是否在對盤面提供或吸收籌碼。
            - **疑似吸籌 / 疑似出貨**：高強度規則訊號。
            """
        )




def _build_long_short_labels(events_df: pd.DataFrame) -> pd.DataFrame:
    """Create short/long horizon 9-category labels and ranking scores."""
    out = events_df.copy()

    event_ratio = (out["event_days"] / out["observed_days"].replace(0, pd.NA)).fillna(0.0)
    buy_ratio = (out["buy_event_days"] / out["event_days"].replace(0, pd.NA)).fillna(0.0)
    sell_ratio = (out["sell_event_days"] / out["event_days"].replace(0, pd.NA)).fillna(0.0)

    out["is_daytrade_like"] = (
        (out["event_days"] >= 2)
        & (out["buy_event_days"] >= 1)
        & (out["sell_event_days"] >= 1)
        & (out["net_flow_ratio"].abs() <= 0.05)
        & (out["vol_share"] >= 0.05)
    )

    out["is_nextday_like"] = (
        (out["event_days"] >= 3)
        & (buy_ratio.between(0.35, 0.65))
        & (sell_ratio.between(0.35, 0.65))
        & (out["net_flow_ratio"].abs() <= 0.08)
    )

    out["is_attack"] = (
        (out["anomaly_score"] >= 70)
        & (out["event_days"] >= 2)
        & (out["net_flow_ratio"] >= 0.12)
        & ((out["flag_accumulation"]) | (out["dominant_action"].isin(["承接籌碼", "偏買"])))
    )

    out["is_dump"] = (
        (out["anomaly_score"] >= 70)
        & (out["event_days"] >= 2)
        & (out["net_flow_ratio"] <= -0.12)
        & ((out["flag_distribution"]) | (out["dominant_action"].isin(["供給籌碼", "偏賣"])))
    )

    out["is_build_long"] = (
        (out["event_days"] >= 2)
        & (event_ratio >= 0.12)
        & (out["net_flow_ratio"] >= 0.08)
        & (out["vol_share"] >= 0.04)
    )
    out["is_distribute_long"] = (
        (out["event_days"] >= 2)
        & (event_ratio >= 0.12)
        & (out["net_flow_ratio"] <= -0.08)
        & (out["vol_share"] >= 0.04)
    )

    out["is_stealth_buy"] = (
        (out["stealth_accum_days"] >= 2)
        & (out["net_flow_ratio"] > 0)
        & (out["vol_share"] < 0.08)
    )
    out["is_stealth_sell"] = (
        (out["stealth_dist_days"] >= 2)
        & (out["net_flow_ratio"] < 0)
        & (out["vol_share"] < 0.08)
    )

    short_conditions = [
        out["is_attack"],
        out["is_dump"],
        out["is_build_long"],
        out["is_distribute_long"],
        out["is_stealth_buy"],
        out["is_stealth_sell"],
        out["is_daytrade_like"],
        out["is_nextday_like"],
    ]
    short_choices = NINE_CATEGORY_ORDER[:-1]
    out["short_category"] = pd.Series(pd.NA, index=out.index, dtype="object")
    for condition, label in zip(short_conditions, short_choices):
        out.loc[out["short_category"].isna() & condition, "short_category"] = label
    out["short_category"] = out["short_category"].fillna("無明顯主力活動")

    long_conditions = [
        out["is_attack"] & (event_ratio >= 0.10),
        out["is_dump"] & (event_ratio >= 0.10),
        out["is_build_long"] & (~out["is_attack"]),
        out["is_distribute_long"] & (~out["is_dump"]),
        out["is_stealth_buy"],
        out["is_stealth_sell"],
        out["is_daytrade_like"] & (event_ratio >= 0.10),
        out["is_nextday_like"] & (event_ratio >= 0.10),
    ]
    out["long_category"] = pd.Series(pd.NA, index=out.index, dtype="object")
    for condition, label in zip(long_conditions, short_choices):
        out.loc[out["long_category"].isna() & condition, "long_category"] = label
    out["long_category"] = out["long_category"].fillna("無明顯主力活動")

    out["short_score"] = (
        out["anomaly_score"] * 0.5
        + out["event_days"].clip(upper=10) * 3.0
        + (out["net_flow_ratio"].abs() * 100).clip(upper=20)
    ).round(2)
    out["long_score"] = (
        out["anomaly_score"] * 0.4
        + (event_ratio * 100).clip(upper=30)
        + (out["vol_share"] * 100).clip(upper=20)
        + (out["net_flow_ratio"].abs() * 100).clip(upper=20)
    ).round(2)

    return out


def _render_nine_category_tab(df: pd.DataFrame, *, category_col: str, score_col: str, title_prefix: str) -> None:
    st.caption(f"{title_prefix}以九大類型分組，組內依分數高到低排序。")
    ranking_cols = [
        "branch_id",
        "branch_name",
        score_col,
        "anomaly_score",
        "event_days",
        "net_flow_ratio",
        "vol_share",
        "dominant_action",
    ]

    for category in NINE_CATEGORY_ORDER:
        subset = df[df[category_col] == category].copy()
        subset = subset.sort_values([score_col, "anomaly_score", "event_days"], ascending=[False, False, False])
        st.markdown(f"#### {category}（{len(subset)} 個分點）")
        if subset.empty:
            st.caption("目前區間內沒有符合條件的分點。")
            continue
        st.dataframe(
            _format_main_table(subset[ranking_cols]),
            use_container_width=True,
            hide_index=True,
        )

def _load_branch_raw(conn, sid: str, start_date: str, end_date: str) -> pd.DataFrame:
    sql = """
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
    JOIN stock_ohlcv_daily o
      ON b.stock_id = o.stock_id
     AND b.date = o.date
    WHERE b.stock_id = ?
      AND b.date >= ?
      AND b.date <= ?
    ORDER BY b.date ASC
    """
    return pd.read_sql(sql, conn, params=(sid, start_date, end_date))


def _render_summary_cards(anomaly_events: pd.DataFrame, watchlist: pd.DataFrame) -> None:
    if anomaly_events.empty:
        return

    major_count = int((anomaly_events["anomaly_level"] == "major").sum())
    non_neutral = int((anomaly_events["dominant_action"] != "中性").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("分點總數", f"{len(anomaly_events)}")
    c2.metric("可追蹤名單", f"{len(watchlist)}")
    c3.metric("高度異常", f"{major_count}")
    c4.metric("有明確行為", f"{non_neutral}")


def show_branch_anomaly():
    st.markdown("### 🚨 分點異常偵測")
    st.caption("重新設計：先看結論，再看細節，避免欄位過載。")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)

    universe = cfg.get("universe", [])
    if not universe:
        st.warning("config.json 未設定 universe。")
        return

    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.8, 1.8, 1.2])
        with c1:
            sid_label = st.selectbox("標的", list(stock_options.keys()))
            sid = stock_options[sid_label]

        date_bounds = conn.execute(
            """
            SELECT MIN(date), MAX(date)
            FROM branch_price_daily
            WHERE stock_id = ?
            """,
            (sid,),
        ).fetchone()

        min_date_raw, max_date_raw = date_bounds if date_bounds else (None, None)
        min_date = pd.to_datetime(min_date_raw).date() if min_date_raw else None
        max_date = pd.to_datetime(max_date_raw).date() if max_date_raw else None

        if not min_date or not max_date:
            st.info("此標的尚無分點資料。")
            return

        with c2:
            date_range = st.date_input(
                "日期區間",
                value=[max(min_date, max_date - pd.Timedelta(days=120)), max_date],
                min_value=min_date,
                max_value=max_date,
            )

        with c3:
            top_n = st.number_input("顯示筆數", min_value=10, max_value=200, value=50, step=10)

        if isinstance(date_range, tuple) or isinstance(date_range, list):
            if len(date_range) != 2:
                st.warning("請選擇開始與結束日期。")
                return
            start_d, end_d = date_range
        else:
            st.warning("請選擇開始與結束日期。")
            return

        c4, c5, c6 = st.columns(3)
        with c4:
            z_threshold = st.slider("Z 分數門檻", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        with c5:
            vol_share_threshold = st.slider("分點成交占比門檻", min_value=0.03, max_value=0.30, value=0.10, step=0.01)
        with c6:
            medium_score = st.slider("追蹤分數門檻", min_value=40, max_value=90, value=60, step=5)

        run = st.button("執行異常偵測", type="primary", use_container_width=True)

    state_key = "branch_anomaly_result"
    if run:
        raw_df = _load_branch_raw(conn, sid, str(start_d), str(end_d))
        if raw_df.empty:
            st.info("選定區間內無可用資料。")
            st.session_state.pop(state_key, None)
            return

        detector_cfg = BranchAnomalyConfig(
            short_window=20,
            long_window=60,
            z_threshold=float(z_threshold),
            gross_z_threshold=float(z_threshold),
            vol_share_threshold=float(vol_share_threshold),
            medium_score_threshold=float(medium_score),
        )

        anomaly_events, watchlist, weekly_report = build_anomaly_outputs(raw_df, detector_cfg)
        classified_events = _build_long_short_labels(anomaly_events)
        classified_watchlist = _build_long_short_labels(watchlist) if not watchlist.empty else watchlist
        st.session_state[state_key] = {
            "sid": sid,
            "start_d": str(start_d),
            "end_d": str(end_d),
            "anomaly_events": anomaly_events,
            "watchlist": watchlist,
            "weekly_report": weekly_report,
            "classified_events": classified_events,
            "classified_watchlist": classified_watchlist,
        }
    else:
        cached_result = st.session_state.get(state_key)
        if cached_result is None:
            return

        anomaly_events = cached_result["anomaly_events"]
        watchlist = cached_result["watchlist"]
        weekly_report = cached_result["weekly_report"]
        classified_events = cached_result["classified_events"]
        classified_watchlist = cached_result["classified_watchlist"]

    st.success(f"完成：共彙整 {len(anomaly_events)} 個分點，追蹤名單 {len(watchlist)} 筆。")
    _render_summary_cards(anomaly_events, watchlist)
    _render_column_guide()

    filter_col1, filter_col2 = st.columns([1.4, 1.0])
    with filter_col1:
        selected_actions = st.multiselect(
            "篩選行為判讀",
            options=ACTION_ORDER,
            default=[a for a in ACTION_ORDER if a != "中性"],
        )
    with filter_col2:
        min_event_days = st.slider("最少異常事件天數", min_value=0, max_value=20, value=1, step=1)

    filtered_events = classified_events.copy()
    if selected_actions:
        filtered_events = filtered_events[filtered_events["dominant_action"].isin(selected_actions)]
    filtered_events = filtered_events[filtered_events["event_days"] >= min_event_days]

    main_cols = [
        "branch_id",
        "branch_name",
        "anomaly_score",
        "anomaly_level",
        "dominant_action",
        "net_flow_ratio",
        "event_days",
        "buy_event_days",
        "sell_event_days",
        "vol_share",
    ]

    detail_cols = [
        "branch_id",
        "branch_name",
        "avg_score",
        "peak_score",
        "z_net_20",
        "z_gross_20",
        "stealth_accum_days",
        "stealth_dist_days",
        "supply_to_market_days",
        "absorb_from_market_days",
        "flag_accumulation",
        "flag_distribution",
        "observed_days",
    ]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["異常排行榜", "追蹤名單", "深度欄位", "每週摘要", "短線九大分類", "長線九大分類"])

    with tab1:
        st.dataframe(
            _format_main_table(filtered_events[main_cols].head(int(top_n))),
            use_container_width=True,
        )

    with tab2:
        watch_filtered = classified_watchlist.copy()
        if selected_actions:
            watch_filtered = watch_filtered[watch_filtered["dominant_action"].isin(selected_actions)]
        watch_filtered = watch_filtered[watch_filtered["event_days"] >= min_event_days]
        st.dataframe(
            _format_main_table(watch_filtered[main_cols].head(int(top_n))),
            use_container_width=True,
        )

    with tab3:
        st.dataframe(
            _format_main_table(filtered_events[detail_cols].head(int(top_n))),
            use_container_width=True,
        )

    with tab4:
        wc1, wc2, _ = st.columns([1.2, 1.2, 1.6])
        with wc1:
            min_weekly_events = st.slider("每週摘要最少事件筆數", min_value=1, max_value=20, value=2, step=1)
        with wc2:
            smoothing_k = st.slider("穩健分數平滑係數", min_value=1, max_value=20, value=5, step=1)

        weekly_filtered = weekly_report[weekly_report["events"] >= min_weekly_events].copy()

        if weekly_filtered.empty:
            st.info("目前條件下沒有符合最少事件筆數的每週資料。")
        else:
            score_prior = float(weekly_report["avg_score"].mean()) if not weekly_report.empty else 0.0
            vol_prior = float(weekly_report["avg_vol_share"].mean()) if not weekly_report.empty else 0.0
            weekly_filtered["adj_avg_score"] = (
                (weekly_filtered["avg_score"] * weekly_filtered["events"] + score_prior * smoothing_k)
                / (weekly_filtered["events"] + smoothing_k)
            )
            weekly_filtered["adj_avg_vol_share"] = (
                (weekly_filtered["avg_vol_share"] * weekly_filtered["events"] + vol_prior * smoothing_k)
                / (weekly_filtered["events"] + smoothing_k)
            )

            weekly_filtered = weekly_filtered.sort_values(
                ["week", "adj_avg_score", "events", "adj_avg_vol_share"],
                ascending=[False, False, False, False],
            )

            weekly_display_cols = [
                "week",
                "anomaly_level",
                "events",
                "avg_score",
                "adj_avg_score",
                "avg_vol_share",
                "adj_avg_vol_share",
            ]
            weekly_localized = _localize_table(weekly_filtered[weekly_display_cols])
            st.caption("已套用穩健分數：事件筆數少的週別會被平滑，避免 1 筆極端值衝到最前面。")
            st.dataframe(
                weekly_localized.style.format(
                    {
                        "週別": lambda d: pd.to_datetime(d).strftime("%Y-%m-%d") if pd.notna(d) else "",
                        "平均異常分數": "{:.3f}",
                        "穩健異常分數": "{:.3f}",
                        "平均成交占比": "{:.2%}",
                        "穩健成交占比": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

    with tab5:
        _render_nine_category_tab(filtered_events, category_col="short_category", score_col="short_score", title_prefix="短線")

    with tab6:
        _render_nine_category_tab(filtered_events, category_col="long_category", score_col="long_score", title_prefix="長線")
