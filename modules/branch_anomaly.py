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
}

LEVEL_LABELS = {
    "normal": "一般",
    "medium": "中度",
    "major": "高度",
}

ACTION_ORDER = ["偷偷進貨", "偷偷在賣", "供給籌碼", "承接籌碼", "偏買", "偏賣", "中性"]


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


def _load_branch_raw(conn, sid: str, start_date: str, end_date: str) -> pd.DataFrame:
    sql = """
    SELECT
        b.stock_id,
        b.date,
        b.securities_trader_id AS branch_id,
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

    if not run:
        return

    raw_df = _load_branch_raw(conn, sid, str(start_d), str(end_d))
    if raw_df.empty:
        st.info("選定區間內無可用資料。")
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

    filtered_events = anomaly_events.copy()
    if selected_actions:
        filtered_events = filtered_events[filtered_events["dominant_action"].isin(selected_actions)]
    filtered_events = filtered_events[filtered_events["event_days"] >= min_event_days]

    main_cols = [
        "stock_id",
        "branch_id",
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
        "stock_id",
        "branch_id",
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

    tab1, tab2, tab3, tab4 = st.tabs(["異常排行榜", "追蹤名單", "深度欄位", "每週摘要"])

    with tab1:
        st.dataframe(
            _format_main_table(filtered_events[main_cols].head(int(top_n))),
            use_container_width=True,
        )

    with tab2:
        watch_filtered = watchlist.copy()
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
        st.dataframe(
            _localize_table(weekly_report).style.format({"avg_score": "{:.2f}", "avg_vol_share": "{:.2%}"}),
            use_container_width=True,
        )
