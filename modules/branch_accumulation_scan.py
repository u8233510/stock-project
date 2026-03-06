import pandas as pd
import streamlit as st

import database
from utility.accumulation_scan import AccumulationScanConfig, run_accumulation_scan


COLUMN_LABELS = {
    "stock_id": "股票代號",
    "stability": "買入穩定度",
    "is_shifting": "結構轉強",
    "has_anomaly": "異常訊號",
    "coordination_score": "集團協同性",
}


def _load_scan_raw(conn, start_date: str, end_date: str) -> pd.DataFrame:
    table_cols = {row[1] for row in conn.execute("PRAGMA table_info(branch_trader_daily_detail)").fetchall()}
    if "price" in table_cols:
        price_expr = "b.price"
    elif "close" in table_cols:
        price_expr = "b.close AS price"
    else:
        price_expr = "NULL AS price"

    sql = f"""
    SELECT
        b.stock_id,
        b.date,
        b.securities_trader_id AS branch_id,
        b.securities_trader AS branch_name,
        b.buy,
        b.sell,
        {price_expr}
    FROM branch_trader_daily_detail b
    WHERE b.date BETWEEN ? AND ?
    ORDER BY b.stock_id ASC, b.securities_trader_id ASC, b.date ASC
    """
    return pd.read_sql(sql, conn, params=(start_date, end_date))


def show_branch_accumulation_scan():
    st.markdown("### 🕵️ 低檔潛伏分點掃描")
    st.caption("以多因子模型掃描：買入穩定度 + 結構轉折 + 異常偵測 + 分點協同性。")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)

    bounds = conn.execute(
        """
        SELECT MIN(date), MAX(date)
        FROM branch_trader_daily_detail
        """
    ).fetchone()

    min_date_raw, max_date_raw = bounds if bounds else (None, None)
    min_date = pd.to_datetime(min_date_raw).date() if min_date_raw else None
    max_date = pd.to_datetime(max_date_raw).date() if max_date_raw else None

    if not min_date or not max_date:
        st.info("目前 branch_trader_daily_detail 尚無資料。")
        return

    c1, c2 = st.columns([2.2, 1.0])
    with c1:
        date_range = st.date_input(
            "日期區間",
            value=[max(min_date, max_date - pd.Timedelta(days=120)), max_date],
            min_value=min_date,
            max_value=max_date,
            key="acc_scan_date_range",
        )
    with c2:
        top_n = st.number_input("顯示筆數", min_value=10, max_value=500, value=100, step=10, key="acc_scan_top_n")

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.warning("請選擇開始與結束日期。")
        return

    start_d, end_d = date_range

    c3, c4, c5 = st.columns(3)
    with c3:
        lookback_days = st.slider("掃描回看天數", min_value=20, max_value=180, value=60, step=5, key="acc_scan_lookback")
    with c4:
        min_stability = st.slider("最低穩定度", min_value=0.3, max_value=0.95, value=0.7, step=0.05, key="acc_scan_stability")
    with c5:
        min_coord = st.slider("最低協同性", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key="acc_scan_coord")

    running_key = "acc_scan_running"
    if running_key not in st.session_state:
        st.session_state[running_key] = False

    run = st.button(
        "執行全市場低檔潛伏掃描",
        type="primary",
        use_container_width=True,
        key="acc_scan_run",
        disabled=st.session_state[running_key],
    )

    state_key = "accumulation_scan_result"
    if run and not st.session_state[running_key]:
        st.session_state[running_key] = True
        st.rerun()

    if st.session_state[running_key]:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        try:
            progress_text.info("掃描進行中：1/3 讀取資料中...")
            progress_bar.progress(15)
            raw_df = _load_scan_raw(conn, str(start_d), str(end_d))

            if raw_df.empty:
                st.info("選定區間內無可用資料。")
                st.session_state.pop(state_key, None)
                return

            progress_text.info("掃描進行中：2/3 計算多因子分數...")
            progress_bar.progress(60)
            scan_cfg = AccumulationScanConfig(
                lookback_days=int(lookback_days),
                min_stability=float(min_stability),
                coord_threshold=float(min_coord),
            )
            result_df = run_accumulation_scan(raw_df, scan_cfg)
            st.session_state[state_key] = result_df

            progress_text.info("掃描進行中：3/3 整理結果...")
            progress_bar.progress(100)
            progress_text.success("掃描完成。")
        finally:
            st.session_state[running_key] = False

    if state_key not in st.session_state:
        st.info("請先設定條件後，點擊「執行全市場低檔潛伏掃描」。")
        return

    result_df = st.session_state[state_key]

    if result_df.empty:
        st.warning("沒有股票同時符合進階條件：穩定吸籌、結構轉折或高度協同。")
        return

    display_df = result_df.head(int(top_n)).copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("符合條件股票數", f"{len(result_df)}")
    c2.metric("平均穩定度", f"{result_df['stability'].mean():.2f}")
    c3.metric("有結構轉強", f"{int(result_df['is_shifting'].sum())}")
    c4.metric("有異常訊號", f"{int(result_df['has_anomaly'].sum())}")

    localized = display_df.rename(columns=COLUMN_LABELS)
    st.dataframe(localized, use_container_width=True, hide_index=True)
