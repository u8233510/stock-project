import pandas as pd
import streamlit as st

import database
from utility.accumulation_scan import AccumulationScanConfig, run_accumulation_scan


COLUMN_LABELS = {
    "stock_id": "股票代號",
    "stability": "買入穩定度",
    "is_shifting": "結構轉強",
    "has_anomaly": "異常訊號",
    "final_score": "綜合評分",
    "coordination_score": "集團協同性",
    "participant_diff": "家數差",
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
    available_days = max((pd.Timestamp(end_d) - pd.Timestamp(start_d)).days + 1, 1)

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        lookback_days = st.slider("掃描回看天數", min_value=20, max_value=180, value=60, step=5, key="acc_scan_lookback")
    with c4:
        min_stability = st.slider("最低穩定度", min_value=0.3, max_value=0.95, value=0.5, step=0.05, key="acc_scan_stability")
    with c5:
        min_coord = st.slider("最低協同性", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key="acc_scan_coord")
    with c6:
        score_threshold = st.slider("最低綜合分數(%)", min_value=0, max_value=100, value=50, step=5, key="acc_scan_score_threshold")


    state_key = "accumulation_scan_result"
    running_key = "acc_scan_running"
    progress_key = "acc_scan_progress"
    page_key = "acc_scan_page"

    if running_key not in st.session_state:
        st.session_state[running_key] = False
    if progress_key not in st.session_state:
        st.session_state[progress_key] = ""

    run = st.button(
        "執行全市場低檔潛伏掃描",
        type="primary",
        use_container_width=True,
        key="acc_scan_run",
        disabled=st.session_state[running_key],
    )

    effective_lookback = min(int(lookback_days), int(available_days))
    if effective_lookback < int(lookback_days):
        st.caption(
            f"⚠️ 目前日期區間僅有 {available_days} 天資料，將自動以 {effective_lookback} 天進行掃描。"
        )

    if run:
        st.session_state[running_key] = True
        st.session_state[progress_key] = "讀取分點資料中..."
        progress_bar = st.progress(0, text="掃描中，請稍候...")

        raw_df = _load_scan_raw(conn, str(start_d), str(end_d))
        progress_bar.progress(40, text="已完成資料讀取，開始計算模型...")

        if raw_df.empty:
            st.info("選定區間內無可用資料。")
            st.session_state.pop(state_key, None)
            st.session_state[running_key] = False
            st.session_state[progress_key] = ""
            return

        scan_cfg = AccumulationScanConfig(
            lookback_days=effective_lookback,
            min_stability=float(min_stability),
            coord_threshold=float(min_coord),
            final_score_threshold=int(score_threshold),
        )
        result_df = run_accumulation_scan(raw_df, scan_cfg)
        progress_bar.progress(100, text="掃描完成")
        st.session_state[state_key] = result_df
        st.session_state[running_key] = False
        st.session_state[progress_key] = ""
        st.session_state[page_key] = 1

    if state_key not in st.session_state:
        st.info("請先設定條件後，點擊「執行全市場低檔潛伏掃描」。")
        return

    result_df = st.session_state[state_key]

    if result_df.empty:
        st.warning("沒有股票同時符合進階條件：穩定吸籌、結構轉折或高度協同。")
        return

    display_df = result_df.head(int(top_n)).copy()
    page_size = 20
    total_rows = len(display_df)
    total_pages = max((total_rows - 1) // page_size + 1, 1)

    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    current_page = int(st.session_state[page_key])
    current_page = min(max(current_page, 1), total_pages)
    st.session_state[page_key] = current_page

    page_start = (current_page - 1) * page_size
    page_end = page_start + page_size
    page_df = display_df.iloc[page_start:page_end].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("符合條件股票數", f"{len(result_df)}")
    c2.metric("平均穩定度", f"{result_df['stability'].mean():.2f}")
    c3.metric("有結構轉強", f"{int(result_df['is_shifting'].sum())}")
    c4.metric("有異常訊號", f"{int(result_df['has_anomaly'].sum())}")

    p1, p2, p3 = st.columns([1, 1, 4])
    with p1:
        if st.button("上一頁", use_container_width=True, disabled=current_page <= 1):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with p2:
        if st.button("下一頁", use_container_width=True, disabled=current_page >= total_pages):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    with p3:
        st.caption(f"第 {current_page} / {total_pages} 頁（每頁 {page_size} 筆，顯示 {total_rows} 筆）")

    localized = page_df.rename(columns=COLUMN_LABELS)
    st.dataframe(localized, use_container_width=True, hide_index=True)
