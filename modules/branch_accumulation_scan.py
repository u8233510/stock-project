import pandas as pd
import streamlit as st

import database
from utility.accumulation_scan import AccumulationScanConfig, run_accumulation_scan


COLUMN_LABELS = {
    "stock_id": "股票代號",
    "branch_id": "分點代號",
    "branch_name": "分點名稱",
    "start_date": "訊號起始日",
    "end_date": "訊號結束日",
    "consecutive_days": "連續買超天數",
    "net_buy_total": "區間買超張數",
    "avg_buy_sell_ratio": "平均買賣比",
    "avg_volume_share": "平均成交佔比",
    "signal_count": "符合區段數",
    "latest_signal_end": "最近訊號日",
}


def _load_scan_raw(conn, start_date: str, end_date: str) -> pd.DataFrame:
    sql = """
    SELECT
        b.stock_id,
        b.date,
        b.securities_trader_id AS branch_id,
        b.securities_trader AS branch_name,
        b.buy,
        b.sell,
        o.Trading_Volume
    FROM branch_trader_daily_detail b
    JOIN stock_ohlcv_daily o
      ON o.stock_id = b.stock_id
     AND o.date = b.date
    WHERE b.date BETWEEN ? AND ?
    ORDER BY b.stock_id ASC, b.securities_trader_id ASC, b.date ASC
    """
    return pd.read_sql(sql, conn, params=(start_date, end_date))


def show_branch_accumulation_scan():
    st.markdown("### 🕵️ 低檔潛伏分點掃描 (The Accumulation Scan)")
    st.caption("全市場掃描器：不用先選股票，直接找出『連續買超、幾乎不賣、且吃下成交量』的分點。")

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
        min_days = st.slider("連續買超天數", min_value=2, max_value=15, value=3, step=1, key="acc_scan_min_days")
    with c4:
        min_ratio = st.slider("最低買賣比", min_value=3.0, max_value=100.0, value=10.0, step=1.0, key="acc_scan_min_ratio")
    with c5:
        min_share = st.slider("最低成交佔比", min_value=0.01, max_value=0.30, value=0.05, step=0.01, key="acc_scan_min_share")

    run = st.button("執行全市場低檔潛伏掃描", type="primary", use_container_width=True, key="acc_scan_run")

    state_key = "accumulation_scan_result"
    if run:
        with st.spinner("掃描中，請稍候..."):
            raw_df = _load_scan_raw(conn, str(start_d), str(end_d))

        if raw_df.empty:
            st.info("選定區間內無可用資料。")
            st.session_state.pop(state_key, None)
            return

        scan_cfg = AccumulationScanConfig(
            min_consecutive_days=min_days,
            min_buy_sell_ratio=min_ratio,
            min_volume_share=min_share,
        )
        result_df = run_accumulation_scan(raw_df, scan_cfg)
        st.session_state[state_key] = result_df

    if state_key not in st.session_state:
        st.info("請先設定條件後，點擊「執行全市場低檔潛伏掃描」。")
        return

    result_df = st.session_state[state_key]

    if result_df.empty:
        st.warning("沒有分點同時符合：連續買超、買賣比門檻與成交佔比門檻。")
        return

    display_df = result_df.head(int(top_n)).copy()
    display_df["start_date"] = pd.to_datetime(display_df["start_date"]).dt.date
    display_df["end_date"] = pd.to_datetime(display_df["end_date"]).dt.date
    display_df["latest_signal_end"] = pd.to_datetime(display_df["latest_signal_end"]).dt.date

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("符合條件分點-股票組合", f"{len(result_df)}")
    c2.metric("涉及分點數", f"{result_df['branch_id'].nunique()}")
    c3.metric("涉及股票數", f"{result_df['stock_id'].nunique()}")
    c4.metric("最長連買天數", f"{int(result_df['consecutive_days'].max())}")

    formatters = {
        "連續買超天數": "{:.0f}",
        "區間買超張數": "{:.0f}",
        "平均買賣比": "{:.2f}",
        "平均成交佔比": "{:.2%}",
        "符合區段數": "{:.0f}",
    }

    localized = display_df.rename(columns=COLUMN_LABELS)
    st.dataframe(
        localized.style.format({k: v for k, v in formatters.items() if k in localized.columns}),
        use_container_width=True,
        hide_index=True,
    )
