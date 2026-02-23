import streamlit as st
import pandas as pd

import database
from utility.branch_anomaly_detection import BranchAnomalyConfig, build_anomaly_outputs


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


def show_branch_anomaly():
    st.markdown("### 🚨 分點異常偵測")
    st.caption("用分點進出 + 價格資料，自動產出異常排行、可交易追蹤清單、週報。")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)

    universe = cfg.get("universe", [])
    if not universe:
        st.warning("config.json 未設定 universe。")
        return

    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}
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

    if not st.button("執行異常偵測", type="primary"):
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

    st.success(f"完成：共偵測 {len(anomaly_events)} 筆分點日資料，追蹤名單 {len(watchlist)} 筆。")

    st.subheader("1) 異常事件排行榜")
    show_cols = [
        "date", "stock_id", "branch_id", "anomaly_score", "anomaly_level",
        "z_net_20", "z_gross_20", "vol_share", "flag_accumulation", "flag_distribution"
    ]
    st.dataframe(
        anomaly_events[show_cols].head(int(top_n)).style.format({
            "anomaly_score": "{:.2f}",
            "z_net_20": "{:.2f}",
            "z_gross_20": "{:.2f}",
            "vol_share": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.subheader("2) 可交易追蹤清單")
    st.dataframe(
        watchlist[show_cols].head(int(top_n)).style.format({
            "anomaly_score": "{:.2f}",
            "z_net_20": "{:.2f}",
            "z_gross_20": "{:.2f}",
            "vol_share": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.subheader("3) 每週摘要")
    st.dataframe(
        weekly_report.style.format({"avg_score": "{:.2f}", "avg_vol_share": "{:.2%}"}),
        use_container_width=True,
    )
