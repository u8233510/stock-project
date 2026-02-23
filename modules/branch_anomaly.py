import streamlit as st
import pandas as pd

import database
from utility.branch_anomaly_detection import BranchAnomalyConfig, build_anomaly_outputs


COLUMN_LABELS = {
    "date": "日期",
    "week": "週別",
    "stock_id": "股票代號",
    "branch_id": "分點代號",
    "anomaly_score": "異常分數",
    "anomaly_level": "異常等級",
    "z_net_20": "淨買賣量 Z 分數(20日)",
    "z_gross_20": "總成交量 Z 分數(20日)",
    "vol_share": "分點成交占比",
    "flag_accumulation": "疑似吸籌",
    "flag_distribution": "疑似出貨",
    "events": "事件筆數",
    "avg_score": "平均異常分數",
    "avg_vol_share": "平均成交占比",
}

LEVEL_LABELS = {
    "normal": "一般",
    "medium": "中度",
    "major": "高度",
}


def _localize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "anomaly_level" in out.columns:
        out["anomaly_level"] = out["anomaly_level"].map(LEVEL_LABELS).fillna(out["anomaly_level"])
    return out.rename(columns=COLUMN_LABELS)


def _render_column_guide() -> None:
    with st.expander("欄位說明（如何解讀）", expanded=True):
        st.markdown(
            """
            - **異常分數**：綜合分數（0~100），越高代表越偏離分點平常行為。
            - **異常等級**：`一般 / 中度 / 高度`，可快速判斷是否要追蹤。
            - **淨買賣量 Z 分數(20日)**：分點淨買賣量相對過去 20 日的偏離程度；\
              正值偏買、負值偏賣，絕對值越大代表越異常。
            - **總成交量 Z 分數(20日)**：該分點總成交量相對過去 20 日是否放大。
            - **分點成交占比**：該分點占當日全股票分點成交量比例；越高代表集中度越高。
            - **疑似吸籌 / 疑似出貨**：系統規則旗標，表示是否符合連續偏買或偏賣特徵。
            - **每週摘要**：看每週事件筆數、平均異常分數、平均成交占比，評估異常是否持續。
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
    _render_column_guide()

    st.subheader("1) 異常事件排行榜")
    show_cols = [
        "date", "stock_id", "branch_id", "anomaly_score", "anomaly_level",
        "z_net_20", "z_gross_20", "vol_share", "flag_accumulation", "flag_distribution"
    ]
    st.dataframe(
        _localize_table(anomaly_events[show_cols].head(int(top_n))).style.format({
            "anomaly_score": "{:.2f}",
            "z_net_20": "{:.2f}",
            "z_gross_20": "{:.2f}",
            "vol_share": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.subheader("2) 可交易追蹤清單")
    st.dataframe(
        _localize_table(watchlist[show_cols].head(int(top_n))).style.format({
            "anomaly_score": "{:.2f}",
            "z_net_20": "{:.2f}",
            "z_gross_20": "{:.2f}",
            "vol_share": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.subheader("3) 每週摘要")
    st.dataframe(
        _localize_table(weekly_report).style.format({"avg_score": "{:.2f}", "avg_vol_share": "{:.2%}"}),
        use_container_width=True,
    )
