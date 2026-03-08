from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

from branch_net_detector_cli import build_summary


def show_branch_net_detector_entry() -> None:
    st.header("🧮 分點買賣超偵測")
    st.caption("計算指定區間內所有股票分點買賣超統計，先顯示於畫面，按下載才存到本機。")

    today = date.today()
    start_date = st.date_input("起始日期", value=today)
    end_date = st.date_input("結束日期", value=today)
    db_path = st.text_input("SQLite 路徑", value="data/stock.db")

    if st.button("執行分點買賣超偵測", type="primary"):
        if start_date > end_date:
            st.error("起始日期不可晚於結束日期")
            return

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            st.error(f"無法開啟資料庫: {exc}")
            return

        with st.spinner("執行中，請稍候..."):
            try:
                rows = build_summary(conn, start_date.isoformat(), end_date.isoformat())
            finally:
                conn.close()

        if not rows:
            st.warning("指定區間查無 branch_trader_daily_detail 資料。")
            return

        df = pd.DataFrame(rows)
        st.success(f"執行完成，共 {len(df)} 檔股票")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "下載結果 CSV",
            data=csv_data,
            file_name="branch_interval_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
