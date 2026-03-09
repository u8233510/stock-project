import csv
import sqlite3
from io import StringIO
from pathlib import Path
from datetime import date

import streamlit as st

from branch_net_detector_cli import FIELDNAMES, build_summary, format_rows_for_output


def _rows_to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(format_rows_for_output(rows))
    return output.getvalue().encode("utf-8-sig")


def show_branch_net_detector_entry() -> None:
    st.header("🧮 分點買賣超偵測")
    st.caption("透過獨立程式 branch_net_detector_cli.py 計算指定區間內所有股票分點買賣超統計。")

    today = date.today()
    start_date = st.date_input("起始日期", value=today)
    end_date = st.date_input("結束日期", value=today)
    db_path = st.text_input("SQLite 路徑", value="data/stock.db")
    output_filename = st.text_input("下載檔名", value="branch_interval_summary.csv")

    if st.button("執行分點買賣超偵測", type="primary"):
        if start_date > end_date:
            st.error("起始日期不可晚於結束日期")
            return

        with st.spinner("執行中，請稍候..."):
            db_file = Path(db_path)
            if not db_file.exists():
                st.error(f"找不到資料庫: {db_file}")
                return

            conn = sqlite3.connect(str(db_file))
            conn.row_factory = sqlite3.Row
            try:
                rows = build_summary(conn, start_date.isoformat(), end_date.isoformat())
            finally:
                conn.close()

        if not rows:
            st.warning("指定區間查無 branch_trader_daily_detail 資料。")
            return

        st.success(f"查詢完成，共 {len(rows)} 檔股票。")
        st.dataframe(format_rows_for_output(rows), use_container_width=True)

        csv_bytes = _rows_to_csv_bytes(rows)
        st.download_button(
            "下載 CSV",
            data=csv_bytes,
            file_name=Path(output_filename).name or "branch_interval_summary.csv",
            mime="text/csv",
        )
