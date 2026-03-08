import subprocess
import sys
from datetime import date

import streamlit as st


def show_branch_net_detector_entry() -> None:
    st.header("🧮 分點買賣超偵測")
    st.caption("透過獨立程式 branch_net_detector_cli.py 計算指定區間內所有股票分點買賣超統計。")

    today = date.today()
    start_date = st.date_input("起始日期", value=today)
    end_date = st.date_input("結束日期", value=today)
    db_path = st.text_input("SQLite 路徑", value="data/stock.db")
    output_path = st.text_input("輸出 CSV 路徑", value="output/branch_interval_summary.csv")

    if st.button("執行分點買賣超偵測", type="primary"):
        if start_date > end_date:
            st.error("起始日期不可晚於結束日期")
            return

        cmd = [
            sys.executable,
            "branch_net_detector_cli.py",
            "--start",
            start_date.isoformat(),
            "--end",
            end_date.isoformat(),
            "--db-path",
            db_path,
            "--output",
            output_path,
        ]

        with st.spinner("執行中，請稍候..."):
            result = subprocess.run(cmd, capture_output=True, text=True)

        st.code(" ".join(cmd), language="bash")
        if result.returncode == 0:
            st.success("執行完成")
            if result.stdout.strip():
                st.text(result.stdout.strip())
        else:
            st.error("執行失敗")
            if result.stdout.strip():
                st.text(result.stdout.strip())
            if result.stderr.strip():
                st.text(result.stderr.strip())
