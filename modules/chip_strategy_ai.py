import tempfile

import pandas as pd
import streamlit as st

from utility.chip_strategy_ai import ChipStrategyAI


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def show_chip_strategy_ai():
    st.header("🧠 ChipStrategyAI（CSV 入口）")
    st.caption("同一入口支援兩個需求：1) 贏家分點自動追蹤 2) AI 策略挖掘")

    uploaded_file = st.file_uploader("上傳分點 CSV", type=["csv"])
    if uploaded_file is None:
        st.info("請先上傳 CSV（需含：日期、分點名稱、買進張數、賣出張數、成交均價）")
        return

    preview_df = _read_uploaded_csv(uploaded_file)
    st.markdown("**資料預覽（前 100 筆）**")
    st.dataframe(preview_df.head(100), use_container_width=True, hide_index=True)

    date_col = pd.to_datetime(preview_df["日期"], errors="coerce") if "日期" in preview_df.columns else pd.Series(dtype="datetime64[ns]")
    min_date = date_col.min().date() if not date_col.empty and pd.notna(date_col.min()) else pd.to_datetime("today").date()
    max_date = date_col.max().date() if not date_col.empty and pd.notna(date_col.max()) else pd.to_datetime("today").date()

    d_range = st.date_input("分析區間", value=[min_date, max_date])
    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
        start_d = pd.to_datetime(d_range[0]).date().isoformat()
        end_d = pd.to_datetime(d_range[1]).date().isoformat()
    else:
        start_d = pd.to_datetime(min_date).date().isoformat()
        end_d = pd.to_datetime(max_date).date().isoformat()

    top_n = st.slider("Top N 贏家分點", min_value=5, max_value=50, value=20, step=1)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    ai = ChipStrategyAI(filepath=tmp_path, start_date=start_d, end_date=end_d)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("🚀 執行需求1：贏家分點自動追蹤", use_container_width=True):
            try:
                track = ai.track_winner_branches(top_n=top_n)
                st.subheader("🏆 Top Winners")
                st.dataframe(track["top_winners"], use_container_width=True, hide_index=True)

                st.subheader("🔔 Daily Alerts")
                st.dataframe(track["daily_alerts"], use_container_width=True, hide_index=True)

                with st.expander("查看完整輸出（winner_rating / concentration / strategy_candidates）"):
                    st.markdown("**winner_rating**")
                    st.dataframe(track["winner_rating"], use_container_width=True, hide_index=True)
                    st.markdown("**concentration**")
                    st.dataframe(track["concentration"], use_container_width=True, hide_index=True)
                    st.markdown("**strategy_candidates**")
                    st.dataframe(track["strategy_candidates"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"追蹤流程執行失敗：{e}")

    with c2:
        if st.button("🧪 執行需求2：AI 挖掘交易策略", use_container_width=True):
            try:
                mine = ai.mine_trading_strategies(start_date=start_d, end_date=end_d)
                st.subheader("📦 訓練資料集")
                st.dataframe(mine["dataset"].head(200), use_container_width=True, hide_index=True)

                st.subheader("🤖 模型結果")
                st.json(mine["model_result"])

                st.subheader("⚙️ 參數掃描")
                st.dataframe(mine["param_scan"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"策略挖掘執行失敗：{e}")
