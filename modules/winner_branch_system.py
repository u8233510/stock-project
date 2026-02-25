import streamlit as st
import pandas as pd

import database
from utility.winner_branch_ai_system import build_winner_branch_outputs, WinnerBranchConfig
from utility.winner_branch_ml import (
    WinnerMLConfig,
    build_phase2_training_dataset,
    train_xgboost_classifier,
    optimize_trade_params,
)


DISPLAY_COLUMN_MAP = {
    "stock_id": "股票代號",
    "date": "日期",
    "branch_id": "分點代碼",
    "hit_rate": "命中率",
    "avg_alpha": "平均超額報酬",
    "pnl_ratio": "盈虧比",
    "sharpe": "夏普值",
    "timing_score": "時機分數",
    "holding_power": "持股續航力",
    "risk_penalty": "風險懲罰",
    "winner_rating": "贏家評分",
    "winner_type": "贏家類型",
    "net_vol": "淨買賣超",
    "vol_share": "成交量占比",
    "alert_level": "警示等級",
    "reason": "觸發原因",
    "hhi": "HHI 集中度",
    "entropy": "熵值",
    "hhi_delta_10": "HHI 10日變化",
    "price_compression": "價格壓縮度",
    "rule_hhi_rise": "規則：HHI上升",
    "rule_compression": "規則：價格壓縮",
    "strategy_candidate": "策略候選",
    "candidate_score": "候選分數",
    "label_positive": "正樣本標記",
    "feature": "特徵",
    "importance": "重要度",
}


def _to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in DISPLAY_COLUMN_MAP.items() if k in df.columns})


def _load_branch_raw(conn, sid: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
    SELECT
        b.stock_id,
        b.date,
        b.securities_trader_id AS branch_id,
        b.price,
        b.buy,
        b.sell,
        o.close
    FROM branch_price_daily b
    LEFT JOIN stock_ohlcv_daily o
      ON o.stock_id = b.stock_id
     AND o.date = b.date
    WHERE b.stock_id = ?
      AND b.date BETWEEN ? AND ?
    ORDER BY b.date ASC
    """
    return pd.read_sql(query, conn, params=(sid, start_date, end_date))


def show_winner_branch_system():
    st.header("🧠 AI 贏家分點追蹤系統（可執行版）")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}

    if not stock_options:
        st.warning("設定檔 universe 為空，請先補上股票清單。")
        conn.close()
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        target_label = st.selectbox("選擇標的", list(stock_options.keys()))
        sid = stock_options[target_label]

    with col2:
        default_start = pd.to_datetime("today") - pd.Timedelta(days=180)
        default_end = pd.to_datetime("today")
        d_range = st.date_input("分析區間", value=[default_start, default_end])

    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
        start_d = pd.to_datetime(d_range[0]).date().isoformat()
        end_d = pd.to_datetime(d_range[1]).date().isoformat()
    else:
        start_d = (pd.to_datetime("today") - pd.Timedelta(days=180)).date().isoformat()
        end_d = pd.to_datetime("today").date().isoformat()

    with st.expander("進階參數", expanded=False):
        top_quantile = st.slider("Top Winner 分位數", min_value=0.7, max_value=0.98, value=0.85, step=0.01)
        compression_threshold = st.slider("價格壓縮閾值", min_value=0.005, max_value=0.08, value=0.02, step=0.005)

    if st.button("🚀 執行贏家分點計算", use_container_width=True):
        with st.spinner("正在計算 winner rating / alerts / strategy candidates..."):
            raw_df = _load_branch_raw(conn, sid, start_d, end_d)
            if raw_df.empty:
                st.warning("查無分點資料，請調整日期區間或先同步資料。")
                conn.close()
                return

            raw_df = raw_df.dropna(subset=["close"])
            if raw_df.empty:
                st.warning("找不到對應收盤價（stock_ohlcv_daily），無法計算 forward return。")
                conn.close()
                return

            wb_cfg = WinnerBranchConfig(
                top_quantile=top_quantile,
                compression_threshold=compression_threshold,
            )
            winner_rating, daily_alerts, concentration, strategy_candidates = build_winner_branch_outputs(raw_df, cfg=wb_cfg)

        st.subheader("🏆 Winner Rating（分點評級）")
        st.dataframe(_to_display_df(winner_rating), use_container_width=True, hide_index=True)

        st.subheader("🔔 Daily Alerts（A/B/C）")
        st.dataframe(_to_display_df(daily_alerts), use_container_width=True, hide_index=True)

        st.subheader("🧪 Strategy Candidates（集中度策略候選）")
        if "strategy_candidate" in strategy_candidates.columns:
            cand = strategy_candidates[strategy_candidates["strategy_candidate"] == True]
        else:
            cand = strategy_candidates
        st.dataframe(_to_display_df(cand), use_container_width=True, hide_index=True)

        with st.expander("查看 Concentration Features 原始輸出"):
            st.dataframe(_to_display_df(concentration), use_container_width=True, hide_index=True)

        st.download_button(
            "📥 下載 Winner Rating CSV",
            winner_rating.to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_winner_rating.csv",
            "text/csv",
        )
        st.download_button(
            "📥 下載 Daily Alerts CSV",
            daily_alerts.to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_winner_alerts.csv",
            "text/csv",
        )


    st.divider()
    st.subheader("🤖 第二階段：模型導向（可選）")
    st.caption("第一階段不必訓練模型；第二階段可用 XGBoost 做正樣本分類與參數優化。")

    colm1, colm2 = st.columns(2)
    with colm1:
        lookahead_days = st.slider("正樣本觀察天數", min_value=5, max_value=40, value=20, step=1)
    with colm2:
        rally_threshold = st.slider("正樣本漲幅門檻", min_value=0.03, max_value=0.20, value=0.08, step=0.01)

    if st.button("🧪 產生模型訓練資料集", use_container_width=True):
        raw_df2 = _load_branch_raw(conn, sid, start_d, end_d)
        raw_df2 = raw_df2.dropna(subset=["close"])
        if raw_df2.empty:
            st.warning("資料不足，無法建立模型資料集。")
        else:
            ml_cfg = WinnerMLConfig(lookahead_days=lookahead_days, rally_threshold=rally_threshold)
            train_ds = build_phase2_training_dataset(raw_df2, cfg=ml_cfg)
            st.markdown("**訓練資料集（含 label_positive）**")
            st.dataframe(_to_display_df(train_ds.head(200)), use_container_width=True, hide_index=True)

            positive_rate = float(train_ds["label_positive"].mean()) if not train_ds.empty else 0.0
            st.info(f"樣本數: {len(train_ds)}，Positive 比例: {positive_rate:.2%}")

            st.download_button(
                "📥 下載訓練資料集 CSV",
                train_ds.to_csv(index=False).encode("utf-8-sig"),
                f"{sid}_winner_phase2_training_dataset.csv",
                "text/csv",
            )

            train_result = train_xgboost_classifier(train_ds)
            if train_result.get("status") == "ok":
                st.success("XGBoost 訓練完成")
                st.write({
                    "split_date": train_result["split_date"],
                    "accuracy": train_result["accuracy"],
                    "precision": train_result["precision"],
                    "recall": train_result["recall"],
                })
                fi = pd.DataFrame(
                    [{"feature": k, "importance": v} for k, v in train_result["feature_importance"].items()]
                ).sort_values("importance", ascending=False)
                st.dataframe(_to_display_df(fi), use_container_width=True, hide_index=True)
            else:
                st.warning(f"模型訓練略過：{train_result.get('message', train_result.get('status'))}")

            opt = optimize_trade_params(train_ds, signal_col="label_positive")
            st.markdown("**持有天數 / 停損參數掃描（proxy backtest）**")
            st.dataframe(_to_display_df(opt), use_container_width=True, hide_index=True)

    conn.close()
