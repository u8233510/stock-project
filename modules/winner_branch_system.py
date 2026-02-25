import streamlit as st
import pandas as pd

import database
from utility.chip_strategy_ai import ChipStrategyAI, ChipStrategyConfig
from utility.winner_branch_ai_system import WinnerBranchConfig
from utility.winner_branch_ml import WinnerMLConfig


DISPLAY_COLUMN_MAP = {
    "stock_id": "股票代號 (Stock ID)",
    "date": "日期 (Date)",
    "branch_id": "分點代碼 (Branch ID)",
    "hit_rate": "命中率 (Hit Rate)",
    "avg_alpha": "平均超額報酬 (Average Alpha)",
    "pnl_ratio": "盈虧比 (PnL Ratio)",
    "sharpe": "夏普值 (Sharpe Ratio)",
    "timing_score": "時機分數 (Timing Score)",
    "holding_power": "持股續航力 (Holding Power)",
    "risk_penalty": "風險懲罰 (Risk Penalty)",
    "winner_rating": "贏家評分 (Winner Rating)",
    "winner_type": "贏家類型 (Winner Type)",
    "net_vol": "淨買賣超 (Net Volume)",
    "vol_share": "成交量占比 (Volume Share)",
    "alert_level": "警示等級 (Alert Level)",
    "reason": "觸發原因 (Trigger Reason)",
    "hhi": "HHI 集中度 (HHI Concentration)",
    "entropy": "熵值 (Entropy)",
    "hhi_delta_10": "HHI 10日變化 (10D HHI Delta)",
    "price_compression": "價格壓縮度 (Price Compression)",
    "rule_hhi_rise": "規則：HHI上升 (Rule: HHI Rise)",
    "rule_compression": "規則：價格壓縮 (Rule: Price Compression)",
    "strategy_candidate": "策略候選 (Strategy Candidate)",
    "candidate_score": "候選分數 (Candidate Score)",
    "label_positive": "正樣本標記 (Positive Label)",
    "feature": "特徵 (Feature)",
    "importance": "重要度 (Importance)",
    "model_score": "模型分數 (Model Score)",
    "model_signal": "模型訊號 (Model Signal)",
    "candidate_rank": "候選排名 (Candidate Rank)",
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
    st.header("🧠 AI 贏家分點追蹤系統 (AI Winner Branch Tracking)")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}

    if not stock_options:
        st.warning("設定檔 universe 為空，請先補上股票清單。 (Universe is empty in config.)")
        conn.close()
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        target_label = st.selectbox("選擇標的 (Select Stock)", list(stock_options.keys()))
        sid = stock_options[target_label]

    with col2:
        default_start = pd.to_datetime("today") - pd.Timedelta(days=180)
        default_end = pd.to_datetime("today")
        d_range = st.date_input("分析區間 (Analysis Range)", value=[default_start, default_end])

    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
        start_d = pd.to_datetime(d_range[0]).date().isoformat()
        end_d = pd.to_datetime(d_range[1]).date().isoformat()
    else:
        start_d = (pd.to_datetime("today") - pd.Timedelta(days=180)).date().isoformat()
        end_d = pd.to_datetime("today").date().isoformat()

    with st.expander("進階參數 (Advanced Settings)", expanded=False):
        top_quantile = st.slider("Top Winner 分位數 (Top Winner Quantile)", min_value=0.7, max_value=0.98, value=0.85, step=0.01)
        compression_threshold = st.slider("價格壓縮閾值 (Price Compression Threshold)", min_value=0.005, max_value=0.08, value=0.02, step=0.005)

    raw_df = _load_branch_raw(conn, sid, start_d, end_d)
    raw_df = raw_df.dropna(subset=["close"])
    if raw_df.empty:
        st.warning("查無分點資料或找不到對應收盤價（stock_ohlcv_daily），請調整日期區間或先同步資料。 (No branch data / close price found.)")
        conn.close()
        return

    st.caption("此頁面已直接整合 ChipStrategyAI，資料來源為資料庫，不需上傳 CSV。 (Database source, no CSV upload needed.)")

    wb_cfg = WinnerBranchConfig(top_quantile=top_quantile, compression_threshold=compression_threshold)

    st.divider()
    st.subheader("🎯 需求 1：針對贏家分點自動化追蹤 (Requirement 1: Winner Branch Tracking)")
    top_n = st.slider("Top N 贏家分點 (Top N Winner Branches)", min_value=5, max_value=50, value=20, step=1)

    if st.button("🚀 執行需求1：贏家分點追蹤 (Run Requirement 1)", use_container_width=True):
        chip = ChipStrategyAI.from_dataframe(
            raw_df,
            start_date=start_d,
            end_date=end_d,
            config=ChipStrategyConfig(winner_cfg=wb_cfg),
        )
        track = chip.track_winner_branches(top_n=top_n)

        st.subheader("🏆 Winner Rating（分點評級）")
        st.dataframe(_to_display_df(track["winner_rating"]), use_container_width=True, hide_index=True)

        st.subheader("🔔 Daily Alerts（A/B/C）")
        st.dataframe(_to_display_df(track["daily_alerts"]), use_container_width=True, hide_index=True)

        st.subheader("🧪 Strategy Candidates（集中度策略候選）")
        cand = track["strategy_candidates"]
        if "strategy_candidate" in cand.columns:
            cand = cand[cand["strategy_candidate"] == True]
        st.dataframe(_to_display_df(cand), use_container_width=True, hide_index=True)

        with st.expander("查看 Concentration Features 原始輸出 (View Raw Output)"):
            st.dataframe(_to_display_df(track["concentration"]), use_container_width=True, hide_index=True)

        st.download_button(
            "📥 下載 Winner Rating CSV (Download)",
            track["winner_rating"].to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_winner_rating.csv",
            "text/csv",
        )
        st.download_button(
            "📥 下載 Daily Alerts CSV (Download)",
            track["daily_alerts"].to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_winner_alerts.csv",
            "text/csv",
        )

    st.divider()
    st.subheader("🤖 需求 2：AI 從海量數據挖掘新交易策略 (Requirement 2: Strategy Mining)")
    st.caption("同樣使用資料庫分點資料，不需上傳 CSV。 (Also from database, no CSV upload.)")

    colm1, colm2 = st.columns(2)
    with colm1:
        lookahead_days = st.slider("正樣本觀察天數 (Positive Sample Window)", min_value=5, max_value=40, value=20, step=1)
    with colm2:
        rally_threshold = st.slider("正樣本漲幅門檻 (Positive Rally Threshold)", min_value=0.03, max_value=0.20, value=0.08, step=0.01)

    if st.button("🧪 執行需求2：策略挖掘 (Run Requirement 2)", use_container_width=True):
        ml_cfg = WinnerMLConfig(lookahead_days=lookahead_days, rally_threshold=rally_threshold)
        chip = ChipStrategyAI.from_dataframe(
            raw_df,
            start_date=start_d,
            end_date=end_d,
            config=ChipStrategyConfig(winner_cfg=wb_cfg, ml_cfg=ml_cfg),
        )

        mine = chip.mine_trading_strategies(ml_cfg=ml_cfg)
        train_ds = mine["dataset"]
        st.markdown("**訓練資料集（含 label_positive） (Training Dataset)**")
        st.dataframe(_to_display_df(train_ds.head(200)), use_container_width=True, hide_index=True)

        positive_rate = float(train_ds["label_positive"].mean()) if not train_ds.empty else 0.0
        st.info(f"樣本數 (Samples): {len(train_ds)}，Positive 比例 (Rate): {positive_rate:.2%}")

        st.download_button(
            "📥 下載訓練資料集 CSV (Download)",
            train_ds.to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_winner_phase2_training_dataset.csv",
            "text/csv",
        )

        train_result = mine["model_result"]
        if train_result.get("status") == "ok":
            st.success("XGBoost 訓練完成 (Training Completed)")
            st.write(
                {
                    "split_date": train_result["split_date"],
                    "accuracy": train_result["accuracy"],
                    "precision": train_result["precision"],
                    "recall": train_result["recall"],
                }
            )
            fi = pd.DataFrame(
                [{"feature": k, "importance": v} for k, v in train_result["feature_importance"].items()]
            ).sort_values("importance", ascending=False)
            st.dataframe(_to_display_df(fi), use_container_width=True, hide_index=True)
        else:
            st.warning(f"模型訓練略過 (Training Skipped): {train_result.get('message', train_result.get('status'))}")

        st.markdown("**持有天數 / 停損參數掃描（proxy backtest） (Holding Days / Stop-Loss Scan)**")
        st.dataframe(_to_display_df(mine["param_scan"]), use_container_width=True, hide_index=True)

        st.markdown("**📌 今日候選股清單（模型分數轉訊號） (Today's Candidates)**")
        today_candidates = mine.get("today_candidates", pd.DataFrame())
        if today_candidates.empty:
            st.info("目前沒有達到模型門檻的候選股（可嘗試調整正樣本參數後重跑）。 (No candidates passed threshold.)")
        else:
            st.dataframe(_to_display_df(today_candidates), use_container_width=True, hide_index=True)
            st.download_button(
                "📥 下載今日候選股 CSV (Download)",
                today_candidates.to_csv(index=False).encode("utf-8-sig"),
                f"{sid}_today_model_candidates.csv",
                "text/csv",
            )

    conn.close()
