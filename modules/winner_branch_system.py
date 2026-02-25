import streamlit as st
import pandas as pd

import database
from utility.chip_strategy_ai import ChipStrategyAI, ChipStrategyConfig
from utility.winner_branch_ai_system import WinnerBranchConfig
from utility.winner_branch_ml import WinnerMLConfig


DISPLAY_COLUMN_MAP = {
    "stock_id": "股票代號\nStock ID",
    "date": "日期\nDate",
    "branch_id": "分點代碼\nBranch ID",
    "branch_name": "分點名稱\nBranch Name",
    "hit_rate": "命中率\nHit Rate",
    "avg_alpha": "平均超額報酬\nAverage Alpha",
    "pnl_ratio": "盈虧比\nPnL Ratio",
    "sharpe": "夏普值\nSharpe Ratio",
    "timing_score": "時機分數\nTiming Score",
    "holding_power": "持股續航力\nHolding Power",
    "risk_penalty": "風險懲罰\nRisk Penalty",
    "winner_rating": "贏家評分\nWinner Rating",
    "winner_type": "贏家類型\nWinner Type",
    "net_vol": "淨買賣超\nNet Volume",
    "vol_share": "成交量占比\nVolume Share",
    "alert_level": "警示等級\nAlert Level",
    "alert_level_desc": "警示說明\nAlert Description",
    "reason": "觸發原因\nTrigger Reason",
    "hhi": "HHI 集中度\nHHI Concentration",
    "entropy": "熵值\nEntropy",
    "hhi_delta_10": "HHI 10日變化\n10D HHI Delta",
    "price_compression": "價格壓縮度\nPrice Compression",
    "rule_hhi_rise": "規則：HHI上升\nRule: HHI Rise",
    "rule_compression": "規則：價格壓縮\nRule: Price Compression",
    "strategy_candidate": "策略候選\nStrategy Candidate",
    "candidate_score": "候選分數\nCandidate Score",
    "label_positive": "正樣本標記\nPositive Label",
    "feature": "特徵\nFeature",
    "importance": "重要度\nImportance",
    "model_score": "模型分數\nModel Score",
    "model_signal": "模型訊號\nModel Signal",
    "candidate_rank": "候選排名\nCandidate Rank",
}

ALERT_LEVEL_DESC = {
    "A": "A 級（最強）",
    "B": "B 級（中強）",
    "C": "C 級（觀察）",
}

RATING_FOCUS_COLUMNS = [
    "date",
    "branch_id",
    "branch_name",
    "winner_type",
    "winner_rating",
    "hit_rate",
    "avg_alpha",
    "pnl_ratio",
    "sharpe",
    "timing_score",
    "holding_power",
    "risk_penalty",
]


def _to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in DISPLAY_COLUMN_MAP.items() if k in df.columns})


def _render_winner_rating_table(rating_df: pd.DataFrame, branch_lookup: pd.DataFrame):
    if rating_df.empty:
        st.info("目前沒有可顯示的分點評級資料。")
        return

    merged = rating_df.merge(branch_lookup, on="branch_id", how="left")
    merged["branch_name"] = merged["branch_name"].fillna("-")

    sort_cols = [
        "winner_rating",
        "hit_rate",
        "avg_alpha",
        "pnl_ratio",
        "sharpe",
        "timing_score",
        "holding_power",
    ]
    available_sort_cols = [c for c in sort_cols if c in merged.columns]
    if available_sort_cols:
        merged = merged.sort_values(available_sort_cols, ascending=[False] * len(available_sort_cols))

    st.caption("同一張表整合全部重點欄位，可依 Rating 篩選要顯示的分點。")
    st.caption("Top N：只保留評分最高前 N 筆；Rating 門檻：保留所有評分 >= 指定分數的分點。")

    filter_mode = st.radio(
        "顯示分點篩選方式",
        options=["全部分點", "Top N（依 Rating）", "Rating 門檻"],
        horizontal=True,
        key="winner_rating_filter_mode",
    )

    if filter_mode == "Top N（依 Rating）":
        max_rows = max(1, len(merged))
        top_n = st.slider(
            "顯示前 N 名分點",
            min_value=1,
            max_value=max_rows,
            value=min(20, max_rows),
            step=1,
            key="winner_rating_top_n",
        )
        display_df = merged.head(top_n)
    elif filter_mode == "Rating 門檻":
        threshold = st.slider(
            "最低 Winner Rating",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.5,
            key="winner_rating_threshold",
        )
        display_df = merged[merged["winner_rating"] >= threshold]
    else:
        display_df = merged

    selected_branches = st.multiselect(
        "進一步指定要顯示的分點（可複選）",
        options=[str(v) for v in display_df["branch_id"].astype(str).unique()],
        default=[],
        help="不選則顯示篩選後的全部分點。",
        key="winner_rating_branch_multiselect",
    )
    if selected_branches:
        display_df = display_df[display_df["branch_id"].astype(str).isin(selected_branches)]

    summary_cols = [c for c in RATING_FOCUS_COLUMNS if c in display_df.columns]
    final_df = display_df[summary_cols].copy() if summary_cols else display_df.copy()
    st.dataframe(_to_display_df(final_df), use_container_width=True, hide_index=True)


def _load_branch_raw(conn, sid: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
    SELECT
        b.stock_id,
        b.date,
        b.securities_trader_id AS branch_id,
        b.securities_trader AS branch_name,
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
        hhi_rise_window = st.slider("HHI 上升視窗 (HHI Rise Window)", min_value=5, max_value=30, value=10, step=1)
        compression_window = st.slider("價格壓縮視窗 (Price Compression Window)", min_value=5, max_value=30, value=10, step=1)
        compression_threshold = st.slider("價格壓縮閾值 (Price Compression Threshold)", min_value=0.005, max_value=0.08, value=0.02, step=0.005)

    raw_df = _load_branch_raw(conn, sid, start_d, end_d)
    raw_df = raw_df.dropna(subset=["close"])
    if raw_df.empty:
        st.warning("查無分點資料或找不到對應收盤價（stock_ohlcv_daily），請調整日期區間或先同步資料。 (No branch data / close price found.)")
        conn.close()
        return

    st.caption("此頁面已直接整合 ChipStrategyAI，資料來源為資料庫，不需上傳 CSV。 (Database source, no CSV upload needed.)")

    wb_cfg = WinnerBranchConfig(top_quantile=top_quantile, hhi_rise_window=hhi_rise_window, compression_window=compression_window, compression_threshold=compression_threshold)

    st.divider()
    st.subheader("🎯 需求 1：針對贏家分點自動化追蹤 (Requirement 1: Winner Branch Tracking)")

    if "winner_track_cache" not in st.session_state:
        st.session_state["winner_track_cache"] = None

    if st.button("🚀 執行需求1：贏家分點追蹤 (Run Requirement 1)", use_container_width=True):
        branch_lookup = (
            raw_df[["branch_id", "branch_name"]]
            .dropna(subset=["branch_id"])
            .astype({"branch_id": str})
            .drop_duplicates(subset=["branch_id"], keep="last")
        )
        branch_lookup["branch_id"] = branch_lookup["branch_id"].astype(str)

        chip = ChipStrategyAI.from_dataframe(
            raw_df,
            start_date=start_d,
            end_date=end_d,
            config=ChipStrategyConfig(winner_cfg=wb_cfg),
        )
        track = chip.track_winner_branches()
        track["winner_rating"]["branch_id"] = track["winner_rating"]["branch_id"].astype(str)
        st.session_state["winner_track_cache"] = {
            "sid": sid,
            "start_d": start_d,
            "end_d": end_d,
            "track": track,
            "branch_lookup": branch_lookup,
        }

    cache = st.session_state.get("winner_track_cache")
    if cache and cache.get("sid") == sid and cache.get("start_d") == start_d and cache.get("end_d") == end_d:
        track = cache["track"]
        branch_lookup = cache["branch_lookup"]

        st.subheader("🏆 Winner Rating（分點評級）")
        _render_winner_rating_table(track["winner_rating"], branch_lookup)

        st.subheader("🔔 Daily Alerts（A/B/C）")
        alerts_df = track["daily_alerts"].copy()
        alerts_df["branch_id"] = alerts_df["branch_id"].astype(str)
        daily_alerts_display = alerts_df.merge(branch_lookup, on="branch_id", how="left")
        daily_alerts_display["branch_name"] = daily_alerts_display["branch_name"].fillna("-")
        daily_alerts_display["alert_level_desc"] = daily_alerts_display["alert_level"].map(ALERT_LEVEL_DESC).fillna("-")
        st.dataframe(_to_display_df(daily_alerts_display), use_container_width=True, hide_index=True)

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
            daily_alerts_display.to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_winner_alerts.csv",
            "text/csv",
        )
    else:
        st.info("請先點擊「執行需求1：贏家分點追蹤」後，再使用 Top N / Rating 門檻篩選。")

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
