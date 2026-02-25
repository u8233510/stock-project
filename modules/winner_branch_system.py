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

COLUMN_NAME_MAP = {
    "stock_id": "股票代號",
    "date": "日期",
    "branch_id": "分點代碼",
    "price": "分點成交均價",
    "buy": "買進張數",
    "sell": "賣出張數",
    "close": "收盤價",
    "hit_rate": "命中率",
    "avg_alpha": "平均超額報酬",
    "pnl_ratio": "盈虧比",
    "sharpe": "夏普值",
    "timing_score": "時機分數",
    "holding_power": "持有力",
    "risk_penalty": "風險扣分",
    "winner_rating": "贏家評級",
    "winner_type": "贏家類型",
    "net_vol": "淨買賣張數",
    "vol_share": "成交占比",
    "alert_level": "警示等級",
    "reason": "觸發原因",
    "hhi": "HHI集中度",
    "entropy": "熵值",
    "hhi_delta_10": "HHI_10日變化",
    "price_compression": "價格壓縮度",
    "rule_hhi_rise": "規則_HHI上升",
    "rule_compression": "規則_價格壓縮",
    "strategy_candidate": "是否策略候選",
    "candidate_score": "候選分數",
    "avg_buy_cost": "平均買入成本",
    "cost_gap": "成本乖離",
    "net_buy_strength": "淨買強度",
    "buy_continuity": "買盤連續性",
    "retail_exit_ratio": "散戶退出比例",
    "future_max_ret": "未來區間最大報酬",
    "label_positive": "正樣本標籤",
    "future_ret_5": "未來5日報酬",
    "future_ret_10": "未來10日報酬",
    "future_ret_20": "未來20日報酬",
    "feature": "特徵",
    "importance": "重要度",
    "hold_days": "持有天數",
    "stop_loss": "停損比例",
    "sample_size": "樣本數",
    "win_rate": "勝率",
    "avg_return": "平均報酬",
    "expectancy": "期望值",
}


def _to_zh_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_NAME_MAP)


def _format_display_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "成交占比" in out.columns:
        out["成交占比"] = (out["成交占比"] * 100).round(2).astype(str) + "%"
    for c in ["命中率", "買盤連續性", "散戶退出比例", "未來區間最大報酬", "未來5日報酬", "未來10日報酬", "未來20日報酬", "勝率", "平均報酬", "期望值", "成本乖離"]:
        if c in out.columns:
            out[c] = (out[c] * 100).round(2).astype(str) + "%"
    for c in ["贏家評級", "候選分數", "夏普值", "盈虧比", "時機分數", "持有力", "風險扣分", "重要度"]:
        if c in out.columns:
            out[c] = out[c].round(4)
    return out


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

        winner_rating_zh = _to_zh_columns(winner_rating)
        daily_alerts_zh = _to_zh_columns(daily_alerts)
        concentration_zh = _to_zh_columns(concentration)
        strategy_candidates_zh = _to_zh_columns(strategy_candidates)

        st.subheader("🏆 贏家評級（分點）")
        st.dataframe(_format_display_values(winner_rating_zh), use_container_width=True, hide_index=True)

        st.subheader("🔔 每日警示（A/B/C）")
        st.dataframe(_format_display_values(daily_alerts_zh), use_container_width=True, hide_index=True)

        st.subheader("🧪 策略候選（集中度）")
        if "是否策略候選" in strategy_candidates_zh.columns:
            cand_zh = strategy_candidates_zh[strategy_candidates_zh["是否策略候選"] == True]
        else:
            cand_zh = strategy_candidates_zh
        st.dataframe(_format_display_values(cand_zh), use_container_width=True, hide_index=True)

        with st.expander("查看集中度特徵原始輸出"):
            st.dataframe(_format_display_values(concentration_zh), use_container_width=True, hide_index=True)

        st.download_button(
            "📥 下載贏家評級 CSV",
            winner_rating_zh.to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_贏家評級.csv",
            "text/csv",
        )
        st.download_button(
            "📥 下載每日警示 CSV",
            daily_alerts_zh.to_csv(index=False).encode("utf-8-sig"),
            f"{sid}_每日警示.csv",
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
            train_ds_zh = _to_zh_columns(train_ds)
            st.markdown("**訓練資料集（含正樣本標籤）**")
            st.dataframe(_format_display_values(train_ds_zh.head(200)), use_container_width=True, hide_index=True)

            positive_rate = float(train_ds["label_positive"].mean()) if not train_ds.empty else 0.0
            st.info(f"樣本數: {len(train_ds)}，正樣本比例: {positive_rate:.2%}")

            st.download_button(
                "📥 下載訓練資料集 CSV",
                train_ds_zh.to_csv(index=False).encode("utf-8-sig"),
                f"{sid}_模型訓練資料集.csv",
                "text/csv",
            )

            train_result = train_xgboost_classifier(train_ds)
            if train_result.get("status") == "ok":
                st.success("XGBoost 訓練完成")
                st.write({
                    "切分日期": train_result["split_date"],
                    "準確率": train_result["accuracy"],
                    "精確率": train_result["precision"],
                    "召回率": train_result["recall"],
                })
                fi = pd.DataFrame(
                    [{"feature": k, "importance": v} for k, v in train_result["feature_importance"].items()]
                ).sort_values("importance", ascending=False)
                fi_zh = _to_zh_columns(fi)
                st.dataframe(_format_display_values(fi_zh), use_container_width=True, hide_index=True)
            else:
                st.warning(f"模型訓練略過：{train_result.get('message', train_result.get('status'))}")

            opt = optimize_trade_params(train_ds, signal_col="label_positive")
            st.markdown("**持有天數 / 停損參數掃描（回測代理）**")
            opt_zh = _to_zh_columns(opt)
            st.dataframe(_format_display_values(opt_zh), use_container_width=True, hide_index=True)

    conn.close()
