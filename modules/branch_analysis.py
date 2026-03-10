# --- modules/branch_analysis.py 完整代碼 ---
import streamlit as st
import pandas as pd
import database
import json
import requests
import importlib.util
import plotly.graph_objects as go
from utility.weighted_cost_utils import compute_interval_metrics
from utility.branch_weighted_cost_helpers import format_snapshot_caption
from modules.llm_model_selector import get_llm_model


def _call_nim(cfg, messages, temperature=0.0, max_tokens=2000):
    llm_cfg = cfg.get("llm", {})
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_cfg.get('api_key')}", "Content-Type": "application/json"}
    payload = {"model": get_llm_model(cfg, "branch"), "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def analyze_branch_pro(cfg, sid, df_summary, total_vol, current_price, industry_name, df_peers, main_force_cost, chip_concentration):
    if df_summary.empty: return "查無數據"
    est_pressure = round(main_force_cost * 1.15, 2)
    prompt = f"""
    你是專業台股籌碼與技術策略專家。請針對 {sid} 提供具備進出場區間的策略報告。
    【關鍵決策數據】
    - 目前股價：{current_price}
    - 核心主力成本 (強支撐)：{main_force_cost}
    - 籌碼集中度：{chip_concentration}%
    - 估計量能壓力位：{est_pressure}
    ### 【專業策略分析報告】
    ## 📊 綜合評分：[X/10]
    ---
    ### 🔍 1. 籌碼位階診斷
    ### 📈 2. 支撐與壓力定位
    ### 🎯 3. AI 進出場策略建議
    ### 💡 4. 操作總結
    ---
    (數據背景：{df_summary.to_json(orient='records', force_ascii=False)})
    """
    return _call_nim(cfg, [
        {"role": "system", "content": "你是一個果斷的交易導師，必須給出明確的進場與出場區間。"}, 
        {"role": "user", "content": prompt}
    ])

def color_volume(val):
    color = 'red' if val > 0 else 'green'
    return f'color: {color}; font-weight: bold'



def _compute_interval_metrics(df, top_n=15):
    return compute_interval_metrics(df, top_n=top_n)


def _compute_window_snapshot_from_branch(interval_df, end_date, top_n=15):
    if interval_df.empty:
        return None
    avg_cost, total_net_volume, concentration = _compute_interval_metrics(interval_df, top_n=top_n)
    return (float(avg_cost or 0), int(total_net_volume or 0), float(concentration or 0), str(end_date))


def _build_lightgbm_feature_frame(conn, sid, max_trade_days=320, top_n=15):
    trade_dates_desc = conn.execute(
        """
        SELECT DISTINCT date
        FROM branch_price_daily
        WHERE stock_id = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (sid, int(max_trade_days)),
    ).fetchall()

    trade_dates = sorted([str(r[0])[:10] for r in trade_dates_desc if r and r[0]])
    if not trade_dates:
        return pd.DataFrame()

    start_date = trade_dates[0]
    end_date = trade_dates[-1]
    branch_df = pd.read_sql(
        """
        SELECT date, securities_trader_id, price, buy, sell
        FROM branch_price_daily
        WHERE stock_id = ?
          AND date >= ?
          AND date <= ?
        """,
        conn,
        params=(sid, start_date, end_date),
    )
    if branch_df.empty:
        return pd.DataFrame()

    branch_df["date"] = pd.to_datetime(branch_df["date"])
    trade_ts = pd.to_datetime(trade_dates)
    records = []

    for idx, end_ts in enumerate(trade_ts):
        row = {"end_date": end_ts}
        ok = True
        for window in (5, 20, 60):
            if idx + 1 < window:
                ok = False
                break
            start_ts = trade_ts[idx - window + 1]
            mask = (branch_df["date"] >= start_ts) & (branch_df["date"] <= end_ts)
            interval_df = branch_df.loc[mask, ["securities_trader_id", "price", "buy", "sell"]]
            if interval_df.empty:
                ok = False
                break
            avg_cost, total_net_volume, concentration = _compute_interval_metrics(interval_df, top_n=top_n)
            row[f"avg_cost_{window}"] = float(avg_cost or 0)
            row[f"net_vol_{window}"] = int(total_net_volume or 0)
            row[f"concentration_{window}"] = float(concentration or 0)
        if ok:
            records.append(row)

    return pd.DataFrame(records)


def _resolve_lightgbm_max_trade_days(conn, sid, start_d, end_d, default_max_trade_days):
    default_days = max(int(default_max_trade_days or 320), 60)
    selected_days = conn.execute(
        """
        SELECT COUNT(DISTINCT date)
        FROM branch_price_daily
        WHERE stock_id = ? AND date >= ? AND date <= ?
        """,
        (sid, start_d.isoformat(), end_d.isoformat()),
    ).fetchone()
    selected_trade_days = int((selected_days or [0])[0] or 0)
    return max(default_days, selected_trade_days)


def _run_lightgbm_branch_forecast(conn, sid, current_price, main_force_cost, chip_concentration, max_trade_days):
    if importlib.util.find_spec("lightgbm") is None:
        return {"status": "missing_dependency", "message": "尚未安裝 lightgbm，請先 `pip install lightgbm`。"}

    from lightgbm import LGBMRegressor

    features = _build_lightgbm_feature_frame(conn, sid, max_trade_days=max_trade_days, top_n=15)
    prices = pd.read_sql(
        """
        SELECT date, close
        FROM stock_ohlcv_daily
        WHERE stock_id = ?
        ORDER BY date ASC
        """,
        conn,
        params=(sid,),
    )

    if features.empty or prices.empty:
        return {"status": "insufficient_data", "message": "歷史特徵資料不足，無法訓練 LightGBM。"}

    features["end_date"] = pd.to_datetime(features["end_date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["future_close_5"] = prices["close"].shift(-5)
    prices["future_return_5d"] = ((prices["future_close_5"] - prices["close"]) / prices["close"]) * 100

    ds = features.merge(prices[["date", "close", "future_return_5d"]], left_on="end_date", right_on="date", how="inner")
    ds["cost_gap_20"] = ((ds["close"] - ds["avg_cost_20"]) / ds["avg_cost_20"]) * 100

    feature_cols = [
        "avg_cost_5", "avg_cost_20", "avg_cost_60",
        "net_vol_5", "net_vol_20", "net_vol_60",
        "concentration_5", "concentration_20", "concentration_60",
        "cost_gap_20",
    ]
    ds = ds[(ds["close"] > 0) & (ds["avg_cost_20"] > 0)].copy()
    ds = ds.replace([float("inf"), float("-inf")], pd.NA)
    model_ds = ds.dropna(subset=feature_cols + ["future_return_5d"]).copy()
    model_ds["future_return_5d"] = pd.to_numeric(model_ds["future_return_5d"], errors="coerce")

    # 避免極端異常值污染回歸，導致預測結果失真（例如 e+34%）。
    model_ds = model_ds[model_ds["future_return_5d"].abs() <= 40]

    if len(model_ds) < 40:
        return {"status": "insufficient_data", "message": f"可用樣本僅 {len(model_ds)} 筆，至少需要 40 筆。"}

    split_idx = int(len(model_ds) * 0.8)
    train_df = model_ds.iloc[:split_idx]
    test_df = model_ds.iloc[split_idx:]
    if test_df.empty:
        return {"status": "insufficient_data", "message": "測試資料不足，無法評估模型。"}

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["future_return_5d"])
    test_pred = model.predict(test_df[feature_cols])
    mae = (pd.Series(test_pred).reset_index(drop=True) - test_df["future_return_5d"].reset_index(drop=True)).abs().mean()

    latest_row = model_ds.iloc[-1].copy()
    latest_row["avg_cost_20"] = main_force_cost
    latest_row["concentration_20"] = chip_concentration
    latest_row["cost_gap_20"] = ((current_price - main_force_cost) / main_force_cost) * 100 if main_force_cost else 0
    forecast_raw = float(model.predict(pd.DataFrame([latest_row[feature_cols]]))[0])
    forecast = max(min(forecast_raw, 40.0), -40.0)

    feature_snapshot = {k: float(latest_row[k]) for k in feature_cols}
    contradictory_chip_signal = (
        feature_snapshot["net_vol_5"] <= 0
        and feature_snapshot["net_vol_20"] <= 0
        and feature_snapshot["net_vol_60"] <= 0
        and feature_snapshot["concentration_5"] <= 0
        and feature_snapshot["concentration_20"] <= 0
        and feature_snapshot["concentration_60"] <= 0
    )

    return {
        "status": "ok",
        "max_trade_days": int(max_trade_days),
        "samples": int(len(model_ds)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "mae": round(float(mae), 3),
        "pred_return_5d": round(forecast, 2),
        "pred_return_5d_raw": round(forecast_raw, 2),
        "feature_snapshot": feature_snapshot,
        "contradictory_chip_signal": bool(contradictory_chip_signal),
    }


def _run_lightgbm_branch_forecast(conn, sid, current_price, main_force_cost, chip_concentration, top_n=15):
    if importlib.util.find_spec("lightgbm") is None:
        return {"status": "missing_dependency", "message": "尚未安裝 lightgbm，請先 `pip install lightgbm`。"}

    from lightgbm import LGBMRegressor

    prices = pd.read_sql(
        """
        SELECT date, close
        FROM stock_ohlcv_daily
        WHERE stock_id = ?
        ORDER BY date ASC
        """,
        conn,
        params=(sid,),
    )
    branch_raw = pd.read_sql(
        """
        SELECT date, securities_trader_id, price, buy, sell
        FROM branch_price_daily
        WHERE stock_id = ?
        ORDER BY date ASC
        """,
        conn,
        params=(sid,),
    )

    if prices.empty or branch_raw.empty:
        return {"status": "insufficient_data", "message": "歷史特徵資料不足，無法訓練 LightGBM。"}

    prices["date"] = pd.to_datetime(prices["date"])
    branch_raw["date"] = pd.to_datetime(branch_raw["date"])

    trading_dates = sorted(branch_raw["date"].dropna().unique())
    if len(trading_dates) < 60:
        return {"status": "insufficient_data", "message": f"分點交易日僅 {len(trading_dates)} 天，至少需要 60 天。"}

    feature_rows = []
    for idx in range(len(trading_dates)):
        end_date = trading_dates[idx]
        if idx < 59:
            continue

        row = {"end_date": end_date}
        enough_history = True
        for window in [5, 20, 60]:
            start_idx = idx - window + 1
            if start_idx < 0:
                enough_history = False
                break

            window_dates = trading_dates[start_idx: idx + 1]
            window_df = branch_raw[branch_raw["date"].isin(window_dates)][["securities_trader_id", "price", "buy", "sell"]]
            if window_df.empty:
                enough_history = False
                break

            avg_cost, total_net_volume, concentration = _compute_interval_metrics(window_df, top_n=top_n)
            row[f"avg_cost_{window}"] = avg_cost
            row[f"net_vol_{window}"] = total_net_volume
            row[f"concentration_{window}"] = concentration

        if enough_history:
            feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    if features.empty:
        return {"status": "insufficient_data", "message": "無法建立滾動特徵，請先同步分點歷史資料。"}

    prices["future_close_5"] = prices["close"].shift(-5)
    prices["future_return_5d"] = ((prices["future_close_5"] - prices["close"]) / prices["close"]) * 100

    ds = features.merge(prices[["date", "close", "future_return_5d"]], left_on="end_date", right_on="date", how="inner")
    ds["cost_gap_20"] = ((ds["close"] - ds["avg_cost_20"]) / ds["avg_cost_20"]) * 100

    feature_cols = [
        "avg_cost_5", "avg_cost_20", "avg_cost_60",
        "net_vol_5", "net_vol_20", "net_vol_60",
        "concentration_5", "concentration_20", "concentration_60",
        "cost_gap_20",
    ]
    model_ds = ds.dropna(subset=feature_cols + ["future_return_5d"]).copy()

    if len(model_ds) < 40:
        return {"status": "insufficient_data", "message": f"可用樣本僅 {len(model_ds)} 筆，至少需要 40 筆。"}

    split_idx = int(len(model_ds) * 0.8)
    train_df = model_ds.iloc[:split_idx]
    test_df = model_ds.iloc[split_idx:]
    if test_df.empty:
        return {"status": "insufficient_data", "message": "測試資料不足，無法評估模型。"}

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["future_return_5d"])
    test_pred = model.predict(test_df[feature_cols])
    mae = (pd.Series(test_pred).reset_index(drop=True) - test_df["future_return_5d"].reset_index(drop=True)).abs().mean()

    latest_row = model_ds.iloc[-1].copy()
    latest_row["avg_cost_20"] = main_force_cost
    latest_row["concentration_20"] = chip_concentration
    latest_row["cost_gap_20"] = ((current_price - main_force_cost) / main_force_cost) * 100 if main_force_cost else 0
    forecast = float(model.predict(pd.DataFrame([latest_row[feature_cols]]))[0])

    return {
        "status": "ok",
        "samples": int(len(model_ds)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "mae": round(float(mae), 3),
        "pred_return_5d": round(forecast, 2),
    }




def show_branch_analysis():
    st.markdown("### 🔍 專業級分點籌碼與產業聯動診斷")
    cfg = database.load_config()
    conn = database.get_db_connection(cfg)

    # 以分點明細為主：優先 branch_trader_daily_detail，無資料時回退 branch_price_daily。
    primary_rows = conn.execute(
        """
        SELECT DISTINCT stock_id
        FROM branch_trader_daily_detail
        WHERE stock_id IS NOT NULL AND stock_id != ''
        ORDER BY stock_id ASC
        """
    ).fetchall()
    source_table = "branch_trader_daily_detail" if primary_rows else "branch_price_daily"

    stock_rows = primary_rows if primary_rows else conn.execute(
        """
        SELECT DISTINCT stock_id
        FROM branch_price_daily
        WHERE stock_id IS NOT NULL AND stock_id != ''
        ORDER BY stock_id ASC
        """
    ).fetchall()
    stock_ids = [str(r[0]) for r in stock_rows if r and r[0]]

    universe = cfg.get("universe", [])
    name_map = {str(s.get("stock_id")): s.get("name", "") for s in universe if s.get("stock_id")}

    stock_options = {
        (f"{sid} {name_map[sid]}".strip() if name_map.get(sid) else sid): sid
        for sid in stock_ids
    }

    if not stock_options:
        st.warning("分點明細資料表尚無可分析股票，請先同步資料。")
        conn.close()
        return

    c1, c2, c3, c4, c5 = st.columns([1.7, 1.5, 1.1, 0.7, 1.2])
    with c1:
        sid_label = st.selectbox("分析標的", list(stock_options.keys()), label_visibility="collapsed")
        sid = stock_options[sid_label]

    date_bounds = conn.execute(
        f"""
        SELECT MIN(date), MAX(date)
        FROM {source_table}
        WHERE stock_id = ?
        """,
        (sid,),
    ).fetchone()
    min_date_raw, max_date_raw = date_bounds if date_bounds else (None, None)
    min_available_date = pd.to_datetime(min_date_raw).date() if min_date_raw else None
    max_available_date = pd.to_datetime(max_date_raw).date() if max_date_raw else None

    with c2:
        if max_available_date:
            default_end = max_available_date
            default_start = max(min_available_date, default_end - pd.Timedelta(days=60))
            date_range = st.date_input(
                "日期區間",
                value=[default_start, default_end],
                min_value=min_available_date,
                max_value=max_available_date,
                label_visibility="collapsed",
            )
        else:
            date_range = st.date_input(
                "日期區間",
                value=[pd.to_datetime("today") - pd.Timedelta(days=60), pd.to_datetime("today")],
                label_visibility="collapsed",
            )
    with c3:
        analyze_btn = st.button("🚀 執行", use_container_width=True)
    with c4:
        if st.button("🔄", use_container_width=True): st.rerun()
    with c5:
        top_n = st.number_input("買賣超分點家數", min_value=1, max_value=50, value=15, step=1)

    ind_cols = database.get_table_columns(conn, "stock_industry_chain")
    industry_col = database.match_column(ind_cols, ["industry"]) 
    industry_name = "未知產業"
    if industry_col:
        row = conn.execute(f"SELECT {industry_col} FROM stock_industry_chain WHERE stock_id=?", (sid,)).fetchone()
        industry_name = row[0] if row else "未知產業"
    
    st.info(f"📍 當前標的：**{sid_label}** | 所屬產業鏈：**{industry_name}**")

    if not max_available_date:
        st.warning("此標的尚無分點資料，暫時無法執行分點分析。")
        return

    if not (isinstance(date_range, (list, tuple)) and len(date_range) == 2): return
    start_d, end_d = pd.to_datetime(date_range[0]).date(), pd.to_datetime(date_range[1]).date()
    date_sql = f"date BETWEEN '{start_d.isoformat()}' AND '{end_d.isoformat()}'"

    try:
        current_price = conn.execute(f"SELECT close FROM stock_ohlcv_daily WHERE stock_id='{sid}' ORDER BY date DESC LIMIT 1").fetchone()[0] or 0
        total_vol = conn.execute(f"SELECT SUM(buy) FROM {source_table} WHERE stock_id='{sid}' AND {date_sql}").fetchone()[0] or 1

        df = pd.read_sql(f"""
            SELECT securities_trader AS "分點", SUM(buy - sell) AS "淨張數", 
                   ROUND(SUM((buy - sell) * price) / NULLIF(SUM(buy - sell), 0), 2) AS "均價"
            FROM {source_table} WHERE stock_id = '{sid}' AND {date_sql}
            GROUP BY securities_trader HAVING "淨張數" != 0 ORDER BY ABS("淨張數") DESC LIMIT 20
        """, conn)
        df['獲利%'] = (((current_price - df['均價']) / df['均價']) * 100).round(2)

        interval_df = pd.read_sql(
            f"""
            SELECT securities_trader_id, price, buy, sell
            FROM {source_table}
            WHERE stock_id = ? AND date >= ? AND date <= ?
            """,
            conn,
            params=(sid, start_d.isoformat(), end_d.isoformat()),
        )

        if interval_df.empty:
            st.warning("所選日期區間沒有分點資料，請調整日期後再執行。")
            return

        main_force_cost, total_net_volume, chip_concentration = _compute_interval_metrics(interval_df, top_n=top_n)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("核心主力加權成本", f"${main_force_cost}")
        with m2:
            cost_gap = round(((current_price - main_force_cost) / main_force_cost) * 100, 2) if main_force_cost > 0 else 0
            st.metric("目前價格位階", f"{cost_gap}%", delta=f"{cost_gap}%", delta_color="normal")
        with m3:
            st.metric("買方籌碼集中度", f"{chip_concentration:.2f}%")
            st.caption("定義：前 N 大買超分點淨買超總和 / 全市場總買量。負值代表賣壓強於買盤。")

        default_max_trade_days = cfg.get("branch_analysis", {}).get("lightgbm_default_max_trade_days", 320)
        lightgbm_max_trade_days = _resolve_lightgbm_max_trade_days(
            conn,
            sid,
            start_d,
            end_d,
            default_max_trade_days,
        )
        lgbm_result = _run_lightgbm_branch_forecast(
            conn,
            sid,
            current_price,
            main_force_cost,
            chip_concentration,
            max_trade_days=lightgbm_max_trade_days,
        )
        with st.expander("📈 LightGBM 分點訊號（未來 5 日）", expanded=True):
            if lgbm_result["status"] == "ok":
                st.metric("預估 5 日報酬", f"{lgbm_result['pred_return_5d']:.2f}%")
                st.caption(
                    f"建模交易日數 {lgbm_result['max_trade_days']}，"
                    f"樣本數 {lgbm_result['samples']}（訓練 {lgbm_result['train_samples']} / 測試 {lgbm_result['test_samples']}），"
                    f"測試 MAE：{lgbm_result['mae']}"
                )
                fs = lgbm_result.get("feature_snapshot", {})
                if fs:
                    st.caption(
                        "模型輸入快照："
                        f"net_vol(5/20/60)=({fs['net_vol_5']:.0f}/{fs['net_vol_20']:.0f}/{fs['net_vol_60']:.0f})，"
                        f"concentration(5/20/60)=({fs['concentration_5']:.2f}%/{fs['concentration_20']:.2f}%/{fs['concentration_60']:.2f}%)，"
                        f"cost_gap_20={fs['cost_gap_20']:.2f}%"
                    )

                if lgbm_result.get("contradictory_chip_signal") and lgbm_result["pred_return_5d"] > 0:
                    st.warning(
                        "⚠️ 目前籌碼為連續賣超/負集中，但模型仍預測上漲。"
                        "這通常代表模型從歷史樣本學到『成本位階/均值回歸』訊號暫時強於籌碼訊號，"
                        "請搭配風險控管與其他模組交叉確認。"
                    )
            elif lgbm_result["status"] == "missing_dependency":
                st.warning(lgbm_result["message"])
            else:
                st.info(lgbm_result["message"])

        lgbm_result = _run_lightgbm_branch_forecast(conn, sid, current_price, main_force_cost, chip_concentration, top_n=top_n)
        with st.expander("📈 LightGBM 分點訊號（未來 5 日）", expanded=True):
            if lgbm_result["status"] == "ok":
                st.metric("預估 5 日報酬", f"{lgbm_result['pred_return_5d']}%")
                st.caption(
                    f"樣本數 {lgbm_result['samples']}（訓練 {lgbm_result['train_samples']} / 測試 {lgbm_result['test_samples']}），"
                    f"測試 MAE：{lgbm_result['mae']}"
                )
            elif lgbm_result["status"] == "missing_dependency":
                st.warning(lgbm_result["message"])
            else:
                st.info(lgbm_result["message"])

        if chip_concentration > 30:
            st.error(f"⚠️ 偵測到籌碼異常集中！前五大分點買超佔比達 {chip_concentration}%")
        
        st.caption("最新 rolling 快照：5日 / 20日 / 60日")
        w1, w2, w3 = st.columns(3)
        for col, window in zip([w1, w2, w3], [5, 20, 60]):
            hist_dates = conn.execute(
                f"""
                SELECT DISTINCT date
                FROM {source_table}
                WHERE stock_id = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (sid, int(window)),
            ).fetchall()

            with col:
                if len(hist_dates) == int(window):
                    dates = sorted([str(d[0])[:10] for d in hist_dates if d and d[0]])
                    win_start, win_end = dates[0], dates[-1]
                    snap_df = pd.read_sql(
                        f"""
                        SELECT securities_trader_id, price, buy, sell
                        FROM {source_table}
                        WHERE stock_id = ?
                          AND date >= ?
                          AND date <= ?
                        """,
                        conn,
                        params=(sid, win_start, win_end),
                    )
                    row = _compute_window_snapshot_from_branch(snap_df, win_end, top_n=15)
                else:
                    row = None

                if row is not None:
                    st.metric(f"{window}日均價成本", f"${row[0]:.2f}")
                    st.caption(format_snapshot_caption(row))
                else:
                    st.metric(f"{window}日均價成本", "N/A")
                    st.caption("尚無快照")

        col_left, col_right = st.columns([5, 5])
        with col_left:
            st.write("🏦 **Top 20 進出分點盈虧**")
            st.dataframe(df.style.applymap(color_volume, subset=['淨張數']), use_container_width=True, hide_index=True, height=500)
        with col_right:
            if analyze_btn:
                with st.spinner("AI 診斷中..."):
                    st.markdown(analyze_branch_pro(cfg, sid, df, total_vol, current_price, industry_name, pd.DataFrame(), main_force_cost, chip_concentration))
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df["分點"], y=df["淨張數"], marker_color=['#FF0000' if x > 0 else '#008000' for x in df["淨張數"]]))
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"執行錯誤：{e}")
    finally: conn.close()
