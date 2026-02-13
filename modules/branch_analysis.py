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


def _upsert_interval_metrics(conn, sid, start_date, end_date, avg_cost, total_net_volume, concentration):
    window_days = (pd.to_datetime(end_date).date() - pd.to_datetime(start_date).date()).days + 1
    conn.execute(
        """
        INSERT OR REPLACE INTO branch_weighted_cost
        (stock_id, start_date, end_date, avg_cost, total_net_volume, concentration, window_type, window_days)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sid,
            str(start_date),
            str(end_date),
            float(avg_cost or 0),
            int(total_net_volume or 0),
            float(concentration or 0),
            "user_custom",
            int(window_days),
        ),
    )


def _load_window_snapshot(conn, sid, window):
    return conn.execute(
        """
        SELECT avg_cost, total_net_volume, concentration, end_date
        FROM branch_weighted_cost
        WHERE stock_id = ? AND window_type = 'rolling' AND window_days = ?
        ORDER BY end_date DESC
        LIMIT 1
        """,
        (sid, int(window)),
    ).fetchone()


def _run_lightgbm_branch_forecast(conn, sid, current_price, main_force_cost, chip_concentration):
    if importlib.util.find_spec("lightgbm") is None:
        return {"status": "missing_dependency", "message": "尚未安裝 lightgbm，請先 `pip install lightgbm`。"}

    from lightgbm import LGBMRegressor

    features = pd.read_sql(
        """
        SELECT
            end_date,
            MAX(CASE WHEN window_days = 5 THEN avg_cost END) AS avg_cost_5,
            MAX(CASE WHEN window_days = 20 THEN avg_cost END) AS avg_cost_20,
            MAX(CASE WHEN window_days = 60 THEN avg_cost END) AS avg_cost_60,
            MAX(CASE WHEN window_days = 5 THEN total_net_volume END) AS net_vol_5,
            MAX(CASE WHEN window_days = 20 THEN total_net_volume END) AS net_vol_20,
            MAX(CASE WHEN window_days = 60 THEN total_net_volume END) AS net_vol_60,
            MAX(CASE WHEN window_days = 5 THEN concentration END) AS concentration_5,
            MAX(CASE WHEN window_days = 20 THEN concentration END) AS concentration_20,
            MAX(CASE WHEN window_days = 60 THEN concentration END) AS concentration_60
        FROM branch_weighted_cost
        WHERE stock_id = ?
          AND window_type = 'rolling'
          AND window_days IN (5, 20, 60)
        GROUP BY end_date
        ORDER BY end_date ASC
        """,
        conn,
        params=(sid,),
    )
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
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}

    c1, c2, c3, c4, c5 = st.columns([1.7, 1.5, 1.1, 0.7, 1.2])
    with c1:
        sid_label = st.selectbox("分析標的", list(stock_options.keys()), label_visibility="collapsed")
        sid = stock_options[sid_label]
    with c2:
        # ✅ 修改：預設值改為今日往前 60 天
        def_s = pd.to_datetime("today") - pd.Timedelta(days=60)
        def_e = pd.to_datetime("today")
        date_range = st.date_input("日期區間", value=[def_s, def_e], label_visibility="collapsed")
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

    if not (isinstance(date_range, (list, tuple)) and len(date_range) == 2): return
    start_d, end_d = pd.to_datetime(date_range[0]).date(), pd.to_datetime(date_range[1]).date()
    date_sql = f"date BETWEEN '{start_d.isoformat()}' AND '{end_d.isoformat()}'"

    try:
        current_price = conn.execute(f"SELECT close FROM stock_ohlcv_daily WHERE stock_id='{sid}' ORDER BY date DESC LIMIT 1").fetchone()[0] or 0
        total_vol = conn.execute(f"SELECT SUM(buy) FROM branch_price_daily WHERE stock_id='{sid}' AND {date_sql}").fetchone()[0] or 1

        df = pd.read_sql(f"""
            SELECT securities_trader AS "分點", SUM(buy - sell) AS "淨張數", 
                   ROUND(SUM((buy - sell) * price) / NULLIF(SUM(buy - sell), 0), 2) AS "均價"
            FROM branch_price_daily WHERE stock_id = '{sid}' AND {date_sql}
            GROUP BY securities_trader HAVING "淨張數" != 0 ORDER BY ABS("淨張數") DESC LIMIT 20
        """, conn)
        df['獲利%'] = (((current_price - df['均價']) / df['均價']) * 100).round(2)

        interval_df = pd.read_sql(
            """
            SELECT securities_trader_id, price, buy, sell
            FROM branch_price_daily
            WHERE stock_id = ? AND date >= ? AND date <= ?
            """,
            conn,
            params=(sid, start_d.isoformat(), end_d.isoformat()),
        )
        main_force_cost, total_net_volume, chip_concentration = _compute_interval_metrics(interval_df, top_n=top_n)

        if analyze_btn:
            _upsert_interval_metrics(conn, sid, start_d.isoformat(), end_d.isoformat(), main_force_cost, total_net_volume, chip_concentration)
            conn.commit()

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("核心主力加權成本", f"${main_force_cost}")
        with m2:
            cost_gap = round(((current_price - main_force_cost) / main_force_cost) * 100, 2) if main_force_cost > 0 else 0
            st.metric("目前價格位階", f"{cost_gap}%", delta=f"{cost_gap}%", delta_color="normal")
        with m3: st.metric("買方籌碼集中度", f"{chip_concentration}%")

        lgbm_result = _run_lightgbm_branch_forecast(conn, sid, current_price, main_force_cost, chip_concentration)
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
            row = _load_window_snapshot(conn, sid, window)
            with col:
                if row:
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
