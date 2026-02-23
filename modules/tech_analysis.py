import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import database
import numpy as np 

def show_tech_analysis():
    st.subheader("📈 專業級技術分析儀表板 (籌碼整合版)")
    
    # 1. 讀取設定與股票清單
    cfg = database.load_config()
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}
    
    col1, col2 = st.columns(2)
    with col1:
        selected_stock_label = st.selectbox("選擇股票標的", list(stock_options.keys()))
        target_sid = stock_options[selected_stock_label]
    with col2:
        # ✅ 修正：由日期區間決定顯示範圍，不再使用消失的 days 變數
        def_s = pd.to_datetime("today") - pd.Timedelta(days=60)
        def_e = pd.to_datetime("today")
        t_range = st.date_input("顯示區間", value=[def_s, def_e])
        
        if isinstance(t_range, (list, tuple)) and len(t_range) == 2:
            start_d, end_d = t_range[0], t_range[1]
        else:
            start_d, end_d = def_s, def_e

    conn = database.get_db_connection(cfg)
    # ✅ 修正：SQL 已經幫我們過濾好日期了
    query = f"SELECT * FROM stock_ohlcv_daily WHERE stock_id = '{target_sid}' AND date BETWEEN '{start_d}' AND '{end_d}' ORDER BY date ASC"
    df = pd.read_sql(query, conn)

    # ✅ 修正：計算區間內前五大主力成本 (W.A.C)
    cost_query = f"""
        SELECT SUM((buy - sell) * price) / SUM(buy - sell) as w_cost
        FROM (
            SELECT securities_trader, SUM(buy - sell) as net_vol, price
            FROM branch_price_daily 
            WHERE stock_id = '{target_sid}' AND date BETWEEN '{start_d}' AND '{end_d}'
            GROUP BY securities_trader HAVING net_vol > 0
            ORDER BY net_vol DESC LIMIT 5
        )
    """
    try:
        main_force_cost = conn.execute(cost_query).fetchone()[0] or 0
    except:
        main_force_cost = 0
    conn.close()

    if df.empty:
        st.warning("⚠️ 該區間內查無股價資料，請調整日期。")
        return

    # 3. 技術指標計算 (保留原始邏輯)
    df['date'] = pd.to_datetime(df['date'])
    df["MA10"] = ta.sma(df["close"], length=10)
    df["MA20"] = ta.sma(df["close"], length=20)
    df["MA60"] = ta.sma(df["close"], length=60)
    df["MA120"] = ta.sma(df["close"], length=120)
    df["MA240"] = ta.sma(df["close"], length=240)
    
    kd = ta.stoch(df["max"], df["min"], df["close"], k=9, d=3)
    df = pd.concat([df, kd], axis=1)
    df["RSI"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # ✅ 解決 NameError：因為 SQL 已經過濾日期，df 即為要顯示的資料
    df_display = df.copy()

    # ✅ 量能檔次分析 POC (支撐/壓力參考)
    counts, bin_edges = np.histogram(df_display['close'], bins=30, weights=df_display['Trading_Volume'])
    poc_price = bin_edges[np.argmax(counts)]

    # 4. 繪製圖表 (保留原始 fig 配置)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=("K線與主力成本", "成交量", "KD / RSI", "MACD"))

    fig.add_trace(go.Candlestick(
        x=df_display['date'], open=df_display['open'], high=df_display['max'], 
        low=df_display['min'], close=df_display['close'], name="K線",
        increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    
    # 繪製參考線
    if main_force_cost > 0:
        fig.add_hline(y=main_force_cost, line_dash="dash", line_color="orange", 
                      annotation_text=f"核心主力成本:{round(main_force_cost,1)}", row=1, col=1)
    fig.add_hline(y=poc_price, line_dash="solid", line_color="gray", opacity=0.3,
                  annotation_text=f"量能密集區:{round(poc_price,1)}", row=1, col=1)

    ma_configs = [("MA10", "#FF9900"), ("MA20", "#FF00FF"), ("MA60", "#00CC00"), ("MA120", "#0000FF"), ("MA240", "#FF0000")]
    for ma, color in ma_configs:
        if ma in df_display.columns:
            fig.add_trace(go.Scatter(x=df_display['date'], y=df_display[ma], name=ma, line=dict(width=1.2, color=color)), row=1, col=1)

    # 成交量區塊
    fig.add_trace(go.Bar(x=df_display['date'], y=df_display['Trading_Volume'], name="成交量", marker_color="#444444"), row=2, col=1)
    
    # 指標區塊
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['STOCHk_9_3_3'], name="K值 (9,3)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['STOCHd_9_3_3'], name="D值 (9,3)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['RSI'], name="RSI (14)", line=dict(color='purple')), row=3, col=1)
    
    # pandas_ta 不同版本可能會產生不同 MACD 欄位命名，這裡動態尋找避免 KeyError
    macd_col = next((c for c in df_display.columns if c.startswith("MACD_") and not c.startswith("MACDh_") and not c.startswith("MACDs_")), None)
    macd_hist_col = next((c for c in df_display.columns if c.startswith("MACDh_")), None)
    macd_signal_col = next((c for c in df_display.columns if c.startswith("MACDs_")), None)

    if macd_hist_col:
        macd_hist_colors = ['red' if val >= 0 else 'green' for val in df_display[macd_hist_col]]
        fig.add_trace(go.Bar(x=df_display['date'], y=df_display[macd_hist_col], name="MACD柱", marker_color=macd_hist_colors), row=4, col=1)

    if macd_col:
        fig.add_trace(go.Scatter(x=df_display['date'], y=df_display[macd_col], name="DIF", line=dict(color='black')), row=4, col=1)

    if macd_signal_col:
        fig.add_trace(go.Scatter(x=df_display['date'], y=df_display[macd_signal_col], name="Signal", line=dict(color='blue')), row=4, col=1)

    if not any([macd_col, macd_hist_col, macd_signal_col]):
        st.warning("⚠️ MACD 欄位未產生，可能是資料筆數不足或套件版本差異。")

    # 5. 隱藏非交易日邏輯
    all_dates = pd.date_range(start=df_display['date'].min(), end=df_display['date'].max())
    trading_dates = df_display['date'].dt.date.unique()
    missing_dates = [d.strftime("%Y-%m-%d") for d in all_dates if d.date() not in trading_dates]
    
    fig.update_xaxes(rangebreaks=[dict(values=missing_dates)], type='date')
    fig.update_layout(height=1100, xaxis_rangeslider_visible=False, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    show_tech_analysis()
