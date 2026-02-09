import streamlit as st
import pandas as pd
import database
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 核心 AI 呼叫工具
def _call_nim(cfg, messages):
    llm_cfg = cfg.get("llm", {})
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_cfg.get('api_key')}", "Content-Type": "application/json"}
    payload = {"model": llm_cfg.get("model"), "messages": messages, "temperature": 0.0, "max_tokens": 2000}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def show_stock_health():
    st.markdown("### 🏥 全方位籌碼體質診斷")
    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}

    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'chart'

    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.2, 1.2]) # 微調欄位寬度以容納日期
    with c1:
        st.caption("分析標的")
        selected_stock = st.selectbox("標的", list(stock_options.keys()), label_visibility="collapsed")
        sid = stock_options[selected_stock]
    with c2:
        # ✅ 修改：將數字輸入改為日期區間選擇
        st.caption("分析區間")
        def_s = pd.to_datetime("today") - pd.Timedelta(days=60)
        def_e = pd.to_datetime("today")
        h_range = st.date_input("區間", value=[def_s, def_e], label_visibility="collapsed")
        
        if isinstance(h_range, (list, tuple)) and len(h_range) == 2:
            start_d, end_d = h_range[0], h_range[1]
        else:
            start_d, end_d = def_s, def_e
    with c3:
        if st.button("🚀 啟動量化診斷", use_container_width=True):
            st.session_state.view_mode = 'analysis'
    with c4:
        if st.button("📊 趨勢圖表", use_container_width=True):
            st.session_state.view_mode = 'chart'

    try:
        m_cols = database.get_table_columns(conn, "margin_short_daily")
        d_cols = database.get_table_columns(conn, "stock_day_trading_daily")
        i_cols = database.get_table_columns(conn, "institutional_investors_daily")
        
        m_bal = database.match_column(m_cols, ["Margin", "Balance"])
        s_bal = database.match_column(m_cols, ["Short", "Balance"])
        d_vol = database.match_column(d_cols, ["Volume"])

        query = f"""
            SELECT o.date AS "日期", o.close AS "收盤", o.Trading_Volume AS "成交量",
                   SUM(CASE WHEN i.name LIKE '%外資%' THEN i.buy - i.sell ELSE 0 END) AS "外資",
                   SUM(CASE WHEN i.name LIKE '%投信%' THEN i.buy - i.sell ELSE 0 END) AS "投信",
                   MAX(m."{m_bal}") AS "融資", MAX(m."{s_bal}") AS "融券",
                   MAX(d."{d_vol}") AS "當沖量",
                   MAX(f.active_buy_vol) AS "主動買", MAX(f.active_sell_vol) AS "主動賣"
            FROM stock_ohlcv_daily o
            LEFT JOIN institutional_investors_daily i ON o.stock_id = i.stock_id AND o.date = i.date
            LEFT JOIN margin_short_daily m ON o.stock_id = m.stock_id AND o.date = m.date
            LEFT JOIN stock_day_trading_daily d ON o.stock_id = d.stock_id AND o.date = d.date
            LEFT JOIN stock_active_flow_daily f ON o.stock_id = f.stock_id AND o.date = f.date
            WHERE o.stock_id = '{sid}' AND o.date BETWEEN '{start_d}' AND '{end_d}'
            GROUP BY o.date ORDER BY o.date DESC
        """
        df = pd.read_sql(query, conn)
        
        df['主動淨力道'] = (df['主動買'].fillna(0) - df['主動賣'].fillna(0))
        df['主動強度%'] = ((df['主動淨力道'] / df['成交量'].replace(0, 1)) * 100).round(2)

        # ✅ 步驟二新增：強度對比與異常偵測
        avg_strength_20d = df['主動強度%'].iloc[1:21].mean() if len(df) > 20 else df['主動強度%'].mean()
        today_strength = df['主動強度%'].iloc[0]
        
        s1, s2 = st.columns(2)
        with s1:
            st.metric("今日主動強度", f"{today_strength}%", delta=f"{round(today_strength - avg_strength_20d, 2)}% vs 20D平均")
        with s2:
            if today_strength > 0 and today_strength > (abs(avg_strength_20d) * 3):
                st.success(f"🔥 偵測到倍數型主動買盤！強度為平均的 {round(today_strength/abs(avg_strength_20d), 1)} 倍。")
            elif today_strength < 0 and abs(today_strength) > (abs(avg_strength_20d) * 3):
                st.error(f"💀 偵測到倍數型砸盤賣壓！賣力為平均的 {round(abs(today_strength)/abs(avg_strength_20d), 1)} 倍。")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.dataframe(df, use_container_width=True, hide_index=True, height=750)

        with col_right:
            if st.session_state.view_mode == 'analysis':
                with st.spinner("AI 執行診斷中..."):
                    prompt = f"分析 {selected_stock} 籌碼數據，注意今日強度 {today_strength}% 與平均 {avg_strength_20d}% 的差異：\n{df.to_csv(index=False)}"
                    st.markdown(_call_nim(cfg, [{"role": "user", "content": prompt}]))
            else:
                df_plot = df.sort_values("日期")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    subplot_titles=("📈 價格與成交量", "📊 融資融券趨勢", "🔥 主動攻擊力道 (紅買綠賣)"),
                                    specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]])
                fig.add_trace(go.Scatter(x=df_plot["日期"], y=df_plot["收盤"], name="收盤價", line=dict(color="#1f77b4")), row=1, col=1)
                fig.add_trace(go.Bar(x=df_plot["日期"], y=df_plot["成交量"], name="成交量", opacity=0.15, marker_color="gray"), row=1, col=1, secondary_y=True)
                fig.add_trace(go.Scatter(x=df_plot["日期"], y=df_plot["融資"], name="融資", line=dict(color="#ff7f0e")), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_plot["日期"], y=df_plot["融券"], name="融券", line=dict(color="#2ca02c")), row=2, col=1, secondary_y=True)
                fig.add_trace(go.Bar(
                    x=df_plot["日期"], 
                    y=df_plot["主動強度%"], 
                    name="主動強度%", 
                    marker_color=['red' if x > 0 else 'green' for x in df_plot["主動強度%"]]
                ), row=3, col=1)
                fig.update_xaxes(type='category')
                fig.update_layout(height=800, margin=dict(l=10, r=10, t=60, b=50), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 診斷出錯：{str(e)}")
    finally:
        conn.close()