import streamlit as st
import pandas as pd
import database
import requests
import plotly.graph_objects as go

def _call_nim_prediction(cfg, prompt):
    """ 呼叫 NVIDIA NIM 進行鏈式思考 (CoT) 預測 """
    llm_cfg = cfg.get("llm", {})
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_cfg.get('api_key')}", "Content-Type": "application/json"}
    payload = {
        "model": llm_cfg.get("model"),
        "messages": [
            {"role": "system", "content": "你是一位量化交易專家，擅長結合技術面、籌碼面與主動力道進行股價短期趨勢預測。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1 # 降低隨機性
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.json()["choices"][0]["message"]["content"]

def show_prediction():
    st.header("🔮 AI 股價趨勢預測中心")
    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}
    
    target_label = st.selectbox("選擇預測標的", list(stock_options.keys()))
    sid = stock_options[target_label]

    # ✅ 修改：新增日期區間選擇器
    def_s = pd.to_datetime("today") - pd.Timedelta(days=60)
    def_e = pd.to_datetime("today")
    p_range = st.date_input("AI 參考特徵區間", value=[def_s, def_e])
    
    if isinstance(p_range, (list, tuple)) and len(p_range) == 2:
        start_d, end_d = p_range[0], p_range[1]
    else:
        start_d, end_d = def_s, def_e

    if st.button("🚀 執行多維度特徵分析與預測", use_container_width=True):
        with st.spinner("正在彙整特徵數據並執行 AI 推理..."):
            # ✅ 修改：將 LIMIT 5 改為 BETWEEN 篩選
            df_tech = pd.read_sql(f"SELECT close, Trading_Volume FROM stock_ohlcv_daily WHERE stock_id='{sid}' AND date BETWEEN '{start_d}' AND '{end_d}' ORDER BY date DESC", conn)
            df_flow = pd.read_sql(f"SELECT active_buy_vol, active_sell_vol FROM stock_active_flow_daily WHERE stock_id='{sid}' AND date BETWEEN '{start_d}' AND '{end_d}' ORDER BY date DESC", conn)
            
            cost_data = conn.execute(f"""
                SELECT SUM((buy - sell) * price) / SUM(buy - sell) as cost,
                       SUM(buy - sell) as net_vol
                FROM (SELECT * FROM branch_price_daily WHERE stock_id='{sid}' AND date BETWEEN '{start_d}' AND '{end_d}' ORDER BY date DESC)
                WHERE buy > sell
            """).fetchone()

            # ✅ 2. 構建 AI 預測 Prompt
            features_prompt = f"""
            請根據以下 {target_label} 的數據特徵進行未來 3-5 個交易日的趨勢預測：
            
            【技術面特徵】
            - 近 5 日收盤價：{df_tech['close'].tolist()}
            - 近 5 日成交量：{df_tech['Trading_Volume'].tolist()}
            
            【主動力道特徵】
            - 近 5 日主動買：{df_flow['active_buy_vol'].tolist() if not df_flow.empty else '無資料'}
            - 近 5 日主動賣：{df_flow['active_sell_vol'].tolist() if not df_flow.empty else '無資料'}
            
            【籌碼成本特徵】
            - 核心主力成本：{round(cost_data[0], 2) if cost_data[0] else '未知'}
            - 目前價格位階：{round((df_tech['close'].iloc[0] / cost_data[0] - 1)*100, 2) if cost_data[0] else '未知'}%
            
            請嚴格以下列格式回覆：
            ### 🏁 預測結論：[看多/看空/震盪]
            ---
            1. **趨勢理由**：(結合主動力道與成本位階說明)
            2. **關鍵位預測**：(支撐位與目標位)
            3. **信心指數**：(0-100%)
            """
            
            prediction_result = _call_nim_prediction(cfg, features_prompt)
            st.markdown(prediction_result)

    conn.close()