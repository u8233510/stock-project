import streamlit as st
import pandas as pd
import database
import requests
from duckduckgo_search import DDGS

# 1. 核心 AI 呼叫工具 (保持穩定，未更動)
def _call_nim_fundamental(cfg, prompt):
    llm_cfg = cfg.get("llm", {})
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_cfg.get('api_key')}", "Content-Type": "application/json"}
    payload = {
        "model": llm_cfg.get("model"),
        "messages": [
            {"role": "system", "content": "你是一位專業的證券分析師。請優先參考連網搜尋到的事實，結合財務數據給出具體的投資評價，嚴禁虛構公司業務。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.json()["choices"][0]["message"]["content"]

def show_fundamental_analysis():
    st.markdown("### 💎 基本面數據全覽與 AI 診斷")
    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    universe = cfg.get("universe", [])
    stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}
    
    selected_stock = st.selectbox("請選擇分析標的", list(stock_options.keys()))
    sid = stock_options[selected_stock]
    
    # 統一基準日
    def_s = pd.to_datetime("today") - pd.Timedelta(days=60)
    def_e = pd.to_datetime("today")
    f_range = st.date_input("數據觀察基準日", value=[def_s, def_e])
    end_d = f_range[1] if len(f_range) == 2 else def_e

    st.divider()

    # --- 1. 數據抓取與格式化 ---
    
    # (1) 每月營收：除以 1000 並加上千分位 ","
    rev_query = f"SELECT date as '日期', revenue FROM stock_month_revenue_monthly WHERE stock_id='{sid}' ORDER BY date DESC LIMIT 12"
    rev_df = pd.read_sql(rev_query, conn)
    if not rev_df.empty:
        # ✅ 修正關鍵：統一欄位名稱為 '營收(百萬元)'，並加上千分位
        rev_df['營收(百萬元)'] = (rev_df['revenue'] / 1000).apply(lambda x: f"{x:,.2f}")
        rev_df = rev_df[['日期', '營收(百萬元)']]
    
    # (2+4) 每季獲利與 EPS：安全轉置處理
    profit_raw = pd.read_sql(f"SELECT date, type, value FROM stock_financial_statements WHERE stock_id='{sid}' AND type IN ('EPS', 'Net Profit') ORDER BY date DESC LIMIT 16", conn)
    if not profit_raw.empty:
        profit_df = profit_raw.pivot(index='date', columns='type', values='value').reset_index()
        # 安全重命名
        rename_map = {'date': '季度', 'EPS': 'EPS', 'Net Profit': '每季獲利'}
        profit_df = profit_df.rename(columns={k: v for k, v in rename_map.items() if k in profit_df.columns})
        # 加上千分位 (每季獲利)
        if '每季獲利' in profit_df.columns:
            profit_df['每季獲利'] = profit_df['每季獲利'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
    else:
        profit_df = pd.DataFrame()

    # (3+5) 本益比與殖利率
    val_query = f"SELECT date as '日期', PER as '本益比', dividend_yield as '殖利率%' FROM stock_per_pbr_daily WHERE stock_id='{sid}' AND date <= '{end_d}' ORDER BY date DESC LIMIT 10"
    valuation_df = pd.read_sql(val_query, conn)

    # (6+7) 現金股利與股息分配
    div_query = f"SELECT year as '年份', CashEarningsDistribution as '現金股利', StockEarningsDistribution as '股票股利' FROM stock_dividend WHERE stock_id='{sid}' ORDER BY year DESC LIMIT 5"
    div_df = pd.read_sql(div_query, conn)

    # --- 2. 表格化顯示分頁 ---
    tab1, tab2, tab3 = st.tabs(["📈 營收與獲利詳情", "💰 股利與估值看板", "🔍 AI 聯網趨勢報告"])

    with tab1:
        st.write("#### 1. 每月營收 (單位：百萬元)")
        st.dataframe(rev_df, use_container_width=True, hide_index=True)
        
        st.write("#### 2 & 4. 每季獲利與 EPS 歷程")
        if not profit_df.empty:
            # 確保僅顯示現有欄位
            cols_to_show = [c for c in ['季度', '每季獲利', 'EPS'] if c in profit_df.columns]
            st.dataframe(profit_df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("尚無獲利數據。")

    with tab2:
        st.write("#### 3 & 5. 本益比與殖利率變動")
        st.dataframe(valuation_df, use_container_width=True, hide_index=True)
        
        st.write("#### 6 & 7. 歷年股利分配 (現金與股票)")
        if not div_df.empty:
            st.table(div_df)
        else:
            st.info("尚無股利歷史數據。")

    with tab3:
        # ✅ 保留聯網搜尋邏輯
        if st.button(f"🚀 啟動 {selected_stock} 聯網事實分析", use_container_width=True):
            with st.spinner("正在搜尋最新產業地位與市場新聞..."):
                search_ctx = ""
                try:
                    with DDGS() as ddgs:
                        for r in ddgs.text(f"{selected_stock} 核心產品 產業地位 最新新聞", max_results=5):
                            search_ctx += f"\n- {r['title']}: {r['body']}"
                except: pass
                
                # 獲取 AI 參考數據
                latest_eps = profit_df['EPS'].iloc[0] if 'EPS' in profit_df.columns else "N/A"
                prompt = f"分析台股標的 {selected_stock} ({sid})。搜尋到的新聞事實：{search_ctx}。財務數據：最新季度 EPS 為 {latest_eps}。請產出深度診斷報告。"
                st.markdown(_call_nim_fundamental(cfg, prompt))

    conn.close()