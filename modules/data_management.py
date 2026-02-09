import streamlit as st
import pandas as pd
from FinMind.data import DataLoader
import database
import time
from datetime import datetime
import importlib

# 嘗試匯入 ingest_manager
try:
    import ingest_manager
except ImportError:
    ingest_manager = None

def run_minute_task(cfg):
    """ 
    精準補洞模式：
    只針對「分鐘表 (stock_ohlcv_minute)」缺失的日期進行更新。
    即使日線表已有資料，只要分鐘表沒資料，就會補洞並重新覆蓋日線數據。
    """
    dl = DataLoader()
    dl.login_by_token(api_token=cfg["finmind"]["api_token"])
    
    stock_list = cfg.get("universe", []) 
    min_cfg = cfg.get("ingest_minute", {})
    start_date = min_cfg["start_date"] 
    end_date = min_cfg.get("end_date") or datetime.now().strftime("%Y-%m-%d")
    date_range = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    
    p_bar = st.progress(0)
    p_text = st.empty()
    count = 0
    total = len(date_range) * len(stock_list)

    conn = database.get_db_connection(cfg)
    # 獲取分鐘表欄位名稱 (通常是 date_time)
    min_cols = database.get_table_columns(conn, "stock_ohlcv_minute")
    time_col = database.match_column(min_cols, ["date"]) or "date_time"

    for d in date_range:
        # 排除週末
        if pd.to_datetime(d).weekday() >= 5: continue 
        
        for stock in stock_list:
            count += 1
            sid = stock["stock_id"]
            
            # ✅ 關鍵修正：檢查「分鐘表」而非「日計表」
            # 使用 date() 函數確保日期格式比對一致 (防止 2025-09-26 找不到 2025/09/26)
            check_sql = f"SELECT 1 FROM stock_ohlcv_minute WHERE stock_id = ? AND date({time_col}) = date(?) LIMIT 1"
            if conn.execute(check_sql, (sid, d)).fetchone():
                # 分鐘表已經有資料了，代表這天不需要補，秒速跳過
                p_bar.progress(count / total)
                continue

            # 進入補洞流程
            p_text.warning(f"🔍 偵測到分鐘級缺口：{d} | {sid}...")
            try:
                df_tick = dl.taiwan_stock_tick(stock_id=sid, date=d)
                if df_tick is not None and not df_tick.empty:
                    # 資料加工 (Tick -> Minute OHLCV)
                    df_tick['date_time'] = pd.to_datetime(df_tick['date'] + ' ' + df_tick['Time'])
                    df_tick = df_tick.set_index('date_time')
                    
                    df_min = df_tick['deal_price'].resample('1min').ohlc()
                    df_min['volume'] = df_tick['volume'].resample('1min').sum()
                    df_min['active_buy_vol'] = df_tick[df_tick['TickType'] == 2]['volume'].resample('1min').sum()
                    df_min['active_sell_vol'] = df_tick[df_tick['TickType'] == 1]['volume'].resample('1min').sum()
                    
                    df_min = df_min.fillna(0).reset_index().rename(columns={'date_time': time_col})
                    df_min['stock_id'] = sid
                    
                    # ✅ 執行原子性寫入 (先刪後寫，確保完全更新且不觸發 UNIQUE 衝突)
                    with conn:
                        # 1. 覆蓋分鐘表
                        conn.execute(f"DELETE FROM stock_ohlcv_minute WHERE stock_id = ? AND date({time_col}) = date(?)", (sid, d))
                        df_min.to_sql("stock_ohlcv_minute", conn, if_exists="append", index=False, method="multi")
                        
                        # 2. 覆蓋日計表 (確保加總數值與分鐘表完全一致)
                        daily_flow = pd.DataFrame([{
                            "date": d, "stock_id": sid,
                            "active_buy_vol": int(df_min['active_buy_vol'].sum()),
                            "active_sell_vol": int(df_min['active_sell_vol'].sum())
                        }])
                        conn.execute("DELETE FROM stock_active_flow_daily WHERE stock_id = ? AND date(date) = date(?)", (sid, d))
                        daily_flow.to_sql("stock_active_flow_daily", conn, if_exists="append", index=False)
                    
                    p_text.success(f"🚀 {d} | {sid} 補洞完成")
                else:
                    p_text.info(f"⚠️ {d} | {sid} 無逐筆資料 (可能是休市)")

                p_bar.progress(count / total)
                time.sleep(cfg.get("ingest", {}).get("sleep_seconds", 0.3))
                
            except Exception as e:
                st.error(f"❌ {sid} {d} 失敗：{e}")

    conn.close()
    st.balloons()
    p_text.success("🎊 補洞案件執行完畢！")

def show_data_management():
    st.header("⚙️ 資料同步管理中心")
    cfg = database.load_config()
    
    task_type = st.radio(
        "請選擇要啟動的執行案件：",
        ["📅 每日 13 項指標 (原 Ingest Manager)", "⏱️ 分鐘與主動力度 (新 Ingest Minute)"],
        horizontal=True
    )
    
    st.divider()

    if task_type == "📅 每日 13 項指標 (原 Ingest Manager)":
        st.subheader("📋 案件：標準日線指標同步")
        if ingest_manager:
            if st.button("🔥 啟動全方位數據同步", use_container_width=True):
                log_container = st.container()
                with st.spinner("同步進行中..."):
                    try:
                        with log_container:
                            placeholder = st.empty()
                            importlib.reload(ingest_manager)
                            failed_items = ingest_manager.main(placeholder=placeholder)
                            if not failed_items:
                                st.success("✅ 所有日線指標同步成功！")
                            else:
                                st.warning(f"⚠️ 部分指標同步失敗：{', '.join(failed_items)}")
                    except Exception as e:
                        st.error(f"💥 程式執行中斷 (嚴重錯誤)：{e}")

    elif task_type == "⏱️ 分鐘與主動力度 (新 Ingest Minute)":
        st.subheader("📋 案件：精準分鐘級補洞同步")
        st.info(f"當前設定：從 **{cfg['ingest_minute']['start_date']}** 開始補齊分鐘資料。")
        if st.button("🚀 啟動分鐘補洞與主動流向運算", use_container_width=True):
            run_minute_task(cfg)