import streamlit as st
import pandas as pd
import database
import math

TABLE_CHINESE_MAP = {
    "ingest_log": "資料同步紀錄", "branch_price_daily": "分點進出明細(依股票)", "branch_trader_daily_detail": "分點每日交易明細(依分點)",
    "securities_trader_info": "分點基本資料查詢",
    "stock_ohlcv_daily": "每日股價行情", "institutional_investors_daily": "三大法人買賣超",
    "margin_short_daily": "融資融券餘額", "stock_financial_statements": "財務報表 (原始長表)",
    "stock_day_trading_daily": "當沖交易統計", "stock_per_pbr_daily": "本益比/股淨比明細",
    "stock_holding_shares_per_daily": "大戶持股比例", "stock_securities_lending_daily": "借券成交明細",
    "stock_month_revenue_monthly": "每月營收統計", "stock_dividend": "歷年股利政策",
    "stock_market_value_daily": "每日市值/發行張數", "stock_industry_chain": "產業鏈資訊",
    "stock_active_flow_daily": "主動買賣力道彙整", "stock_ohlcv_minute": "分鐘級 K 線行情",
    "branch_weighted_cost": "分點加權成本 (分析用)"
}

def show_data_browser():
    st.header("📊 全資料表即時查詢")
    cfg = database.load_config()
    conn = database.get_db_connection(cfg)
    
    # 1. 基礎資料獲取
    db_tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
    table_display = {TABLE_CHINESE_MAP.get(t, t): t for t in db_tables}

    st.divider()

    # --- 2. 頂部過濾器 (6 欄位併排佈局) ---
    c1, c2, c3, c4, c5, c6 = st.columns([2, 1.8, 1.8, 1.5, 2, 1])
    
    with c1:
        date_range = st.date_input("日期區間篩選", value=[pd.to_datetime("2025-09-01"), pd.to_datetime("today")])
    
    with c2:
        selected_table_zh = st.selectbox("選擇資料類別", list(table_display.keys()))
        selected_table_en = table_display[selected_table_zh]
    
    # ✅ 關鍵優化：使用 database.py 內建功能自動偵測欄位
    cols = database.get_table_columns(conn, selected_table_en)
    stock_col = database.match_column(cols, ["stock_id"])
    branch_id_col = database.match_column(cols, ["securities_trader_id"])
    branch_name_col = database.match_column(cols, ["securities_trader"])
    time_col = (
        database.match_column(cols, ["date_time"])
        or database.match_column(cols, ["datetime"])
        or database.match_column(cols, ["timestamp"])
        or database.match_column(cols, ["date"])
    )

    use_stock_filter = bool(stock_col) and selected_table_en not in {"branch_trader_daily_detail", "securities_trader_info"}
    use_branch_filter = bool(branch_id_col) and selected_table_en in {"branch_trader_daily_detail", "securities_trader_info"}
    use_date_filter = bool(time_col) and selected_table_en != "securities_trader_info"

    target_sid = None
    target_branch_id = None

    with c3:
        if use_stock_filter:
            universe = cfg.get("universe", [])
            stock_options = {f"{s['stock_id']} {s['name']}": s['stock_id'] for s in universe}

            # 若 config 無股票池，退回從資料表抓 distinct stock_id
            if not stock_options:
                stock_rows = conn.execute(f"SELECT DISTINCT {stock_col} FROM {selected_table_en} ORDER BY {stock_col}").fetchall()
                stock_options = {row[0]: row[0] for row in stock_rows if row[0]}

            if stock_options:
                selected_stock_label = st.selectbox("選擇股票標的", list(stock_options.keys()))
                target_sid = stock_options[selected_stock_label]
            else:
                st.info("此資料表無可用股票代號可供篩選。")
        elif use_branch_filter:
            # 優先使用 securities_trader_info 當作分點中文名稱來源，
            # 避免明細表內名稱欄位缺漏或重複顯示代號。
            info_table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='securities_trader_info'"
            ).fetchone()

            branch_options = {}
            if info_table_exists:
                info_rows = conn.execute(
                    """
                    SELECT DISTINCT d.securities_trader_id, i.securities_trader
                    FROM branch_trader_daily_detail d
                    LEFT JOIN securities_trader_info i
                      ON i.securities_trader_id = d.securities_trader_id
                    ORDER BY d.securities_trader_id
                    """
                ).fetchall()
                branch_options = {
                    f"{row[0]} {str(row[1]).strip() if row[1] else ''}".strip(): row[0]
                    for row in info_rows
                    if row[0]
                }

            if not branch_options and branch_name_col:
                branch_rows = conn.execute(
                    f"SELECT DISTINCT {branch_id_col}, {branch_name_col} FROM {selected_table_en} ORDER BY {branch_id_col}"
                ).fetchall()
                branch_options = {
                    f"{row[0]} {str(row[1]).strip() if row[1] else ''}".strip(): row[0]
                    for row in branch_rows
                    if row[0]
                }

            if not branch_options:
                branch_rows = conn.execute(f"SELECT DISTINCT {branch_id_col} FROM {selected_table_en} ORDER BY {branch_id_col}").fetchall()
                branch_options = {row[0]: row[0] for row in branch_rows if row[0]}

            if branch_options:
                selected_branch_label = st.selectbox("選擇分點", list(branch_options.keys()))
                target_branch_id = branch_options[selected_branch_label]
            else:
                st.info("此資料表無可用分點可供篩選。")
        else:
            st.caption("此資料表不需股票/分點篩選")

    with c4:
        # 排序欄位預設對齊偵測到的時間欄位
        sort_col = st.selectbox("排序欄位", cols, index=cols.index(time_col) if time_col in cols else 0)
        
    with c5:
        sort_order = st.radio("排序方式", ["遞減 (DESC)", "遞增 (ASC)"], horizontal=True)
        order_sql = "DESC" if "遞減" in sort_order else "ASC"

    with c6:
        rows_per_page = st.selectbox("每頁筆數", [10, 20, 50, 100], index=2)

    # 3. 建立 SQL 查詢語句 (使用動態偵測的時間欄位)
    where_parts = []
    query_params = []

    if use_stock_filter and target_sid:
        where_parts.append(f"{stock_col} = ?")
        query_params.append(target_sid)

    if use_branch_filter and target_branch_id:
        where_parts.append(f"{branch_id_col} = ?")
        query_params.append(target_branch_id)

    if use_date_filter and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_day = pd.to_datetime(date_range[0]).date()
        end_day = pd.to_datetime(date_range[1]).date()

        # 分鐘級資料通常為 datetime 字串，改用整日邊界比對，
        # 可避免 date() 解析格式差異造成 2/10 這類資料被漏掉。
        if any(key in time_col.lower() for key in ["time", "datetime", "timestamp"]):
            start_dt = f"{start_day.isoformat()} 00:00:00"
            end_dt_exclusive = f"{(end_day + pd.Timedelta(days=1)).isoformat()} 00:00:00"
            where_parts.append(f"{time_col} >= ? AND {time_col} < ?")
            query_params.extend([start_dt, end_dt_exclusive])
        else:
            where_parts.append(f"{time_col} BETWEEN ? AND ?")
            query_params.extend([start_day.isoformat(), end_day.isoformat()])

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    
    count_query = f"SELECT COUNT(*) FROM {selected_table_en} {where_clause}"
    total_rows = conn.execute(count_query, query_params).fetchone()[0]
    total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1

    # 4. 資料顯示與分頁
    if total_rows > 0:
        page = st.number_input(f"目前共 {total_rows} 筆資料，請選擇頁碼 (共 {total_pages} 頁)", min_value=1, max_value=total_pages, value=1)
        offset = (page - 1) * rows_per_page
        
        query = f"SELECT * FROM {selected_table_en} {where_clause} ORDER BY {sort_col} {order_sql} LIMIT {rows_per_page} OFFSET {offset}"
        df = pd.read_sql(query, conn, params=query_params)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        csv = df.to_csv(index=False).encode('utf-8-sig')
        export_tag = target_sid or target_branch_id or selected_table_en
        st.download_button("📥 匯出當前頁面 CSV", csv, f"{export_tag}_data.csv", "text/csv")
    else:
        st.warning("ℹ️ 該篩選條件下無資料。")
    
    conn.close()
