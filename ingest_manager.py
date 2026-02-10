import time
from datetime import datetime, timedelta

import pandas as pd
from FinMind.data import DataLoader

import database
from ingest_log_utils import ensure_data_ingest_log_table, get_data_ingest_status, write_data_ingest_log


def process_data(df, table, conn):
    if df is None or df.empty:
        return pd.DataFrame()

    db_cols = database.get_table_columns(conn, table)
    clean_to_real_db = {c.lower(): c for c in db_cols}
    final_rename = {
        df_c: clean_to_real_db[df_c.lower()]
        for df_c in df.columns
        if df_c.lower() in clean_to_real_db
    }
    df = df.rename(columns=final_rename)
    keep = [c for c in df.columns if c in db_cols]
    return df[keep].drop_duplicates()


def upsert_data(conn, table, df):
    if df.empty:
        return
    df = df.where(pd.notnull(df), None)
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, df.values.tolist())
    conn.commit()


def _merge_dates_to_ranges(dates):
    if not dates:
        return []

    sorted_dates = sorted(set(dates))
    ranges = []
    s = sorted_dates[0]
    prev = s

    for d in sorted_dates[1:]:
        if d == prev + timedelta(days=1):
            prev = d
            continue
        ranges.append((s, prev))
        s = d
        prev = d

    ranges.append((s, prev))
    return ranges


def get_missing_dates(conn, table_name, stock_id, date_col, target_start):
    available_cols = database.get_table_columns(conn, table_name)
    if not available_cols or date_col not in available_cols:
        return list(pd.date_range(start=target_start, end=datetime.now().date(), freq="B").date)

    t_start = datetime.strptime(target_start, "%Y-%m-%d").date()
    t_end = datetime.now().date()

    try:
        sql = (
            f"SELECT DISTINCT substr({date_col}, 1, 10) "
            f"FROM {table_name} "
            f"WHERE stock_id = ? AND {date_col} != ''"
        )
        exists = {
            datetime.strptime(str(r[0])[:10], "%Y-%m-%d").date()
            for r in conn.execute(sql, (stock_id,)).fetchall()
            if r and r[0]
        }
    except Exception:
        return list(pd.date_range(start=t_start, end=t_end, freq="B").date)

    all_dates = pd.date_range(start=t_start, end=t_end, freq="B").date
    return [d for d in all_dates if d not in exists]


def _filter_dates_by_log(conn, missing_dates, stock_id, api_name):
    fetch_dates = []
    for d in missing_dates:
        d_str = d.strftime("%Y-%m-%d")
        status = get_data_ingest_status(conn, stock_id, api_name, d_str)
        if status in {"Success", "NoTrade"}:
            continue
        fetch_dates.append(d)
    return fetch_dates


def start_ingest(st_placeholder=None):
    failed_log = []
    cfg = database.load_config()
    api = DataLoader()
    api.login_by_token(api_token=cfg["finmind"]["api_token"])
    conn = database.get_db_connection(cfg)
    ensure_data_ingest_log_table(conn)

    universe = cfg.get("universe", [])
    enabled = cfg.get("datasets", {}).get("enabled", [])
    t_start = cfg["ingest_master"]["start_date"]
    sleep_sec = float(cfg.get("ingest", {}).get("sleep_seconds", 0.3))

    d_map = {
        "ohlcv": ("TaiwanStockPrice", "stock_ohlcv_daily", "date"),
        "institutional": ("TaiwanStockInstitutionalInvestorsBuySell", "institutional_investors_daily", "date"),
        "branch": ("TaiwanStockTradingDailyReport", "branch_price_daily", "date"),
        "per_pbr": ("TaiwanStockPER", "stock_per_pbr_daily", "date"),
        "margin_short": ("TaiwanStockMarginPurchaseShortSale", "margin_short_daily", "date"),
        "day_trading": ("TaiwanStockDayTrading", "stock_day_trading_daily", "date"),
        "holding_shares": ("TaiwanStockHoldingSharesPer", "stock_holding_shares_per_daily", "date"),
        "securities_lending": ("TaiwanStockSecuritiesLending", "stock_securities_lending_daily", "date"),
        "month_revenue": ("TaiwanStockMonthRevenue", "stock_month_revenue_monthly", "date"),
        "financial_statements": ("TaiwanStockFinancialStatements", "stock_financial_statements", "date"),
        "dividend": ("TaiwanStockDividend", "stock_dividend", "date"),
        "market_value": ("TaiwanStockMarketValue", "stock_market_value_daily", "date"),
        "industry_chain": ("TaiwanStockIndustryChain", "stock_industry_chain", "date"),
    }

    def log(msg):
        if st_placeholder:
            st_placeholder.write(msg)
        print(msg)

    for stock in universe:
        sid = stock["stock_id"]
        log(f"📂 **同步標的: {sid} {stock['name']}**")

        for key in enabled:
            if key not in d_map:
                continue

            fm_api, table, d_col = d_map[key]
            missing_dates = get_missing_dates(conn, table, sid, d_col, t_start)
            fetch_dates = _filter_dates_by_log(conn, missing_dates, sid, key)
            fetch_ranges = _merge_dates_to_ranges(fetch_dates)

            if not fetch_ranges:
                continue

            for start, end in fetch_ranges:
                s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                try:
                    if key == "branch":
                        for d_str in pd.date_range(start, end, freq="B").strftime("%Y-%m-%d"):
                            df = api.taiwan_stock_trading_daily_report(stock_id=sid, date=d_str)
                            if df is not None and not df.empty:
                                clean_df = process_data(df, table, conn)
                                upsert_data(conn, table, clean_df)
                                write_data_ingest_log(conn, d_str, sid, key, len(df), len(clean_df), "Success")
                            else:
                                write_data_ingest_log(conn, d_str, sid, key, 0, 0, "NoTrade")
                            time.sleep(sleep_sec)
                    else:
                        req_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="B")]
                        df = api.get_data(dataset=fm_api, data_id=sid, start_date=s_str, end_date=e_str)
                        clean_df = process_data(df, table, conn) if df is not None and not df.empty else pd.DataFrame()

                        if not clean_df.empty:
                            upsert_data(conn, table, clean_df)

                        existing_by_date = {}
                        if not clean_df.empty and d_col in clean_df.columns:
                            date_series = clean_df[d_col].astype(str).str[:10]
                            for d_str, grp in clean_df.groupby(date_series):
                                existing_by_date[d_str] = len(grp)

                        for d_str in req_dates:
                            db_count = existing_by_date.get(d_str, 0)
                            status = "Success" if db_count > 0 else "NoTrade"
                            write_data_ingest_log(conn, d_str, sid, key, db_count, db_count, status)

                    log(f"    🚀 [{key}] 同步成功: {s_str} ~ {e_str}")
                except Exception as e:
                    log(f"    ❌ [{key}] 失敗: {e}")
                    failed_log.append(f"{sid} {key}: {e}")
                    for d_str in pd.date_range(start, end, freq="B").strftime("%Y-%m-%d"):
                        write_data_ingest_log(conn, d_str, sid, key, 0, 0, "Failed")

                time.sleep(sleep_sec)

    conn.close()
    return failed_log
