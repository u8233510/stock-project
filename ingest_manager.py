import time
from datetime import datetime, timedelta

import pandas as pd
from FinMind.data import DataLoader

import database


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


def get_missing_ranges(conn, table_name, stock_id, date_col, target_start):
    available_cols = database.get_table_columns(conn, table_name)
    if not available_cols or date_col not in available_cols:
        return [(datetime.strptime(target_start, "%Y-%m-%d").date(), datetime.now().date())]

    t_start = datetime.strptime(target_start, "%Y-%m-%d").date()
    t_end = datetime.now().date()

    try:
        sql = (
            f"SELECT MIN({date_col}), MAX({date_col}) "
            f"FROM {table_name} WHERE stock_id = ? AND {date_col} != ''"
        )
        res = conn.execute(sql, (stock_id,)).fetchone()
        db_min_str = str(res[0]) if res[0] else None
        db_max_str = str(res[1]) if res[1] else None
    except Exception:
        return [(t_start, t_end)]

    ranges = []
    if not db_min_str:
        ranges.append((t_start, t_end))
    else:
        db_min = datetime.strptime(db_min_str[:10], "%Y-%m-%d").date()
        db_max = datetime.strptime(db_max_str[:10], "%Y-%m-%d").date()
        if t_start < db_min:
            ranges.append((t_start, db_min - timedelta(days=1)))
        if t_end > db_max:
            ranges.append((db_max + timedelta(days=1), t_end))

    return ranges


def start_ingest(st_placeholder=None):
    failed_log = []
    cfg = database.load_config()
    api = DataLoader()
    api.login_by_token(api_token=cfg["finmind"]["api_token"])
    conn = database.get_db_connection(cfg)

    universe = cfg.get("universe", [])
    enabled = cfg.get("datasets", {}).get("enabled", [])
    t_start = cfg["ingest_master"]["start_date"]
    sleep_sec = float(cfg.get("ingest", {}).get("sleep_seconds", 0.3))

    d_map = {
        "ohlcv": ("TaiwanStockPrice", "stock_ohlcv_daily", "date"),
        "institutional": (
            "TaiwanStockInstitutionalInvestorsBuySell",
            "institutional_investors_daily",
            "date",
        ),
        "branch": ("TaiwanStockTradingDailyReport", "branch_price_daily", "date"),
        "per_pbr": ("TaiwanStockPER", "stock_per_pbr_daily", "date"),
        "margin_short": ("TaiwanStockMarginPurchaseShortSale", "margin_short_daily", "date"),
        "day_trading": ("TaiwanStockDayTrading", "stock_day_trading_daily", "date"),
        "holding_shares": (
            "TaiwanStockHoldingSharesPer",
            "stock_holding_shares_per_daily",
            "date",
        ),
        "securities_lending": (
            "TaiwanStockSecuritiesLending",
            "stock_securities_lending_daily",
            "date",
        ),
        "month_revenue": ("TaiwanStockMonthRevenue", "stock_month_revenue_monthly", "date"),
        "financial_statements": (
            "TaiwanStockFinancialStatements",
            "stock_financial_statements",
            "date",
        ),
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
            missing_ranges = get_missing_ranges(conn, table, sid, d_col, t_start)
            if not missing_ranges:
                continue

            for start, end in missing_ranges:
                s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                try:
                    if key == "branch":
                        for d_str in pd.date_range(start, end).strftime("%Y-%m-%d"):
                            df = api.taiwan_stock_trading_daily_report(stock_id=sid, date=d_str)
                            if df is not None and not df.empty:
                                upsert_data(conn, table, process_data(df, table, conn))
                            time.sleep(sleep_sec)
                    else:
                        df = api.get_data(
                            dataset=fm_api,
                            data_id=sid,
                            start_date=s_str,
                            end_date=e_str,
                        )
                        if df is not None and not df.empty:
                            upsert_data(conn, table, process_data(df, table, conn))
                    log(f"    🚀 [{key}] 同步成功: {s_str} ~ {e_str}")
                except Exception as e:
                    log(f"    ❌ [{key}] 失敗: {e}")
                    failed_log.append(f"{sid} {key}: {e}")

                time.sleep(sleep_sec)

    conn.close()
    return failed_log
