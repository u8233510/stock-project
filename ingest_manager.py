import time
from datetime import datetime, timedelta

import pandas as pd
from FinMind.data import DataLoader

import database
from ingest_log_utils import ensure_data_ingest_log_table, get_data_ingest_status, write_data_ingest_log


def _ensure_data_ingest_log_table(conn):
    conn.execute(database.TABLE_REGISTRY["data_ingest_log"])
    conn.commit()


def _is_no_trade_marked(conn, stock_id, trade_date):
    sql = (
        "SELECT 1 FROM data_ingest_log "
        "WHERE stock_id = ? AND date = ? AND status = 'NoTrade' LIMIT 1"
    )
    return conn.execute(sql, (stock_id, trade_date)).fetchone() is not None


def _write_data_ingest_log(conn, trade_date, stock_id, api_count, db_count, status):
    conn.execute(
        """
        INSERT OR REPLACE INTO data_ingest_log(date, stock_id, api_count, db_count, status, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (trade_date, stock_id, int(api_count), int(db_count), status),
    )
    conn.commit()


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


def get_missing_ranges(conn, table_name, stock_id, date_col, target_start, rolling_recheck_days=1):
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

    # 只針對平日做補洞，避免每次都重查週末造成不必要 API 成本。
    all_dates = pd.date_range(start=t_start, end=t_end, freq="B").date
    missing_dates = [d for d in all_dates if d not in exists]

    # 每次同步都回頭重刷最近 N 天，避免「當天同步到一半中斷」造成資料不完整。
    recheck_days = max(int(rolling_recheck_days or 0), 0)
    if recheck_days > 0:
        recheck_start = max(t_start, t_end - timedelta(days=recheck_days - 1))
        rolling_dates = pd.date_range(start=recheck_start, end=t_end, freq="B").date
        missing_dates.extend(rolling_dates)

    return _merge_dates_to_ranges(missing_dates)


def start_ingest(st_placeholder=None):
    failed_log = []
    cfg = database.load_config()
    api = DataLoader()
    api.login_by_token(api_token=cfg["finmind"]["api_token"])
    conn = database.get_db_connection(cfg)
    _ensure_data_ingest_log_table(conn)

    universe = cfg.get("universe", [])
    enabled = cfg.get("datasets", {}).get("enabled", [])
    t_start = cfg["ingest_master"]["start_date"]
    sleep_sec = float(cfg.get("ingest", {}).get("sleep_seconds", 0.3))
    rolling_recheck_days = int(cfg.get("ingest", {}).get("rolling_recheck_days", 1))

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
            missing_ranges = get_missing_ranges(
                conn,
                table,
                sid,
                d_col,
                t_start,
                rolling_recheck_days=rolling_recheck_days,
            )
            if not missing_ranges:
                continue

            for start, end in fetch_ranges:
                s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                try:
                    if key == "branch":
                        for d_str in pd.date_range(start, end, freq="B").strftime("%Y-%m-%d"):
                            if _is_no_trade_marked(conn, sid, d_str):
                                continue
                            df = api.taiwan_stock_trading_daily_report(stock_id=sid, date=d_str)
                            if df is not None and not df.empty:
                                clean_df = process_data(df, table, conn)
                                upsert_data(conn, table, clean_df)
                                _write_data_ingest_log(conn, d_str, sid, len(df), len(clean_df), "Success")
                            else:
                                _write_data_ingest_log(conn, d_str, sid, 0, 0, "NoTrade")
                            time.sleep(sleep_sec)
                    else:
                        df = api.get_data(
                            dataset=fm_api,
                            data_id=sid,
                            start_date=s_str,
                            end_date=e_str,
                        )
                        if df is not None and not df.empty:
                            clean_df = process_data(df, table, conn)
                            upsert_data(conn, table, clean_df)
                            for d_str, grp in clean_df.groupby(clean_df[d_col].astype(str).str[:10]):
                                _write_data_ingest_log(conn, d_str, sid, len(grp), len(grp), "Success")
                    log(f"    🚀 [{key}] 同步成功: {s_str} ~ {e_str}")
                except Exception as e:
                    log(f"    ❌ [{key}] 失敗: {e}")
                    failed_log.append(f"{sid} {key}: {e}")
                    for d_str in pd.date_range(start, end, freq="B").strftime("%Y-%m-%d"):
                        write_data_ingest_log(conn, d_str, sid, key, 0, 0, "Failed")

                time.sleep(sleep_sec)

    conn.close()
    return failed_log
