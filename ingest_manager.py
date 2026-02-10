import time
from datetime import datetime, timedelta

import pandas as pd
from FinMind.data import DataLoader

import database


def _ensure_data_ingest_log_table(conn):
    database.ensure_data_ingest_log_schema(conn)


def _get_data_ingest_status(conn, stock_id, trade_date, api_name):
    sql = (
        "SELECT status FROM data_ingest_log "
        "WHERE stock_id = ? AND date = ? AND api_name = ? LIMIT 1"
    )
    row = conn.execute(sql, (stock_id, trade_date, api_name)).fetchone()
    return row[0] if row else None


def _write_data_ingest_log(conn, trade_date, stock_id, api_name, api_count, db_count, status):
    conn.execute(
        """
        INSERT OR REPLACE INTO data_ingest_log(date, stock_id, api_name, api_count, db_count, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (trade_date, stock_id, api_name, int(api_count), int(db_count), status),
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


def _get_known_holidays(conn):
    sql = "SELECT DISTINCT date FROM data_ingest_log WHERE status = 'NoTrade'"
    rows = conn.execute(sql).fetchall()
    holidays = set()
    for row in rows:
        try:
            holidays.add(datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date())
        except Exception:
            continue
    return holidays


def get_missing_ranges(conn, table_name, stock_id, date_col, target_start, rolling_recheck_days=1):
    available_cols = database.get_table_columns(conn, table_name)
    t_start = datetime.strptime(target_start, "%Y-%m-%d").date()
    t_end = datetime.now().date()

    if not available_cols or date_col not in available_cols:
        return [(t_start, t_end)]

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
        return [(t_start, t_end)]

    business_dates = pd.date_range(start=t_start, end=t_end, freq="B").date
    holiday_dates = _get_known_holidays(conn)
    target_dates = [d for d in business_dates if d not in holiday_dates]
    missing_dates = [d for d in target_dates if d not in exists]

    recheck_days = max(int(rolling_recheck_days or 0), 0)
    if recheck_days > 0:
        recheck_start = max(t_start, t_end - timedelta(days=recheck_days - 1))
        rolling_dates = [
            d
            for d in pd.date_range(start=recheck_start, end=t_end, freq="B").date
            if d not in holiday_dates
        ]
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

            for start, end in missing_ranges:
                s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                try:
                    if key == "branch":
                        for d_str in pd.date_range(start, end, freq="B").strftime("%Y-%m-%d"):
                            status = _get_data_ingest_status(conn, sid, d_str, fm_api)
                            if status in {"Success", "NoTrade"}:
                                continue
                            df = api.taiwan_stock_trading_daily_report(stock_id=sid, date=d_str)
                            if df is not None and not df.empty:
                                clean_df = process_data(df, table, conn)
                                upsert_data(conn, table, clean_df)
                                _write_data_ingest_log(conn, d_str, sid, fm_api, len(df), len(clean_df), "Success")
                            else:
                                _write_data_ingest_log(conn, d_str, sid, fm_api, 0, 0, "NoTrade")
                            time.sleep(sleep_sec)
                    else:
                        # 逐日檢查 ingest log，只有無紀錄或 Failed 才送 API
                        for d_str in pd.date_range(start, end, freq="B").strftime("%Y-%m-%d"):
                            status = _get_data_ingest_status(conn, sid, d_str, fm_api)
                            if status in {"Success", "NoTrade"}:
                                continue

                            df = api.get_data(
                                dataset=fm_api,
                                data_id=sid,
                                start_date=d_str,
                                end_date=d_str,
                            )
                            if df is not None and not df.empty:
                                clean_df = process_data(df, table, conn)
                                upsert_data(conn, table, clean_df)
                                db_count = int((clean_df[d_col].astype(str).str[:10] == d_str).sum()) if d_col in clean_df.columns else len(clean_df)
                                _write_data_ingest_log(conn, d_str, sid, fm_api, len(df), db_count, "Success")
                            else:
                                _write_data_ingest_log(conn, d_str, sid, fm_api, 0, 0, "NoTrade")
                            time.sleep(sleep_sec)
                    log(f"    🚀 [{key}] 同步成功: {s_str} ~ {e_str}")
                except Exception as e:
                    log(f"    ❌ [{key}] 失敗: {e}")
                    failed_log.append(f"{sid} {key}: {e}")

                time.sleep(sleep_sec)

    conn.close()
    return failed_log
