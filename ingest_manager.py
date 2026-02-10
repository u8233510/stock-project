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


def _get_known_holidays(conn, stock_id=None, api_name=None):
    sql = "SELECT DISTINCT date FROM data_ingest_log WHERE status = 'NoTrade'"
    params = []
    if stock_id is not None:
        sql += " AND stock_id = ?"
        params.append(stock_id)
    if api_name is not None:
        sql += " AND api_name = ?"
        params.append(api_name)

    rows = conn.execute(sql, params).fetchall()
    holidays = set()
    for row in rows:
        try:
            holidays.add(datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date())
        except Exception:
            continue
    return holidays


def _resolve_column(conn, table_name, keywords, configured_col=None):
    cols = database.get_table_columns(conn, table_name)
    if not cols:
        return configured_col

    if configured_col and configured_col in cols:
        return configured_col

    matched = database.match_column(cols, keywords)
    return matched or configured_col


def _exclude_weekends(dates):
    return [d for d in dates if d.weekday() < 5]


def get_pending_dates(conn, stock_id, api_name, target_start, target_end=None, check_freq="B"):
    """依據同步區間 -> 排除已知休市 -> 比對 ingest_log，挑出 Missing/Failed 日期。"""
    today = datetime.now().date()
    t_start = datetime.strptime(target_start, "%Y-%m-%d").date()
    if target_end:
        t_end = datetime.strptime(target_end, "%Y-%m-%d").date()
    else:
        t_end = datetime.now().date()

    candidate_dates = list(pd.date_range(start=t_start, end=t_end, freq=check_freq).date)
    if check_freq == "B":
        candidate_dates = _exclude_weekends(candidate_dates)
    if not candidate_dates:
        return []

    holidays = _get_known_holidays(conn, stock_id=stock_id, api_name=api_name)
    candidate_dates = [d for d in candidate_dates if d == today or d not in holidays]
    if not candidate_dates:
        return []

    min_date = candidate_dates[0].strftime("%Y-%m-%d")
    max_date = candidate_dates[-1].strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT date, status
        FROM data_ingest_log
        WHERE stock_id = ?
          AND api_name = ?
          AND date >= ?
          AND date <= ?
        """,
        (stock_id, api_name, min_date, max_date),
    ).fetchall()

    status_map = {str(d)[:10]: s for d, s in rows if d}
    pending_dates = []
    for d in candidate_dates:
        d_str = d.strftime("%Y-%m-%d")
        status = status_map.get(d_str)
        if status == "Success":
            continue
        if d != today and status == "NoTrade":
            continue
        pending_dates.append(d)

    return pending_dates


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
    t_end = cfg.get("ingest_master", {}).get("end_date")
    sleep_sec = float(cfg.get("ingest", {}).get("sleep_seconds", 0.3))

    d_map = {
        "ohlcv": ("TaiwanStockPrice", "stock_ohlcv_daily", "date", "B"),
        "institutional": (
            "TaiwanStockInstitutionalInvestorsBuySell",
            "institutional_investors_daily",
            "date",
            "B",
        ),
        "branch": ("TaiwanStockTradingDailyReport", "branch_price_daily", "date", "B"),
        "per_pbr": ("TaiwanStockPER", "stock_per_pbr_daily", "date", "B"),
        "margin_short": ("TaiwanStockMarginPurchaseShortSale", "margin_short_daily", "date", "B"),
        "day_trading": ("TaiwanStockDayTrading", "stock_day_trading_daily", "date", "B"),
        "holding_shares": (
            "TaiwanStockHoldingSharesPer",
            "stock_holding_shares_per_daily",
            "date",
            "W-THU",
        ),
        "securities_lending": (
            "TaiwanStockSecuritiesLending",
            "stock_securities_lending_daily",
            "date",
            "B",
        ),
        "month_revenue": ("TaiwanStockMonthRevenue", "stock_month_revenue_monthly", "date", "MS"),
        "financial_statements": (
            "TaiwanStockFinancialStatements",
            "stock_financial_statements",
            "date",
            "QS",
        ),
        "dividend": ("TaiwanStockDividend", "stock_dividend", "date", "YS"),
        "market_value": ("TaiwanStockMarketValue", "stock_market_value_daily", "date", "B"),
        "industry_chain": ("TaiwanStockIndustryChain", "stock_industry_chain", "date", "MS"),
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

            fm_api, table, d_col, check_freq = d_map[key]
            resolved_date_col = _resolve_column(conn, table, ["date"], configured_col=d_col)

            pending_dates = get_pending_dates(
                conn,
                sid,
                fm_api,
                t_start,
                target_end=t_end,
                check_freq=check_freq,
            )
            if not pending_dates:
                continue

            pending_ranges = _merge_dates_to_ranges(pending_dates)
            ranges_txt = ", ".join(
                [f"{s.strftime('%Y-%m-%d')}~{e.strftime('%Y-%m-%d')}" for s, e in pending_ranges]
            )

            try:
                for d_str in [d.strftime("%Y-%m-%d") for d in pending_dates]:
                    status = _get_data_ingest_status(conn, sid, d_str, fm_api)
                    if status == "Success":
                        continue
                    is_today = d_str == datetime.now().strftime("%Y-%m-%d")
                    if (not is_today) and status == "NoTrade":
                        continue

                    if key == "branch":
                        df = api.taiwan_stock_trading_daily_report(stock_id=sid, date=d_str)
                    else:
                        df = api.get_data(
                            dataset=fm_api,
                            data_id=sid,
                            start_date=d_str,
                            end_date=d_str,
                        )

                    if df is not None and not df.empty:
                        clean_df = process_data(df, table, conn)
                        upsert_data(conn, table, clean_df)
                        if resolved_date_col in clean_df.columns:
                            db_count = int((clean_df[resolved_date_col].astype(str).str[:10] == d_str).sum())
                        else:
                            db_count = len(clean_df)
                        _write_data_ingest_log(conn, d_str, sid, fm_api, len(df), db_count, "Success")
                    else:
                        _write_data_ingest_log(conn, d_str, sid, fm_api, 0, 0, "NoTrade")

                    time.sleep(sleep_sec)

                log(f"    🚀 [{key}] 同步成功: {ranges_txt}")
            except Exception as e:
                log(f"    ❌ [{key}] 失敗: {e}")
                failed_log.append(f"{sid} {key}: {e}")

            time.sleep(sleep_sec)

    conn.close()
    return failed_log
