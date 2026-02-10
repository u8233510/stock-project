import time
from datetime import datetime, timedelta

import pandas as pd
from FinMind.data import DataLoader
import streamlit as st

import database


MINUTE_API_NAME = "TaiwanStockTick"


def _ensure_data_ingest_log_table(conn):
    database.ensure_data_ingest_log_schema(conn)


def _get_data_ingest_status(conn, stock_id, trade_date, api_name=MINUTE_API_NAME):
    sql = "SELECT status FROM data_ingest_log WHERE stock_id = ? AND date = ? AND api_name = ? LIMIT 1"
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


def _exclude_weekends(dates):
    return [d for d in dates if d.weekday() < 5]


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


def _resolve_required_column(conn, table_name, keywords, default=None):
    cols = database.get_table_columns(conn, table_name)
    if default and default in cols:
        return default
    matched = database.match_column(cols, keywords)
    if matched:
        return matched
    return default


def _require_column(column_name, table_name, keywords):
    if not column_name:
        raise RuntimeError(f"找不到 {table_name} 欄位，關鍵字: {keywords}")
    return column_name


def _get_pending_dates_for_minute(conn, stock_id, target_start, target_end, api_name=MINUTE_API_NAME):
    t_start = datetime.strptime(target_start, "%Y-%m-%d").date()
    t_end = datetime.strptime(target_end, "%Y-%m-%d").date()
    today = datetime.now().date()

    candidate_dates = list(pd.date_range(start=t_start, end=t_end, freq="B").date)
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


def run_minute_task(cfg):
    """
    與每日同步規則一致：
    1) 依 stock_id + 區間產生候選日期。
    2) 先排除 ingest_log 已知 NoTrade 日期。
    3) 比對 data_ingest_log，僅保留 missing/failed。
    4) 逐日打 API，並將結果寫回 data_ingest_log。
    """
    dl = DataLoader()
    dl.login_by_token(api_token=cfg["finmind"]["api_token"])

    stock_list = cfg.get("universe", [])
    min_cfg = cfg.get("ingest_minute", {})
    start_date = min_cfg["start_date"]
    end_date = min_cfg.get("end_date") or datetime.now().strftime("%Y-%m-%d")
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    total = max(len(business_days) * max(len(stock_list), 1), 1)

    p_bar = st.progress(0)
    p_text = st.empty()
    count = 0

    conn = database.get_db_connection(cfg)
    _ensure_data_ingest_log_table(conn)

    time_col = _require_column(
        _resolve_required_column(conn, "stock_ohlcv_minute", ["date"], default="date_time"),
        "stock_ohlcv_minute",
        ["date"],
    )
    minute_stock_col = _require_column(
        _resolve_required_column(conn, "stock_ohlcv_minute", ["stock", "id"], default="stock_id"),
        "stock_ohlcv_minute",
        ["stock", "id"],
    )
    flow_date_col = _require_column(
        _resolve_required_column(conn, "stock_active_flow_daily", ["date"], default="date"),
        "stock_active_flow_daily",
        ["date"],
    )
    flow_stock_col = _require_column(
        _resolve_required_column(conn, "stock_active_flow_daily", ["stock", "id"], default="stock_id"),
        "stock_active_flow_daily",
        ["stock", "id"],
    )
    flow_buy_col = _require_column(
        _resolve_required_column(conn, "stock_active_flow_daily", ["active", "buy"], default="active_buy_vol"),
        "stock_active_flow_daily",
        ["active", "buy"],
    )
    flow_sell_col = _require_column(
        _resolve_required_column(conn, "stock_active_flow_daily", ["active", "sell"], default="active_sell_vol"),
        "stock_active_flow_daily",
        ["active", "sell"],
    )

    for stock in stock_list:
        sid = stock["stock_id"]
        pending_dates = _get_pending_dates_for_minute(
            conn,
            sid,
            start_date,
            end_date,
            api_name=MINUTE_API_NAME,
        )
        if not pending_dates:
            continue

        pending_ranges = _merge_dates_to_ranges(pending_dates)
        ranges_txt = ", ".join([f"{s.strftime('%Y-%m-%d')}~{e.strftime('%Y-%m-%d')}" for s, e in pending_ranges])
        p_text.info(f"📌 {sid} 分鐘資料待補區間: {ranges_txt}")

        for d in [x.strftime("%Y-%m-%d") for x in pending_dates]:
            count += 1
            status = _get_data_ingest_status(conn, sid, d, MINUTE_API_NAME)
            if status == "Success":
                p_bar.progress(min(count / total, 1.0))
                continue
            is_today = d == datetime.now().strftime("%Y-%m-%d")
            if (not is_today) and status == "NoTrade":
                p_bar.progress(min(count / total, 1.0))
                continue

            p_text.warning(f"🔍 偵測到分鐘級缺口：{d} | {sid}...")
            try:
                df_tick = dl.taiwan_stock_tick(stock_id=sid, date=d)
                if df_tick is not None and not df_tick.empty:
                    df_tick["date_time"] = pd.to_datetime(df_tick["date"] + " " + df_tick["Time"])
                    df_tick = df_tick.set_index("date_time")

                    df_min = df_tick["deal_price"].resample("1min").ohlc()
                    df_min["volume"] = df_tick["volume"].resample("1min").sum()
                    df_min["active_buy_vol"] = (
                        df_tick[df_tick["TickType"] == 2]["volume"].resample("1min").sum()
                    )
                    df_min["active_sell_vol"] = (
                        df_tick[df_tick["TickType"] == 1]["volume"].resample("1min").sum()
                    )

                    df_min = df_min.fillna(0).reset_index().rename(columns={"date_time": time_col})
                    df_min[minute_stock_col] = sid

                    with conn:
                        conn.execute(
                            f"DELETE FROM stock_ohlcv_minute WHERE {minute_stock_col} = ? AND date({time_col}) = date(?)",
                            (sid, d),
                        )
                        df_min.to_sql("stock_ohlcv_minute", conn, if_exists="append", index=False, method="multi")

                        daily_flow = pd.DataFrame(
                            [
                                {
                                    flow_date_col: d,
                                    flow_stock_col: sid,
                                    flow_buy_col: int(df_min["active_buy_vol"].sum()),
                                    flow_sell_col: int(df_min["active_sell_vol"].sum()),
                                }
                            ]
                        )
                        conn.execute(
                            f"DELETE FROM stock_active_flow_daily WHERE {flow_stock_col} = ? AND date({flow_date_col}) = date(?)",
                            (sid, d),
                        )
                        daily_flow.to_sql("stock_active_flow_daily", conn, if_exists="append", index=False)

                    p_text.success(f"🚀 {d} | {sid} 補洞完成")
                    _write_data_ingest_log(conn, d, sid, MINUTE_API_NAME, len(df_tick), len(df_min), "Success")
                else:
                    p_text.info(f"⚠️ {d} | {sid} 無逐筆資料 (可能是休市)")
                    _write_data_ingest_log(conn, d, sid, MINUTE_API_NAME, 0, 0, "NoTrade")

                p_bar.progress(min(count / total, 1.0))
                time.sleep(cfg.get("ingest", {}).get("sleep_seconds", 0.3))

            except Exception as e:
                st.error(f"❌ {sid} {d} 失敗：{e}")
                _write_data_ingest_log(conn, d, sid, MINUTE_API_NAME, 0, 0, "Failed")

    conn.close()
    st.balloons()
    p_text.success("🎊 補洞案件執行完畢！")
