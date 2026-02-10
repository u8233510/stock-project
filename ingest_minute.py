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

def run_minute_task(cfg):
    """
    統一補洞規則：
    1) 從設定起始日掃描到 today，先排除週末/已知休市日。
    2) 若分鐘表該日缺資料，且 ingest log 無紀錄或 Failed，才打 API。
    3) ingest log 為 Success/NoTrade 則跳過。
    """
    dl = DataLoader()
    dl.login_by_token(api_token=cfg["finmind"]["api_token"])

    stock_list = cfg.get("universe", [])
    min_cfg = cfg.get("ingest_minute", {})
    start_date = min_cfg["start_date"]
    end_date = min_cfg.get("end_date") or datetime.now().strftime("%Y-%m-%d")
    rolling_recheck_days = int(
        min_cfg.get("rolling_recheck_days", cfg.get("ingest", {}).get("rolling_recheck_days", 1))
    )
    date_range = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

    p_bar = st.progress(0)
    p_text = st.empty()
    count = 0
    total = len(date_range) * len(stock_list)

    conn = database.get_db_connection(cfg)
    _ensure_data_ingest_log_table(conn)
    known_holidays = _get_known_holidays(conn)

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

    for d in date_range:
        d_obj = pd.to_datetime(d).date()
        if d_obj.weekday() >= 5 or d_obj in known_holidays:
            continue

        for stock in stock_list:
            count += 1
            sid = stock["stock_id"]

            today = datetime.now().date()
            force_recheck = rolling_recheck_days > 0 and d_obj >= today - timedelta(days=rolling_recheck_days - 1)

            status = _get_data_ingest_status(conn, sid, d, MINUTE_API_NAME)
            if not force_recheck and status in {"Success", "NoTrade"}:
                p_bar.progress(count / total)
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

                p_bar.progress(count / total)
                time.sleep(cfg.get("ingest", {}).get("sleep_seconds", 0.3))

            except Exception as e:
                st.error(f"❌ {sid} {d} 失敗：{e}")
                _write_data_ingest_log(conn, d, sid, MINUTE_API_NAME, 0, 0, "Failed")

    conn.close()
    st.balloons()
    p_text.success("🎊 補洞案件執行完畢！")
