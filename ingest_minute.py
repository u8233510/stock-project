import time
from datetime import datetime, timedelta

import pandas as pd
from FinMind.data import DataLoader
import streamlit as st

import database
from ingest_log_utils import ensure_data_ingest_log_table, get_data_ingest_status, write_data_ingest_log


def _ensure_data_ingest_log_table(conn):
    conn.execute(database.TABLE_REGISTRY["data_ingest_log"])
    conn.commit()


def _get_data_ingest_status(conn, stock_id, trade_date):
    sql = "SELECT status FROM data_ingest_log WHERE stock_id = ? AND date = ? LIMIT 1"
    row = conn.execute(sql, (stock_id, trade_date)).fetchone()
    return row[0] if row else None


def _write_data_ingest_log(conn, trade_date, stock_id, api_count, db_count, status):
    conn.execute(
        """
        INSERT OR REPLACE INTO data_ingest_log(date, stock_id, api_count, db_count, status, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (trade_date, stock_id, int(api_count), int(db_count), status),
    )
    conn.commit()


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
    min_cols = database.get_table_columns(conn, "stock_ohlcv_minute")
    time_col = database.match_column(min_cols, ["date"]) or "date_time"

    api_name = "minute"

    for d in date_range:
        if pd.to_datetime(d).weekday() >= 5:
            continue

        for stock in stock_list:
            count += 1
            sid = stock["stock_id"]

            today = datetime.now().date()
            d_obj = pd.to_datetime(d).date()
            force_recheck = rolling_recheck_days > 0 and d_obj >= today - timedelta(days=rolling_recheck_days - 1)

            status = _get_data_ingest_status(conn, sid, d)
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
                    df_min["stock_id"] = sid

                    with conn:
                        conn.execute(
                            f"DELETE FROM stock_ohlcv_minute WHERE stock_id = ? AND date({time_col}) = date(?)",
                            (sid, d),
                        )
                        df_min.to_sql("stock_ohlcv_minute", conn, if_exists="append", index=False, method="multi")

                        daily_flow = pd.DataFrame(
                            [
                                {
                                    "date": d,
                                    "stock_id": sid,
                                    "active_buy_vol": int(df_min["active_buy_vol"].sum()),
                                    "active_sell_vol": int(df_min["active_sell_vol"].sum()),
                                }
                            ]
                        )
                        conn.execute(
                            "DELETE FROM stock_active_flow_daily WHERE stock_id = ? AND date(date) = date(?)",
                            (sid, d),
                        )
                        daily_flow.to_sql("stock_active_flow_daily", conn, if_exists="append", index=False)

                    p_text.success(f"🚀 {d} | {sid} 補洞完成")
                    _write_data_ingest_log(conn, d, sid, len(df_tick), len(df_min), "Success")
                else:
                    p_text.info(f"⚠️ {d} | {sid} 無逐筆資料 (可能是休市)")
                    _write_data_ingest_log(conn, d, sid, 0, 0, "NoTrade")

                p_bar.progress(count / total)
                time.sleep(cfg.get("ingest", {}).get("sleep_seconds", 0.3))

            except Exception as e:
                st.error(f"❌ {sid} {d} 失敗：{e}")
                _write_data_ingest_log(conn, d, sid, 0, 0, "Failed")

    conn.close()
    st.balloons()
    p_text.success("🎊 補洞案件執行完畢！")
