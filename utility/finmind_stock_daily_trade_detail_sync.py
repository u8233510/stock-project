from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from FinMind.data import DataLoader

import database
from ingest_manager import get_pending_dates, _merge_dates_to_ranges

API_NAME = "TaiwanStockPrice"
REQUIRED_COLUMNS = [
    "date",
    "stock_id",
    "Trading_Volume",
    "Trading_money",
    "open",
    "max",
    "min",
    "close",
    "spread",
    "Trading_turnover",
]


@dataclass
class SyncResult:
    row_count: int
    status: str
    message: str = ""


def normalize_trade_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["stock_id"] = out["stock_id"].fillna("").astype(str)

    numeric_int_cols = ["Trading_Volume", "Trading_money", "Trading_turnover"]
    for c in numeric_int_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")

    numeric_float_cols = ["open", "max", "min", "close", "spread"]
    for c in numeric_float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[(out["stock_id"] != "") & out["date"].notna()]
    return out.drop_duplicates(subset=["date", "stock_id"], keep="last")


def upsert_trade_detail_df(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return

    payload = [
        (
            str(row.date),
            str(row.stock_id),
            int(row.Trading_Volume),
            int(row.Trading_money),
            None if pd.isna(row.open) else float(row.open),
            None if pd.isna(row.max) else float(row.max),
            None if pd.isna(row.min) else float(row.min),
            None if pd.isna(row.close) else float(row.close),
            None if pd.isna(row.spread) else float(row.spread),
            int(row.Trading_turnover),
        )
        for row in df.itertuples(index=False)
    ]

    conn.executemany(
        """
        INSERT OR REPLACE INTO stock_daily_trade_detail(
            date, stock_id, Trading_Volume, Trading_money,
            open, max, min, close, spread, Trading_turnover
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def write_sync_log(conn: sqlite3.Connection, trade_date: str, result: SyncResult):
    conn.execute(
        """
        INSERT OR REPLACE INTO stock_daily_trade_detail_sync_log(
            date, api_name, row_count, status, message, updated_at
        ) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (trade_date, API_NAME, int(result.row_count), str(result.status), str(result.message or "")),
    )


def write_data_ingest_log(conn: sqlite3.Connection, trade_date: str, result: SyncResult):
    conn.execute(
        """
        INSERT OR REPLACE INTO data_ingest_log(
            date, stock_id, api_name, api_count, db_count, status, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (
            trade_date,
            "__ALL__",
            API_NAME,
            int(result.row_count),
            int(result.row_count),
            str(result.status),
        ),
    )


def _with_retry(fn, max_retries: int, retry_sleep_sec: float):
    retries = max(int(max_retries or 0), 0)
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(max(float(retry_sleep_sec or 0), 0.0))


def sync_stock_daily_trade_detail(
    token: str,
    sqlite_path: Path,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    sleep_sec: float = 0.3,
    max_retries: int = 2,
    retry_sleep_sec: float = 1.0,
    retry_notrade_days: int = 14,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if not token:
        raise ValueError("Missing FinMind token")

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("🔐 登入 FinMind...")
    api = DataLoader()
    api.login_by_token(api_token=token)

    conn = sqlite3.connect(str(sqlite_path))
    try:
        database.ensure_stock_daily_trade_detail_schema(conn)
        database.ensure_data_ingest_log_schema(conn)

        pending_dates = get_pending_dates(
            conn,
            stock_id="__ALL__",
            api_name=API_NAME,
            target_start=start_date,
            target_end=end_date,
            check_freq="B",
            retry_notrade_days=retry_notrade_days,
        )
        if not pending_dates:
            if progress_callback:
                progress_callback("✅ 無待同步日期（已是最新）")
            return pd.DataFrame(columns=["date", "status", "row_count", "message"])

        ranges = _merge_dates_to_ranges(pending_dates)
        ranges_txt = ", ".join([f"{s.strftime('%Y-%m-%d')}~{e.strftime('%Y-%m-%d')}" for s, e in ranges])
        if progress_callback:
            progress_callback(f"📅 待同步區間: {ranges_txt}")

        summary_rows: list[dict] = []
        for day in pending_dates:
            day_str = day.strftime("%Y-%m-%d")
            if progress_callback:
                progress_callback(f"📡 同步 {day_str} ...")

            try:
                df = _with_retry(
                    lambda: api.get_data(
                        dataset=API_NAME,
                        start_date=day_str,
                        end_date=day_str,
                    ),
                    max_retries=max_retries,
                    retry_sleep_sec=retry_sleep_sec,
                )
                clean_df = normalize_trade_detail_df(df)

                if clean_df.empty:
                    result = SyncResult(row_count=0, status="NoTrade", message="FinMind API returned 0 rows")
                else:
                    upsert_trade_detail_df(conn, clean_df)
                    result = SyncResult(row_count=len(clean_df), status="Success")
                    clean_df.to_csv(raw_dir / f"stock_daily_trade_detail_{day_str}.csv", index=False, encoding="utf-8-sig")

                write_sync_log(conn, day_str, result)
                write_data_ingest_log(conn, day_str, result)
                conn.commit()
                summary_rows.append(
                    {
                        "date": day_str,
                        "status": result.status,
                        "row_count": int(result.row_count),
                        "message": result.message,
                    }
                )
            except Exception as e:
                err_msg = str(e)
                failed_result = SyncResult(row_count=0, status="Failed", message=err_msg)
                write_sync_log(conn, day_str, failed_result)
                write_data_ingest_log(conn, day_str, failed_result)
                conn.commit()
                summary_rows.append({"date": day_str, "status": "Failed", "row_count": 0, "message": err_msg})

            time.sleep(max(float(sleep_sec or 0), 0.0))

        return pd.DataFrame(summary_rows)
    finally:
        conn.close()
