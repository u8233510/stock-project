from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from FinMind.data import DataLoader

import database

API_NAME = "taiwan_stock_info_with_warrant"
REQUIRED_COLUMNS = ["industry_category", "stock_id", "stock_name", "type", "date"]


@dataclass
class SyncResult:
    row_count: int
    status: str
    message: str = ""


def normalize_stock_info_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[REQUIRED_COLUMNS].copy()
    out["industry_category"] = out["industry_category"].fillna("").astype(str)
    out["stock_id"] = out["stock_id"].fillna("").astype(str)
    out["stock_name"] = out["stock_name"].fillna("").astype(str)
    out["type"] = out["type"].fillna("").astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return out[(out["stock_id"] != "") & out["date"].notna()].drop_duplicates(subset=["date", "stock_id"], keep="last")


def upsert_stock_info_df(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return

    payload = [
        (
            str(row.date),
            str(row.stock_id),
            str(row.stock_name),
            str(row.type),
            str(row.industry_category),
        )
        for row in df.itertuples(index=False)
    ]

    conn.executemany(
        """
        INSERT OR REPLACE INTO stock_info(
            date, stock_id, stock_name, type, industry_category
        ) VALUES (?, ?, ?, ?, ?)
        """,
        payload,
    )


def write_stock_info_sync_log(conn: sqlite3.Connection, result: SyncResult, date_str: str):
    conn.execute(
        """
        INSERT OR REPLACE INTO stock_info_sync_log(
            date, api_name, row_count, status, message, updated_at
        ) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (date_str, API_NAME, int(result.row_count), str(result.status), str(result.message or "")),
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


def refresh_stock_info(
    token: str,
    sqlite_path: Path | None,
    raw_dir: Path,
    sleep_sec: float = 0.2,
    max_retries: int = 2,
    retry_sleep_sec: float = 1.0,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if not token:
        raise ValueError("Missing FinMind token")

    if progress_callback:
        progress_callback("🔐 登入 FinMind...")
    api = DataLoader()
    api.login_by_token(api_token=token)

    raw_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("📡 呼叫 FinMind 股票資訊 API...")
    df = _with_retry(
        lambda: api.taiwan_stock_info_with_warrant(),
        max_retries=max_retries,
        retry_sleep_sec=retry_sleep_sec,
    )

    time.sleep(max(float(sleep_sec or 0), 0.0))
    clean_df = normalize_stock_info_df(df)

    if not clean_df.empty:
        clean_df.to_csv(raw_dir / "stock_info_snapshot.csv", index=False, encoding="utf-8-sig")

    conn = None
    try:
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(sqlite_path))
            database.ensure_stock_info_sync_log_schema(conn)
            upsert_stock_info_df(conn, clean_df)

            sync_date = datetime.now().date().isoformat()
            status = "Success" if not clean_df.empty else "NoData"
            message = "" if not clean_df.empty else "FinMind API returned 0 rows"
            write_stock_info_sync_log(
                conn,
                SyncResult(row_count=len(clean_df), status=status, message=message),
                date_str=sync_date,
            )
            conn.commit()
    finally:
        if conn is not None:
            conn.close()

    if progress_callback:
        progress_callback(f"✅ 股票資訊同步完成，共 {len(clean_df)} 筆")
    return clean_df
