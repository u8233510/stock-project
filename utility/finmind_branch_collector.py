"""Download FinMind branch trading detail in incremental mode.

Use this utility when API is keyed by (securities_trader_id, date).
It can be called either from CLI or Streamlit UI.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from FinMind.data import DataLoader

import database

REQUIRED_COLUMNS = [
    "securities_trader",
    "price",
    "buy",
    "sell",
    "securities_trader_id",
    "stock_id",
    "date",
]
API_NAME = "taiwan_stock_trading_daily_report_by_trader"


@dataclass
class FetchStats:
    branch_id: str
    trade_date: str
    rows: int
    status: str
    message: str = ""


def daterange(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    cursor = start
    while cursor <= end:
        yield cursor.isoformat()
        cursor += timedelta(days=1)


def normalize_branch_df(df: pd.DataFrame, branch_id: str, trade_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[REQUIRED_COLUMNS].copy()
    out["securities_trader_id"] = out["securities_trader_id"].fillna(branch_id).astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["date"] = out["date"].fillna(trade_date)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["buy"] = pd.to_numeric(out["buy"], errors="coerce").fillna(0).astype(int)
    out["sell"] = pd.to_numeric(out["sell"], errors="coerce").fillna(0).astype(int)
    out["stock_id"] = out["stock_id"].astype(str)
    out["securities_trader"] = out["securities_trader"].fillna("").astype(str)
    return out.dropna(subset=["price", "stock_id"]).drop_duplicates()


def fetch_branch_day(api: DataLoader, branch_id: str, trade_date: str) -> pd.DataFrame:
    return api.taiwan_stock_trading_daily_report(
        securities_trader_id=str(branch_id),
        date=str(trade_date),
    )


def fetch_trader_info(api: DataLoader, branch_id: str) -> pd.DataFrame:
    """Fetch trader metadata for validating branch_id and storing reference data."""
    return api.taiwan_securities_trader_info(securities_trader_id=str(branch_id))


def normalize_trader_info_df(df: pd.DataFrame, branch_id: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["securities_trader_id", "securities_trader", "date", "address", "phone"])

    out = df.copy()
    for col in ["securities_trader_id", "securities_trader", "date", "address", "phone"]:
        if col not in out.columns:
            out[col] = None

    out = out[["securities_trader_id", "securities_trader", "date", "address", "phone"]].copy()
    out["securities_trader_id"] = out["securities_trader_id"].fillna(branch_id).astype(str)
    out["securities_trader"] = out["securities_trader"].fillna("").astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["address"] = out["address"].fillna("").astype(str)
    out["phone"] = out["phone"].fillna("").astype(str)
    return out.drop_duplicates(subset=["securities_trader_id"], keep="last")


def ensure_branch_tables(conn: sqlite3.Connection):
    conn.execute(database.TABLE_REGISTRY["branch_trader_daily_detail"])
    conn.execute(database.TABLE_REGISTRY["securities_trader_info"])
    for idx in database.INDEX_REGISTRY.get("branch_trader_daily_detail", []):
        conn.execute(idx)
    for idx in database.INDEX_REGISTRY.get("securities_trader_info", []):
        conn.execute(idx)
    database.ensure_branch_sync_log_schema(conn)
    conn.commit()


def upsert_branch_detail_df(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return
    payload = [
        (
            str(row.date),
            str(row.stock_id),
            str(row.securities_trader_id),
            str(row.securities_trader),
            float(row.price),
            int(row.buy),
            int(row.sell),
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO branch_trader_daily_detail(
            date, stock_id, securities_trader_id, securities_trader, price, buy, sell
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def upsert_trader_info_df(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return
    payload = [
        (
            str(row.securities_trader_id),
            str(row.securities_trader),
            str(row.date) if pd.notna(row.date) else None,
            str(row.address),
            str(row.phone),
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO securities_trader_info(
            securities_trader_id, securities_trader, date, address, phone
        ) VALUES (?, ?, ?, ?, ?)
        """,
        payload,
    )


def write_branch_sync_log(conn: sqlite3.Connection, stat: FetchStats):
    conn.execute(
        """
        INSERT OR REPLACE INTO branch_sync_log(
            date, securities_trader_id, api_name, row_count, status, message, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """,
        (
            str(stat.trade_date),
            str(stat.branch_id),
            API_NAME,
            int(stat.rows),
            str(stat.status),
            str(stat.message or ""),
        ),
    )


def run_collection(
    token: str,
    branch_ids: list[str],
    start_date: str,
    end_date: str,
    raw_dir: Path,
    sqlite_path: Path | None,
    sleep_sec: float,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    api = DataLoader()
    api.login_by_token(api_token=token)

    conn = sqlite3.connect(sqlite_path) if sqlite_path else None
    raw_dir.mkdir(parents=True, exist_ok=True)

    stats: list[FetchStats] = []
    total_jobs = len(branch_ids) * sum(1 for _ in daterange(start_date, end_date))
    done_jobs = 0
    valid_branch_ids: set[str] = set()
    invalid_branch_reason: dict[str, str] = {}

    try:
        if conn is not None:
            ensure_branch_tables(conn)

        for branch_id in branch_ids:
            try:
                info_df = fetch_trader_info(api, branch_id)
                clean_info_df = normalize_trader_info_df(info_df, branch_id=branch_id)
                if not clean_info_df.empty:
                    valid_branch_ids.add(branch_id)
                    if conn is not None:
                        upsert_trader_info_df(conn, clean_info_df)
                        conn.commit()
                else:
                    invalid_branch_reason[branch_id] = (
                        "查無分點基本資料，需先確認分點代碼存在，才能透過 API 抓交易明細。"
                    )
            except Exception as exc:  # noqa: BLE001
                invalid_branch_reason[branch_id] = f"驗證分點代碼失敗: {exc}"

        for trade_date in daterange(start_date, end_date):
            for branch_id in branch_ids:
                msg = f"[{done_jobs + 1}/{total_jobs}] branch={branch_id}, date={trade_date}"
                if progress_callback:
                    progress_callback(msg)

                if branch_id not in valid_branch_ids:
                    stat = FetchStats(
                        branch_id,
                        trade_date,
                        0,
                        "error",
                        invalid_branch_reason.get(branch_id, "分點代碼驗證未通過"),
                    )
                    if conn is not None:
                        write_branch_sync_log(conn, stat)
                        conn.commit()
                    stats.append(stat)
                    done_jobs += 1
                    continue

                try:
                    raw_df = fetch_branch_day(api, branch_id=branch_id, trade_date=trade_date)
                    clean_df = normalize_branch_df(raw_df, branch_id=branch_id, trade_date=trade_date)

                    raw_file = raw_dir / f"{trade_date}_{branch_id}.csv"
                    clean_df.to_csv(raw_file, index=False, encoding="utf-8-sig")

                    stat = FetchStats(branch_id, trade_date, len(clean_df), "ok")
                    if conn is not None:
                        upsert_branch_detail_df(conn, clean_df)
                        write_branch_sync_log(conn, stat)
                        conn.commit()
                except Exception as exc:  # noqa: BLE001
                    stat = FetchStats(branch_id, trade_date, 0, "error", str(exc))
                    if conn is not None:
                        write_branch_sync_log(conn, stat)
                        conn.commit()
                stats.append(stat)
                done_jobs += 1
                time.sleep(max(sleep_sec, 0.0))
    finally:
        if conn is not None:
            conn.close()

    stat_df = pd.DataFrame([s.__dict__ for s in stats])
    stat_df.to_csv(raw_dir / "fetch_log.csv", index=False, encoding="utf-8-sig")

    ok = int((stat_df["status"] == "ok").sum()) if not stat_df.empty else 0
    err = int((stat_df["status"] == "error").sum()) if not stat_df.empty else 0
    print(f"Done. success={ok}, error={err}, log={raw_dir / 'fetch_log.csv'}")
    return stat_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FinMind branch daily report by branch/date.")
    parser.add_argument("--token", required=True, help="FinMind sponsor token")
    parser.add_argument(
        "--branch-ids",
        required=True,
        help="Comma-separated branch ids, e.g. 1102,1160,7000",
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--raw-dir", default="data/branch_raw", help="Folder for raw csv")
    parser.add_argument(
        "--sqlite-path",
        default="data/stock.db",
        help="Optional sqlite path for upsert (use empty string to disable)",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep seconds between API calls")
    return parser.parse_args()


def main():
    args = parse_args()
    branch_ids = [x.strip() for x in args.branch_ids.split(",") if x.strip()]
    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None
    run_collection(
        token=args.token,
        branch_ids=branch_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_dir=Path(args.raw_dir),
        sqlite_path=sqlite_path,
        sleep_sec=args.sleep_sec,
    )


if __name__ == "__main__":
    main()
