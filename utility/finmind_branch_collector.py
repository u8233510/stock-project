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


def _get_known_holidays(conn: sqlite3.Connection, branch_id: str | None = None) -> set[datetime.date]:
    sql = "SELECT DISTINCT date FROM branch_sync_log WHERE status = 'NoTrade'"
    params: list[str] = []
    if branch_id is not None:
        sql += " AND securities_trader_id = ?"
        params.append(str(branch_id))

    rows = conn.execute(sql, params).fetchall()
    holidays: set[datetime.date] = set()
    for row in rows:
        try:
            holidays.add(datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date())
        except Exception:
            continue
    return holidays


def _get_pending_dates_for_branch(
    conn: sqlite3.Connection,
    branch_id: str,
    start_date: str,
    end_date: str,
    retry_notrade_days: int,
) -> list[datetime.date]:
    today = datetime.now().date()
    t_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    t_end = datetime.strptime(end_date, "%Y-%m-%d").date()

    candidate_dates = list(pd.date_range(start=t_start, end=t_end, freq="B").date)
    if not candidate_dates:
        return []

    retry_cutoff = today - timedelta(days=max(int(retry_notrade_days or 0), 0))
    holidays = _get_known_holidays(conn, branch_id=branch_id)
    candidate_dates = [
        d
        for d in candidate_dates
        if d == today or d >= retry_cutoff or d not in holidays
    ]
    if not candidate_dates:
        return []

    min_date = candidate_dates[0].strftime("%Y-%m-%d")
    max_date = candidate_dates[-1].strftime("%Y-%m-%d")
    rows = conn.execute(
        """
        SELECT date, status
        FROM branch_sync_log
        WHERE securities_trader_id = ?
          AND api_name = ?
          AND date >= ?
          AND date <= ?
        """,
        (str(branch_id), API_NAME, min_date, max_date),
    ).fetchall()

    status_map = {str(d)[:10]: s for d, s in rows if d}
    pending_dates: list[datetime.date] = []
    for d in candidate_dates:
        d_str = d.strftime("%Y-%m-%d")
        status = status_map.get(d_str)
        if status == "Success":
            continue
        if d != today and status == "NoTrade" and d < retry_cutoff:
            continue
        pending_dates.append(d)
    return pending_dates


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


def fetch_all_trader_info(api: DataLoader) -> pd.DataFrame:
    """Fetch full trader list from FinMind for bootstrap/manual refresh."""
    return api.taiwan_securities_trader_info()


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


def _retry_api_call(func: Callable[[], pd.DataFrame], max_retries: int, retry_sleep_sec: float) -> pd.DataFrame:
    attempts = max(int(max_retries or 0), 0) + 1
    last_exc: Exception | None = None
    for idx in range(attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if idx >= attempts - 1:
                break
            time.sleep(max(float(retry_sleep_sec or 0.0), 0.0))
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def refresh_trader_info(
    token: str,
    sqlite_path: Path | None,
    raw_dir: Path,
    branch_ids: list[str] | None = None,
    sleep_sec: float = 0.2,
    max_retries: int = 2,
    retry_sleep_sec: float = 1.0,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Manual task: download branch master data into DB/CSV."""
    api = DataLoader()
    api.login_by_token(api_token=token)
    raw_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path) if sqlite_path else None
    try:
        if conn is not None:
            ensure_branch_tables(conn)

        if branch_ids:
            info_frames = []
            total = len(branch_ids)
            for idx, branch_id in enumerate(branch_ids, start=1):
                if progress_callback:
                    progress_callback(f"[{idx}/{total}] 下載分點基本資料: {branch_id}")
                df = _retry_api_call(
                    lambda b=branch_id: fetch_trader_info(api, b),
                    max_retries=max_retries,
                    retry_sleep_sec=retry_sleep_sec,
                )
                clean_df = normalize_trader_info_df(df, branch_id=branch_id)
                if not clean_df.empty:
                    info_frames.append(clean_df)
                time.sleep(max(float(sleep_sec or 0.0), 0.0))
            all_info_df = pd.concat(info_frames, ignore_index=True) if info_frames else pd.DataFrame()
        else:
            if progress_callback:
                progress_callback("下載全部分點基本資料中...")
            all_info_df = _retry_api_call(
                lambda: fetch_all_trader_info(api),
                max_retries=max_retries,
                retry_sleep_sec=retry_sleep_sec,
            )
            all_info_df = normalize_trader_info_df(all_info_df, branch_id="")

        if conn is not None and not all_info_df.empty:
            upsert_trader_info_df(conn, all_info_df)
            conn.commit()

        if not all_info_df.empty:
            all_info_df.to_csv(raw_dir / "trader_info_snapshot.csv", index=False, encoding="utf-8-sig")
        return all_info_df
    finally:
        if conn is not None:
            conn.close()


def run_collection(
    token: str,
    branch_ids: list[str],
    start_date: str,
    end_date: str,
    raw_dir: Path,
    sqlite_path: Path | None,
    sleep_sec: float,
    retry_notrade_days: int,
    refresh_trader_info: bool,
    max_retries: int = 2,
    retry_sleep_sec: float = 1.0,
    commit_interval: int = 100,
    write_raw_csv: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    api = DataLoader()
    api.login_by_token(api_token=token)

    conn = sqlite3.connect(sqlite_path) if sqlite_path else None
    if conn is not None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    if write_raw_csv:
        raw_dir.mkdir(parents=True, exist_ok=True)

    stats: list[FetchStats] = []
    total_jobs = len(branch_ids) * sum(1 for _ in daterange(start_date, end_date))
    done_jobs = 0
    valid_branch_ids: set[str] = set()
    invalid_branch_reason: dict[str, str] = {}
    pending_commit_ops = 0
    commit_interval = max(1, int(commit_interval or 1))

    def _flush_if_needed(force: bool = False):
        nonlocal pending_commit_ops
        if conn is None:
            return
        if force or pending_commit_ops >= commit_interval:
            conn.commit()
            pending_commit_ops = 0

    try:
        if conn is not None:
            ensure_branch_tables(conn)

        if conn is not None and refresh_trader_info:
            conn.execute("DELETE FROM securities_trader_info")
            pending_commit_ops += 1
            _flush_if_needed(force=True)

        if not refresh_trader_info:
            # 正常流程下 branch_ids 來自 securities_trader_info，逐筆驗證會造成 990+ 次 API 呼叫，
            # 使用者體感會像「卡住」在 0 筆很久。此情境直接信任既有分點清單，
            # 由後續每筆分點明細 API 回應決定 Success/Failed。
            valid_branch_ids = {str(branch_id) for branch_id in branch_ids}
        else:
            if progress_callback:
                progress_callback("正在重建分點基本資料（一次性抓取全部分點）...")

            try:
                all_info_df = _retry_api_call(
                    lambda: fetch_all_trader_info(api),
                    max_retries=max_retries,
                    retry_sleep_sec=retry_sleep_sec,
                )
                if all_info_df is None or all_info_df.empty:
                    invalid_reason = "FinMind 未回傳分點基本資料。"
                    for branch_id in branch_ids:
                        invalid_branch_reason[str(branch_id)] = invalid_reason
                else:
                    info_df = all_info_df.copy()
                    info_df["securities_trader_id"] = info_df["securities_trader_id"].astype(str)
                    available_ids = set(info_df["securities_trader_id"].dropna().astype(str))

                    valid_branch_ids = {str(branch_id) for branch_id in branch_ids if str(branch_id) in available_ids}
                    for branch_id in branch_ids:
                        branch_id = str(branch_id)
                        if branch_id not in valid_branch_ids:
                            invalid_branch_reason[branch_id] = (
                                "查無分點基本資料，需先確認分點代碼存在，才能透過 API 抓交易明細。"
                            )

                    if conn is not None and valid_branch_ids:
                        clean_info_df = normalize_trader_info_df(
                            info_df[info_df["securities_trader_id"].isin(valid_branch_ids)],
                            branch_id="",
                        )
                        upsert_trader_info_df(conn, clean_info_df)
                        pending_commit_ops += 1
                        _flush_if_needed(force=True)
            except Exception as exc:  # noqa: BLE001
                for branch_id in branch_ids:
                    invalid_branch_reason[str(branch_id)] = f"驗證分點代碼失敗: {exc}"

        pending_by_branch: dict[str, set[str]] = {}
        if conn is not None:
            if progress_callback:
                progress_callback("正在計算待同步日期...")
            for idx, branch_id in enumerate(branch_ids, start=1):
                pending_dates = _get_pending_dates_for_branch(
                    conn,
                    branch_id=branch_id,
                    start_date=start_date,
                    end_date=end_date,
                    retry_notrade_days=retry_notrade_days,
                )
                pending_by_branch[branch_id] = {d.isoformat() for d in pending_dates}
                if progress_callback and (idx == len(branch_ids) or idx % 100 == 0):
                    progress_callback(f"待同步日期計算中：{idx}/{len(branch_ids)} 分點")

            total_jobs = sum(len(v) for v in pending_by_branch.values())

        if total_jobs == 0:
            if progress_callback:
                progress_callback("沒有需要同步的新資料（全部已是最新或在免重試期間）。")
        
        for trade_date in daterange(start_date, end_date):
            for branch_id in branch_ids:
                if conn is not None and trade_date not in pending_by_branch.get(branch_id, set()):
                    continue

                msg = f"[{done_jobs + 1}/{max(total_jobs, 1)}] branch={branch_id}, date={trade_date}"
                if progress_callback:
                    progress_callback(msg)

                if branch_id not in valid_branch_ids:
                    stat = FetchStats(
                        branch_id,
                        trade_date,
                        0,
                        "Failed",
                        invalid_branch_reason.get(branch_id, "分點代碼驗證未通過"),
                    )
                    if conn is not None:
                        write_branch_sync_log(conn, stat)
                        pending_commit_ops += 1
                        _flush_if_needed()
                    stats.append(stat)
                    done_jobs += 1
                    continue

                try:
                    raw_df = _retry_api_call(
                        lambda b=branch_id, d=trade_date: fetch_branch_day(api, branch_id=b, trade_date=d),
                        max_retries=max_retries,
                        retry_sleep_sec=retry_sleep_sec,
                    )
                    clean_df = normalize_branch_df(raw_df, branch_id=branch_id, trade_date=trade_date)

                    if write_raw_csv:
                        raw_file = raw_dir / f"{trade_date}_{branch_id}.csv"
                        clean_df.to_csv(raw_file, index=False, encoding="utf-8-sig")

                    if clean_df.empty:
                        stat = FetchStats(branch_id, trade_date, 0, "NoTrade")
                    else:
                        stat = FetchStats(branch_id, trade_date, len(clean_df), "Success")
                    if conn is not None:
                        upsert_branch_detail_df(conn, clean_df)
                        write_branch_sync_log(conn, stat)
                        pending_commit_ops += 1
                        _flush_if_needed()
                except Exception as exc:  # noqa: BLE001
                    stat = FetchStats(branch_id, trade_date, 0, "Failed", str(exc))
                    if conn is not None:
                        write_branch_sync_log(conn, stat)
                        pending_commit_ops += 1
                        _flush_if_needed()
                stats.append(stat)
                done_jobs += 1
                time.sleep(max(sleep_sec, 0.0))
    finally:
        if conn is not None:
            _flush_if_needed(force=True)
            conn.close()

    stat_df = pd.DataFrame([s.__dict__ for s in stats])
    if write_raw_csv:
        raw_dir.mkdir(parents=True, exist_ok=True)
        stat_df.to_csv(raw_dir / "fetch_log.csv", index=False, encoding="utf-8-sig")

    ok = int((stat_df["status"] == "Success").sum()) if not stat_df.empty else 0
    err = int((stat_df["status"] == "Failed").sum()) if not stat_df.empty else 0
    log_msg = str(raw_dir / "fetch_log.csv") if write_raw_csv else "(skip raw csv logging)"
    print(f"Done. success={ok}, error={err}, log={log_msg}")
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
    parser.add_argument(
        "--retry-notrade-days",
        type=int,
        default=14,
        help="NoTrade 幾天後才視為穩定休市，不再重試",
    )
    parser.add_argument(
        "--refresh-trader-info",
        action="store_true",
        help="執行前清空 securities_trader_info，並重抓分點基本資料",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="API 失敗重試次數")
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0, help="API 重試等待秒數")
    parser.add_argument("--commit-interval", type=int, default=100, help="每 N 筆寫入才 commit 一次")
    parser.add_argument(
        "--write-raw-csv",
        action="store_true",
        help="將每筆分點-日期結果與 fetch_log 輸出到 raw-dir（預設關閉以提升同步速度）",
    )
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
        retry_notrade_days=args.retry_notrade_days,
        refresh_trader_info=args.refresh_trader_info,
        max_retries=args.max_retries,
        retry_sleep_sec=args.retry_sleep_sec,
        commit_interval=args.commit_interval,
        write_raw_csv=args.write_raw_csv,
    )


if __name__ == "__main__":
    main()
