#!/usr/bin/env python3
"""區間分點買賣超偵測獨立程式。

使用方式:
    python branch_net_detector_cli.py --start 2024-01-01 --end 2024-01-31
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class TraderAgg:
    buy_shares: int = 0
    sell_shares: int = 0
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    buy_days: int = 0
    sell_days: int = 0
    buy_trade_count: int = 0
    sell_trade_count: int = 0

    @property
    def net_shares(self) -> int:
        return self.buy_shares - self.sell_shares

    @property
    def avg_buy_price(self) -> float:
        return (self.buy_amount / self.buy_shares) if self.buy_shares else 0.0

    @property
    def avg_sell_price(self) -> float:
        return (self.sell_amount / self.sell_shares) if self.sell_shares else 0.0


@dataclass
class StockAgg:
    buy_shares: int = 0
    sell_shares: int = 0
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    total_trade_count: int = 0

    @property
    def net_shares(self) -> int:
        return self.buy_shares - self.sell_shares

    @property
    def avg_buy_price(self) -> float:
        return (self.buy_amount / self.buy_shares) if self.buy_shares else 0.0

    @property
    def avg_sell_price(self) -> float:
        return (self.sell_amount / self.sell_shares) if self.sell_shares else 0.0


def normalize_stock_id(value: object) -> str:
    stock_id = str(value or "").strip()
    if stock_id.isdigit() and len(stock_id) < 4:
        return stock_id.zfill(4)
    return stock_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="計算區間內所有股票分點買賣統計")
    parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="結束日期 YYYY-MM-DD")
    parser.add_argument(
        "--db-path",
        default="data/stock.db",
        help="SQLite 資料庫路徑，預設 data/stock.db",
    )
    parser.add_argument(
        "--output",
        default="output/branch_interval_summary.csv",
        help="輸出 CSV 路徑，預設 output/branch_interval_summary.csv",
    )
    return parser.parse_args()


def load_branch_rows(conn: sqlite3.Connection, start: str, end: str) -> Iterable[sqlite3.Row]:
    sql = """
    SELECT
        date,
        stock_id,
        COALESCE(securities_trader_id, '') AS securities_trader_id,
        COALESCE(NULLIF(securities_trader, ''), securities_trader_id, 'UNKNOWN') AS trader_name,
        COALESCE(price, 0) AS price,
        COALESCE(buy, 0) AS buy,
        COALESCE(sell, 0) AS sell
    FROM branch_trader_daily_detail
    WHERE date BETWEEN ? AND ?
    """
    return conn.execute(sql, (start, end)).fetchall()


def load_latest_close(conn: sqlite3.Connection, end: str) -> Dict[str, float]:
    sql = """
    SELECT d.stock_id, d.close
    FROM stock_daily_trade_detail d
    JOIN (
        SELECT stock_id, MAX(date) AS max_date
        FROM stock_daily_trade_detail
        WHERE date <= ?
        GROUP BY stock_id
    ) m ON d.stock_id = m.stock_id AND d.date = m.max_date
    """
    return {normalize_stock_id(sid): float(close or 0.0) for sid, close in conn.execute(sql, (end,)).fetchall()}


def load_latest_stock_name(conn: sqlite3.Connection) -> Dict[str, str]:
    sql = """
    SELECT s.stock_id, COALESCE(s.stock_name, s.stock_id)
    FROM stock_info s
    JOIN (
        SELECT stock_id, MAX(date) AS max_date
        FROM stock_info
        GROUP BY stock_id
    ) m ON s.stock_id = m.stock_id AND s.date = m.max_date
    """
    return {normalize_stock_id(sid): str(name).strip() for sid, name in conn.execute(sql).fetchall()}


def load_volume_metrics(conn: sqlite3.Connection, start: str, end: str) -> Dict[str, dict]:
    sql = """
    WITH latest_day AS (
        SELECT stock_id, MAX(date) AS latest_date
        FROM stock_daily_trade_detail
        WHERE date <= ?
        GROUP BY stock_id
    ),
    interval_avg AS (
        SELECT stock_id, AVG(COALESCE(Trading_Volume, 0)) AS interval_avg_volume
        FROM stock_daily_trade_detail
        WHERE date BETWEEN ? AND ?
        GROUP BY stock_id
    ),
    latest_window AS (
        SELECT
            d.stock_id,
            d.date,
            COALESCE(d.Trading_Volume, 0) AS volume,
            ROW_NUMBER() OVER (PARTITION BY d.stock_id ORDER BY d.date DESC) AS rn
        FROM stock_daily_trade_detail d
        JOIN latest_day ld
            ON ld.stock_id = d.stock_id
           AND d.date <= ld.latest_date
    )
    SELECT
        ld.stock_id,
        COALESCE(latest.volume, 0) AS latest_volume,
        COALESCE(ia.interval_avg_volume, 0) AS interval_avg_volume,
        COALESCE(AVG(CASE WHEN lw.rn <= 3 THEN lw.volume END), 0) AS recent_3d_avg_volume,
        COALESCE(AVG(CASE WHEN lw.rn <= 5 THEN lw.volume END), 0) AS recent_5d_avg_volume
    FROM latest_day ld
    LEFT JOIN latest_window latest
        ON latest.stock_id = ld.stock_id AND latest.rn = 1
    LEFT JOIN interval_avg ia
        ON ia.stock_id = ld.stock_id
    LEFT JOIN latest_window lw
        ON lw.stock_id = ld.stock_id
    GROUP BY ld.stock_id, latest.volume, ia.interval_avg_volume
    """

    metrics: Dict[str, dict] = {}
    for sid, latest_v, interval_avg_v, recent3_v, recent5_v in conn.execute(sql, (end, start, end)).fetchall():
        stock_id = normalize_stock_id(sid)
        metrics[stock_id] = {
            "latest_volume": int(round(float(latest_v or 0))),
            "interval_avg_volume": float(interval_avg_v or 0),
            "recent_3d_avg_volume": float(recent3_v or 0),
            "recent_5d_avg_volume": float(recent5_v or 0),
        }
    return metrics


def build_summary(conn: sqlite3.Connection, start: str, end: str) -> List[dict]:
    rows = load_branch_rows(conn, start, end)
    if not rows:
        return []

    stock_agg: Dict[str, StockAgg] = defaultdict(StockAgg)
    trader_agg: Dict[Tuple[str, str], TraderAgg] = defaultdict(TraderAgg)
    trader_name_map: Dict[Tuple[str, str], str] = {}
    trader_day_net: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for r in rows:
        date = str(r[0])
        stock_id = normalize_stock_id(r[1])
        trader_id = str(r[2])
        trader_name = str(r[3])
        price = float(r[4] or 0.0)
        buy = int(r[5] or 0)
        sell = int(r[6] or 0)

        sagg = stock_agg[stock_id]
        sagg.buy_shares += buy
        sagg.sell_shares += sell
        sagg.buy_amount += price * buy
        sagg.sell_amount += price * sell
        if buy > 0:
            sagg.total_trade_count += 1
        if sell > 0:
            sagg.total_trade_count += 1

        key = (stock_id, trader_id)
        tagg = trader_agg[key]
        tagg.buy_shares += buy
        tagg.sell_shares += sell
        tagg.buy_amount += price * buy
        tagg.sell_amount += price * sell
        if buy > 0:
            tagg.buy_trade_count += 1
        if sell > 0:
            tagg.sell_trade_count += 1
        trader_name_map[key] = trader_name

        trader_day_net[(stock_id, trader_id, date)] += buy - sell

    for (stock_id, trader_id, _date), day_net in trader_day_net.items():
        tagg = trader_agg[(stock_id, trader_id)]
        if day_net > 0:
            tagg.buy_days += 1
        elif day_net < 0:
            tagg.sell_days += 1

    latest_close_map = load_latest_close(conn, end)
    stock_name_map = load_latest_stock_name(conn)
    volume_metrics = load_volume_metrics(conn, start, end)

    stock_to_traders: Dict[str, List[Tuple[str, TraderAgg]]] = defaultdict(list)
    for (stock_id, trader_id), tagg in trader_agg.items():
        stock_to_traders[stock_id].append((trader_id, tagg))

    output: List[dict] = []
    for stock_id, sagg in stock_agg.items():
        traders = stock_to_traders.get(stock_id, [])
        buy_positive = [x for x in traders if x[1].net_shares > 0]
        sell_negative = [x for x in traders if x[1].net_shares < 0]
        buy_trader_count = len(buy_positive)
        sell_trader_count = len(sell_negative)
        total_trader_count = buy_trader_count + sell_trader_count
        concentration = (
            (buy_trader_count - sell_trader_count) / total_trader_count if total_trader_count else 0.0
        )

        top_buy_days = max(traders, key=lambda x: x[1].buy_days, default=None)
        top_sell_days = max(traders, key=lambda x: x[1].sell_days, default=None)
        top_buy_trade_count = max(traders, key=lambda x: x[1].buy_trade_count, default=None)
        top_sell_trade_count = max(traders, key=lambda x: x[1].sell_trade_count, default=None)

        top_buy_amount = max(buy_positive, key=lambda x: x[1].buy_amount, default=None)

        buy_positive_total_shares = sum(tagg.buy_shares for _, tagg in buy_positive)
        buy_positive_total_amount = sum(tagg.buy_amount for _, tagg in buy_positive)
        sell_negative_total_shares = sum(tagg.sell_shares for _, tagg in sell_negative)
        sell_negative_total_amount = sum(tagg.sell_amount for _, tagg in sell_negative)
        avg_buy_price = (
            buy_positive_total_amount / buy_positive_total_shares if buy_positive_total_shares else 0.0
        )
        avg_sell_price = (
            sell_negative_total_amount / sell_negative_total_shares if sell_negative_total_shares else 0.0
        )

        best_profit = None
        for tid, tagg in traders:
            matched = min(tagg.buy_shares, tagg.sell_shares)
            if matched <= 0:
                continue
            profit = (tagg.avg_sell_price - tagg.avg_buy_price) * matched
            if best_profit is None or profit > best_profit[1]:
                best_profit = (tid, profit)

        if best_profit:
            bp_tid = best_profit[0]
            bp_agg = trader_agg[(stock_id, bp_tid)]
            best_profit_name = trader_name_map[(stock_id, bp_tid)]
            best_profit_value = best_profit[1]
            best_profit_avg_sell = bp_agg.avg_sell_price
        else:
            best_profit_name = ""
            best_profit_value = 0.0
            best_profit_avg_sell = 0.0

        if top_buy_days:
            top_buy_tid, top_buy_agg = top_buy_days
            top_buy_days_name = trader_name_map[(stock_id, top_buy_tid)]
            top_buy_days_count = top_buy_agg.buy_days
        else:
            top_buy_days_name = ""
            top_buy_days_count = 0

        if top_buy_amount:
            top_buy_tid, top_buy_agg = top_buy_amount
            top_buy_name = trader_name_map[(stock_id, top_buy_tid)]
            top_buy_cost = top_buy_agg.net_shares * top_buy_agg.avg_buy_price
            top_buy_avg_price = top_buy_agg.avg_buy_price
        else:
            top_buy_name = ""
            top_buy_cost = 0.0
            top_buy_avg_price = 0.0

        if top_sell_days:
            top_sell_tid, top_sell_agg = top_sell_days
            top_sell_name = trader_name_map[(stock_id, top_sell_tid)]
            top_sell_days_count = top_sell_agg.sell_days
        else:
            top_sell_name = ""
            top_sell_days_count = 0

        if top_buy_trade_count:
            top_buy_trade_tid, top_buy_trade_agg = top_buy_trade_count
            top_buy_trade_name = trader_name_map[(stock_id, top_buy_trade_tid)]
            top_buy_trade_total = top_buy_trade_agg.buy_trade_count
        else:
            top_buy_trade_name = ""
            top_buy_trade_total = 0

        if top_sell_trade_count:
            top_sell_trade_tid, top_sell_trade_agg = top_sell_trade_count
            top_sell_trade_name = trader_name_map[(stock_id, top_sell_trade_tid)]
            top_sell_trade_total = top_sell_trade_agg.sell_trade_count
        else:
            top_sell_trade_name = ""
            top_sell_trade_total = 0

        vol = volume_metrics.get(stock_id, {})

        output.append(
            {
                "股票代號": stock_id,
                "股票名稱": stock_name_map.get(stock_id, stock_id),
                "最新成交量": int(vol.get("latest_volume", 0)),
                "區間平均成交量": round(vol.get("interval_avg_volume", 0.0), 2),
                "最近三日平均成交量": round(vol.get("recent_3d_avg_volume", 0.0), 2),
                "最近五日平均成交量": round(vol.get("recent_5d_avg_volume", 0.0), 2),
                "總交易筆數": sagg.total_trade_count,
                "買筆數最多分點": top_buy_trade_name,
                "買筆數": top_buy_trade_total,
                "賣筆數最多分點": top_sell_trade_name,
                "賣筆數": top_sell_trade_total,
                "買超分點數": buy_trader_count,
                "賣超分點數": sell_trader_count,
                "籌碼集中度": round(concentration, 4),
                "買最多分點名稱": top_buy_days_name,
                "買超天數": top_buy_days_count,
                "賣超最多分點名稱": top_sell_name,
                "賣超天數": top_sell_days_count,
                "目前收盤價": round(latest_close_map.get(stock_id, 0.0), 2),
                "平均買超價格": round(avg_buy_price, 2),
                "平均賣超價格": round(avg_sell_price, 2),
                "獲利最高分點": best_profit_name,
                "獲利最高分點獲利金額": round(best_profit_value, 2),
                "獲利最高分點平均賣價": round(best_profit_avg_sell, 2),
                "買超最高分點": top_buy_name,
                "買超最高分點買超成本": round(top_buy_cost, 2),
                "買超最高分點平均買超價格": round(top_buy_avg_price, 2),
            }
        )

    output.sort(key=lambda x: x["股票代號"])
    return output


def write_csv(rows: List[dict], output_path: str) -> None:
    if not rows:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "股票代號",
        "股票名稱",
        "最新成交量",
        "區間平均成交量",
        "最近三日平均成交量",
        "最近五日平均成交量",
        "總交易筆數",
        "買筆數最多分點",
        "買筆數",
        "賣筆數最多分點",
        "賣筆數",
        "買超分點數",
        "賣超分點數",
        "籌碼集中度",
        "買最多分點名稱",
        "買超天數",
        "賣超最多分點名稱",
        "賣超天數",
        "目前收盤價",
        "平均買超價格",
        "平均賣超價格",
        "獲利最高分點",
        "獲利最高分點獲利金額",
        "獲利最高分點平均賣價",
        "買超最高分點",
        "買超最高分點買超成本",
        "買超最高分點平均買超價格",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"找不到資料庫: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        result = build_summary(conn, args.start, args.end)
    finally:
        conn.close()

    if not result:
        print("指定區間查無 branch_trader_daily_detail 資料。")
        return 0

    write_csv(result, args.output)
    print(f"完成，共 {len(result)} 檔股票，輸出: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
