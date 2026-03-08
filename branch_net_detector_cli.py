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

    @property
    def net_shares(self) -> int:
        return self.buy_shares - self.sell_shares

    @property
    def concentration(self) -> float:
        total = self.buy_shares + self.sell_shares
        return (self.net_shares / total * 100.0) if total else 0.0

    @property
    def avg_buy_price(self) -> float:
        return (self.buy_amount / self.buy_shares) if self.buy_shares else 0.0

    @property
    def avg_sell_price(self) -> float:
        return (self.sell_amount / self.sell_shares) if self.sell_shares else 0.0


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
    FROM branch_price_daily
    WHERE date BETWEEN ? AND ?
    """
    return conn.execute(sql, (start, end)).fetchall()


def load_latest_close(conn: sqlite3.Connection, end: str) -> Dict[str, float]:
    sql = """
    SELECT o.stock_id, o.close
    FROM stock_ohlcv_daily o
    JOIN (
        SELECT stock_id, MAX(date) AS max_date
        FROM stock_ohlcv_daily
        WHERE date <= ?
        GROUP BY stock_id
    ) m ON o.stock_id = m.stock_id AND o.date = m.max_date
    """
    return {sid: float(close or 0.0) for sid, close in conn.execute(sql, (end,)).fetchall()}


def load_latest_stock_name(conn: sqlite3.Connection, end: str) -> Dict[str, str]:
    sql = """
    SELECT s.stock_id, COALESCE(s.stock_name, s.stock_id)
    FROM stock_info s
    JOIN (
        SELECT stock_id, MAX(date) AS max_date
        FROM stock_info
        WHERE date <= ?
        GROUP BY stock_id
    ) m ON s.stock_id = m.stock_id AND s.date = m.max_date
    """
    return {sid: name for sid, name in conn.execute(sql, (end,)).fetchall()}


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
        stock_id = str(r[1])
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

        key = (stock_id, trader_id)
        tagg = trader_agg[key]
        tagg.buy_shares += buy
        tagg.sell_shares += sell
        tagg.buy_amount += price * buy
        tagg.sell_amount += price * sell
        trader_name_map[key] = trader_name

        trader_day_net[(stock_id, trader_id, date)] += buy - sell

    for (stock_id, trader_id, _date), day_net in trader_day_net.items():
        tagg = trader_agg[(stock_id, trader_id)]
        if day_net > 0:
            tagg.buy_days += 1
        elif day_net < 0:
            tagg.sell_days += 1

    latest_close_map = load_latest_close(conn, end)
    stock_name_map = load_latest_stock_name(conn, end)

    stock_to_traders: Dict[str, List[Tuple[str, TraderAgg]]] = defaultdict(list)
    for (stock_id, trader_id), tagg in trader_agg.items():
        stock_to_traders[stock_id].append((trader_id, tagg))

    output: List[dict] = []
    for stock_id, sagg in stock_agg.items():
        traders = stock_to_traders.get(stock_id, [])
        buy_positive = [x for x in traders if x[1].net_shares > 0]
        sell_negative = [x for x in traders if x[1].net_shares < 0]

        top_buy = max(traders, key=lambda x: x[1].net_shares, default=None)
        top_sell = min(traders, key=lambda x: x[1].net_shares, default=None)

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

        if top_buy:
            top_buy_tid, top_buy_agg = top_buy
            top_buy_name = trader_name_map[(stock_id, top_buy_tid)]
            top_buy_days = top_buy_agg.buy_days
            top_buy_cost = max(top_buy_agg.net_shares, 0) * top_buy_agg.avg_buy_price
            top_buy_avg_price = top_buy_agg.avg_buy_price
        else:
            top_buy_name = ""
            top_buy_days = 0
            top_buy_cost = 0.0
            top_buy_avg_price = 0.0

        if top_sell:
            top_sell_tid, top_sell_agg = top_sell
            top_sell_name = trader_name_map[(stock_id, top_sell_tid)]
            top_sell_days = top_sell_agg.sell_days
        else:
            top_sell_name = ""
            top_sell_days = 0

        output.append(
            {
                "股票代號": stock_id,
                "股票名稱": stock_name_map.get(stock_id, stock_id),
                "買(張數)": sagg.buy_shares,
                "賣(張數)": sagg.sell_shares,
                "買賣超": sagg.net_shares,
                "買超分點數": len(buy_positive),
                "賣超分點數": len(sell_negative),
                "籌碼集中度": round(sagg.concentration, 2),
                "買最多分點名稱": top_buy_name,
                "買超天數": top_buy_days,
                "賣超最多分點名稱": top_sell_name,
                "賣超天數": top_sell_days,
                "目前收盤價": round(latest_close_map.get(stock_id, 0.0), 2),
                "平均買超價格": round(sagg.avg_buy_price, 2),
                "平均賣超價格": round(sagg.avg_sell_price, 2),
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
        "買(張數)",
        "賣(張數)",
        "買賣超",
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
        print("指定區間查無 branch_price_daily 資料。")
        return 0

    write_csv(result, args.output)
    print(f"完成，共 {len(result)} 檔股票，輸出: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
