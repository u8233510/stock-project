import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import database


API_TABLE_MAP = {
    "TaiwanStockTick": "stock_ohlcv_minute",
    "TaiwanStockPrice": "stock_ohlcv_daily",
    "TaiwanStockInstitutionalInvestorsBuySell": "institutional_investors_daily",
    "TaiwanStockTradingDailyReport": "branch_price_daily",
    "TaiwanStockPER": "stock_per_pbr_daily",
    "TaiwanStockMarginPurchaseShortSale": "margin_short_daily",
    "TaiwanStockDayTrading": "stock_day_trading_daily",
    "TaiwanStockHoldingSharesPer": "stock_holding_shares_per_daily",
    "TaiwanStockSecuritiesLending": "stock_securities_lending_daily",
    "TaiwanStockMonthRevenue": "stock_month_revenue_monthly",
    "TaiwanStockFinancialStatements": "stock_financial_statements",
    "TaiwanStockDividend": "stock_dividend",
    "TaiwanStockMarketValue": "stock_market_value_daily",
    "TaiwanStockIndustryChain": "stock_industry_chain",
}


def _resolve_columns(conn, table_name):
    cols = database.get_table_columns(conn, table_name)
    if not cols:
        return None, None

    date_col = database.match_column(cols, ["date", "time"]) or database.match_column(cols, ["date"])
    stock_col = database.match_column(cols, ["stock", "id"]) or database.match_column(cols, ["stock"])
    return date_col, stock_col


def _query_db_count(conn, table_name, date_col, stock_col, trade_date, stock_id):
    if table_name == "stock_ohlcv_minute":
        sql = (
            f"SELECT COUNT(*) FROM {table_name} "
            f"WHERE {stock_col} = ? AND date({date_col}) = date(?)"
        )
    else:
        sql = (
            f"SELECT COUNT(*) FROM {table_name} "
            f"WHERE {stock_col} = ? AND substr({date_col}, 1, 10) = ?"
        )
    return conn.execute(sql, (stock_id, trade_date)).fetchone()[0]


def _query_log_row(conn, trade_date, stock_id, api_name):
    return conn.execute(
        """
        SELECT date, stock_id, api_name, api_count, db_count, status, updated_at
        FROM data_ingest_log
        WHERE date = ? AND stock_id = ? AND api_name = ?
        LIMIT 1
        """,
        (trade_date, stock_id, api_name),
    ).fetchone()


def _query_history(conn, stock_id, api_name, start_date, end_date):
    return conn.execute(
        """
        SELECT date, api_count, db_count, status, updated_at
        FROM data_ingest_log
        WHERE stock_id = ?
          AND api_name = ?
          AND date >= ?
          AND date <= ?
        ORDER BY date
        """,
        (stock_id, api_name, start_date, end_date),
    ).fetchall()


def _validate_date(date_str):
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def main():
    parser = argparse.ArgumentParser(description="分析 API count 與 DB count 差異")
    parser.add_argument("--date", required=True, type=_validate_date, help="交易日期，格式 YYYY-MM-DD")
    parser.add_argument("--stock-id", required=True, help="股票代號，例如 4104")
    parser.add_argument("--api-name", default="TaiwanStockTick", help="API 名稱，預設 TaiwanStockTick")
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="列出前後幾個交易日做對照，預設 5",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="設定檔路徑，預設 config.json",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="直接指定 SQLite 檔案路徑 (優先於 --config)",
    )
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
    else:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"❌ 找不到設定檔: {cfg_path}")
            print("請提供 --db-path 或可讀取的 --config")
            return
        cfg = database.load_config(str(cfg_path))
        db_path = Path(cfg["storage"]["sqlite_path"])

    if not db_path.exists():
        print(f"❌ 找不到資料庫檔案: {db_path}")
        print("請確認 --db-path 或 config.json 內的 storage.sqlite_path")
        return

    conn = sqlite3.connect(str(db_path))

    try:
        api_name = args.api_name
        table_name = API_TABLE_MAP.get(api_name)
        if not table_name:
            print(f"❌ 不支援的 api_name: {api_name}")
            print("可用 api_name:", ", ".join(sorted(API_TABLE_MAP.keys())))
            return

        date_col, stock_col = _resolve_columns(conn, table_name)
        if not date_col or not stock_col:
            print(f"❌ 找不到表格欄位: {table_name}")
            return

        print("=" * 72)
        print(f"🔎 對帳條件: date={args.date}, stock_id={args.stock_id}, api_name={api_name}")
        print(f"📦 目標資料表: {table_name} (date_col={date_col}, stock_col={stock_col})")

        log_row = _query_log_row(conn, args.date, args.stock_id, api_name)
        if not log_row:
            print("⚠️ data_ingest_log 沒有這筆資料，可能尚未同步或條件輸入錯誤。")
        else:
            _, _, _, api_count, db_count_logged, status, updated_at = log_row
            print("\n[1] data_ingest_log")
            print(f"- status     : {status}")
            print(f"- api_count  : {api_count}")
            print(f"- db_count   : {db_count_logged}")
            print(f"- updated_at : {updated_at}")

        db_count_actual = _query_db_count(
            conn,
            table_name,
            date_col,
            stock_col,
            args.date,
            args.stock_id,
        )
        print("\n[2] 資料表實際筆數")
        print(f"- actual_db_count: {db_count_actual}")

        if log_row:
            api_count = int(log_row[3] or 0)
            db_count_logged = int(log_row[4] or 0)
            print("\n[3] 差異分析")
            print(f"- db_count(log) - api_count     = {db_count_logged - api_count}")
            print(f"- actual_db_count - db_count(log) = {db_count_actual - db_count_logged}")
            print(f"- actual_db_count - api_count     = {db_count_actual - api_count}")

            if api_name == "TaiwanStockTick":
                print("\n💡 TaiwanStockTick 提醒:")
                print("- api_count 是逐筆 tick 筆數。")
                print("- db_count 是 1 分鐘聚合後 K 棒筆數。")
                print("- 兩者口徑不同，通常不會相等；應先確認是否拿來比較了不同粒度。")

        center = datetime.strptime(args.date, "%Y-%m-%d").date()
        start = (center - timedelta(days=args.window * 2)).strftime("%Y-%m-%d")
        end = (center + timedelta(days=args.window * 2)).strftime("%Y-%m-%d")
        history = _query_history(conn, args.stock_id, api_name, start, end)

        print("\n[4] 附近日期對照 (data_ingest_log)")
        if not history:
            print("- 無資料")
        else:
            print("date       | api_count | db_count | status   | updated_at")
            print("-" * 72)
            for d, api_c, db_c, s, u in history:
                marker = "<--" if str(d)[:10] == args.date else ""
                print(f"{str(d)[:10]} | {int(api_c or 0):9d} | {int(db_c or 0):8d} | {str(s or ''):8s} | {str(u or '')} {marker}")

        print("=" * 72)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
