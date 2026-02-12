"""TaiwanStockMarketValue API 測試範例。

用途：
- 抓取 TaiwanStockMarketValue 在指定日期與股票代號的資料
- 輸出回傳筆數與資料內容，方便確認 API 是否出現多筆

預設測試條件：
- date: 2026-02-05
- stock_id: 2330
"""

import argparse

from FinMind.data import DataLoader

import database


def parse_args():
    parser = argparse.ArgumentParser(description="TaiwanStockMarketValue API 測試範例")
    parser.add_argument("--date", default="2026-02-05", help="查詢日期，格式 YYYY-MM-DD")
    parser.add_argument("--stock-id", default="2330", help="股票代號")
    parser.add_argument(
        "--token",
        default=None,
        help="FinMind API token；若未提供，會改讀 config.json 的 finmind.api_token",
    )
    return parser.parse_args()


def resolve_token(cli_token):
    if cli_token:
        return cli_token
    cfg = database.load_config()
    return cfg["finmind"]["api_token"]


def main():
    args = parse_args()
    token = resolve_token(args.token)

    api = DataLoader()
    api.login_by_token(api_token=token)

    df = api.get_data(
        dataset="TaiwanStockMarketValue",
        data_id=args.stock_id,
        start_date=args.date,
        end_date=args.date,
    )

    print("=" * 80)
    print(f"dataset   : TaiwanStockMarketValue")
    print(f"stock_id  : {args.stock_id}")
    print(f"date      : {args.date}")
    print(f"row_count : {len(df)}")
    print("=" * 80)

    if df is None or df.empty:
        print("查無資料（可能休市或 API 該日無資料）")
        return

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
