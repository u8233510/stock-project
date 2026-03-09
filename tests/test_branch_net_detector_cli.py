import sqlite3
import unittest

from branch_net_detector_cli import build_summary


class BranchNetDetectorCLITest(unittest.TestCase):
    def test_build_summary_maps_numeric_stock_ids_to_zero_padded_ids(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        conn.executescript(
            """
            CREATE TABLE branch_trader_daily_detail (
                date TEXT,
                stock_id TEXT,
                securities_trader_id TEXT,
                securities_trader TEXT,
                price REAL,
                buy INTEGER,
                sell INTEGER
            );

            CREATE TABLE stock_daily_trade_detail (
                date TEXT,
                stock_id TEXT,
                close REAL,
                Trading_Volume INTEGER
            );

            CREATE TABLE stock_info (
                date TEXT,
                stock_id TEXT,
                stock_name TEXT
            );
            """
        )

        conn.execute(
            """
            INSERT INTO branch_trader_daily_detail
            (date, stock_id, securities_trader_id, securities_trader, price, buy, sell)
            VALUES ('2024-01-05', '0050', 'T1', '交易員A', 100, 10, 0)
            """
        )
        conn.executemany(
            "INSERT INTO stock_daily_trade_detail (date, stock_id, close, Trading_Volume) VALUES (?, ?, ?, ?)",
            [
                ("2024-01-03", "0050", 101.5, 1000),
                ("2024-01-04", "0050", 102.0, 2000),
                ("2024-01-05", "0050", 103.0, 3000),
            ],
        )
        conn.execute(
            "INSERT INTO stock_info (date, stock_id, stock_name) VALUES ('2024-01-05', '0050', '元大台灣50')"
        )
        conn.commit()

        rows = build_summary(conn, "2024-01-01", "2024-01-05")
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["股票代號"], "0050")
        self.assertEqual(row["目前收盤價"], 103.0)
        self.assertEqual(row["最新成交量"], 3000)
        self.assertEqual(row["區間平均成交量"], 2000.0)
        self.assertEqual(row["最近三日平均成交量"], 2000.0)
        self.assertEqual(row["最近五日平均成交量"], 2000.0)

    def test_build_summary_counts_buy_sell_trade_occurrences(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        conn.executescript(
            """
            CREATE TABLE branch_trader_daily_detail (
                date TEXT,
                stock_id TEXT,
                securities_trader_id TEXT,
                securities_trader TEXT,
                price REAL,
                buy INTEGER,
                sell INTEGER
            );

            CREATE TABLE stock_daily_trade_detail (
                date TEXT,
                stock_id TEXT,
                close REAL,
                Trading_Volume INTEGER
            );

            CREATE TABLE stock_info (
                date TEXT,
                stock_id TEXT,
                stock_name TEXT
            );
            """
        )

        conn.executemany(
            """
            INSERT INTO branch_trader_daily_detail
            (date, stock_id, securities_trader_id, securities_trader, price, buy, sell)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("2024-01-05", "2330", "B1", "分點甲", 30.2, 200, 0),
                ("2024-01-05", "2330", "S1", "分點乙", 30.1, 0, 10),
                ("2024-01-06", "2330", "B1", "分點甲", 30.0, 100, 0),
                ("2024-01-06", "2330", "S2", "分點丙", 29.8, 0, 20),
                ("2024-01-07", "2330", "S1", "分點乙", 29.7, 0, 5),
            ],
        )
        conn.execute(
            "INSERT INTO stock_daily_trade_detail (date, stock_id, close, Trading_Volume) VALUES ('2024-01-07', '2330', 600.0, 10000)"
        )
        conn.execute(
            "INSERT INTO stock_info (date, stock_id, stock_name) VALUES ('2024-01-07', '2330', '台積電')"
        )
        conn.commit()

        rows = build_summary(conn, "2024-01-01", "2024-01-07")
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["總交易筆數"], 5)
        self.assertEqual(row["買筆數最多分點"], "分點甲")
        self.assertEqual(row["買筆數"], 2)
        self.assertEqual(row["賣筆數最多分點"], "分點乙")
        self.assertEqual(row["賣筆數"], 2)

    def test_total_trade_count_counts_buy_and_sell_separately_when_both_exist(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        conn.executescript(
            """
            CREATE TABLE branch_trader_daily_detail (
                date TEXT,
                stock_id TEXT,
                securities_trader_id TEXT,
                securities_trader TEXT,
                price REAL,
                buy INTEGER,
                sell INTEGER
            );

            CREATE TABLE stock_daily_trade_detail (
                date TEXT,
                stock_id TEXT,
                close REAL,
                Trading_Volume INTEGER
            );

            CREATE TABLE stock_info (
                date TEXT,
                stock_id TEXT,
                stock_name TEXT
            );
            """
        )

        conn.executemany(
            """
            INSERT INTO branch_trader_daily_detail
            (date, stock_id, securities_trader_id, securities_trader, price, buy, sell)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("2024-01-05", "2330", "B1", "分點甲", 30.2, 200, 0),
                ("2024-01-05", "2330", "S1", "分點乙", 30.1, 0, 10),
                ("2024-01-06", "2330", "B2", "分點丁", 30.15, 1, 0),
                ("2024-01-06", "2330", "S2", "分點丙", 30.0, 0, 2),
                ("2024-01-07", "2330", "X1", "分點戊", 30.05, 1, 1),
            ],
        )
        conn.execute(
            "INSERT INTO stock_daily_trade_detail (date, stock_id, close, Trading_Volume) VALUES ('2024-01-07', '2330', 600.0, 10000)"
        )
        conn.execute(
            "INSERT INTO stock_info (date, stock_id, stock_name) VALUES ('2024-01-07', '2330', '台積電')"
        )
        conn.commit()

        rows = build_summary(conn, "2024-01-01", "2024-01-07")
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["總交易筆數"], 6)


if __name__ == "__main__":
    unittest.main()
