import sqlite3
import unittest

from branch_net_detector_cli import FIELDNAMES, build_summary, format_rows_for_output


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

    def test_flags_and_volume_trend_fields_are_computed(self):
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
                ("2024-01-01", "1101", "A", "分點A", 10, 120, 0),
                ("2024-01-02", "1101", "A", "分點A", 11, 80, 0),
                ("2024-01-03", "1101", "A", "分點A", 12, 0, 100),
                ("2024-01-04", "1101", "A", "分點A", 13, 0, 80),
                ("2024-01-01", "1101", "B", "分點B", 10, 0, 120),
            ],
        )
        conn.executemany(
            "INSERT INTO stock_daily_trade_detail (date, stock_id, close, Trading_Volume) VALUES (?, ?, ?, ?)",
            [
                ("2024-01-01", "1101", 20.0, 100),
                ("2024-01-02", "1101", 20.5, 200),
                ("2024-01-03", "1101", 21.0, 300),
                ("2024-01-04", "1101", 21.5, 400),
            ],
        )
        conn.execute(
            "INSERT INTO stock_info (date, stock_id, stock_name) VALUES ('2024-01-04', '1101', '台泥')"
        )
        conn.commit()

        rows = build_summary(conn, "2024-01-01", "2024-01-04")
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["BDCV"], "是")
        self.assertEqual(row["SDCV"], "否")
        self.assertEqual(row["區間成交量趨勢"], "觀察不出來")


    def test_format_rows_for_output_applies_column_order_and_number_format(self):
        rows = [
            {
                "股票代號": "2330",
                "股票名稱": "台積電",
                "最新成交量": 1234567,
                "區間平均成交量": 98765.4321,
                "最近三日平均成交量": 11111.666,
                "最近五日平均成交量": 22222.555,
                "區間成交量趨勢": "逐漸變大",
                "目前收盤價": 600.5,
                "平均買超價格": 599.1,
                "平均賣超價格": 601.2,
                "買超分點數": 5,
                "賣超分點數": 3,
                "籌碼集中度": 0.2345,
                "總交易筆數": 1200,
                "買筆數最多分點": "分點甲",
                "買筆數": 12,
                "賣筆數最多分點": "分點乙",
                "賣筆數": 10,
                "BDCV": "是",
                "買最多分點名稱": "分點甲",
                "買超天數": 8,
                "買超最高分點": "分點甲",
                "買超最高分點買超成本": 1234567.89,
                "買超最高分點平均買超價格": 598.8,
                "買超張數最多的分點": "分點甲",
                "買超張數": 999,
                "買超張數區間成交量佔比": 0.1299,
                "SDCV": "否",
                "賣超最多分點名稱": "分點丙",
                "賣超天數": 7,
                "獲利最高分點": "分點丁",
                "獲利最高分點獲利金額": 7654321.99,
                "獲利最高分點平均賣價": 602.3,
                "賣超張數最多的分點": "分點丙",
                "賣超張數": 3333,
                "賣超張數區間成交量佔比": 0.0456,
            }
        ]

        formatted = format_rows_for_output(rows)
        self.assertEqual(list(formatted[0].keys()), FIELDNAMES)
        self.assertEqual(formatted[0]["最新成交量"], "1,234,567")
        self.assertEqual(formatted[0]["區間平均成交量"], "98,765.43")
        self.assertEqual(formatted[0]["籌碼集中度"], "0.23")
        self.assertEqual(formatted[0]["買超張數區間成交量佔比"], "12.99%")
        self.assertEqual(formatted[0]["賣超張數區間成交量佔比"], "4.56%")


if __name__ == "__main__":
    unittest.main()
