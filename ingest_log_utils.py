import database


DATA_INGEST_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS data_ingest_log (
    date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    api TEXT NOT NULL,
    api_count INTEGER DEFAULT 0,
    db_count INTEGER DEFAULT 0,
    status TEXT,
    updated_at TEXT,
    PRIMARY KEY (date, stock_id, api)
);
"""


def ensure_data_ingest_log_table(conn):
    cols = database.get_table_columns(conn, "data_ingest_log")
    if not cols:
        conn.execute(DATA_INGEST_LOG_SCHEMA)
        conn.commit()
        return

    if "api" not in cols:
        with conn:
            conn.execute("ALTER TABLE data_ingest_log RENAME TO data_ingest_log_old")
            conn.execute(DATA_INGEST_LOG_SCHEMA)
            conn.execute(
                """
                INSERT INTO data_ingest_log(date, stock_id, api, api_count, db_count, status, updated_at)
                SELECT date, stock_id, 'legacy', api_count, db_count, status, updated_at
                FROM data_ingest_log_old
                """
            )
            conn.execute("DROP TABLE data_ingest_log_old")


def get_data_ingest_status(conn, stock_id, api_name, trade_date):
    cols = database.get_table_columns(conn, "data_ingest_log")
    if not cols:
        return None

    if "api" in cols:
        sql = "SELECT status FROM data_ingest_log WHERE stock_id = ? AND api = ? AND date = ? LIMIT 1"
        row = conn.execute(sql, (stock_id, api_name, trade_date)).fetchone()
    else:
        sql = "SELECT status FROM data_ingest_log WHERE stock_id = ? AND date = ? LIMIT 1"
        row = conn.execute(sql, (stock_id, trade_date)).fetchone()
    return row[0] if row else None


def write_data_ingest_log(conn, trade_date, stock_id, api_name, api_count, db_count, status):
    cols = database.get_table_columns(conn, "data_ingest_log")
    if "api" in cols:
        conn.execute(
            """
            INSERT OR REPLACE INTO data_ingest_log(date, stock_id, api, api_count, db_count, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
            """,
            (trade_date, stock_id, api_name, int(api_count), int(db_count), status),
        )
    else:
        conn.execute(
            """
            INSERT OR REPLACE INTO data_ingest_log(date, stock_id, api_count, db_count, status, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
            """,
            (trade_date, stock_id, int(api_count), int(db_count), status),
        )
    conn.commit()
