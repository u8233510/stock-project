import json
import sqlite3
from pathlib import Path

# -----------------------------
# 1. 資料表定義 (完全對齊 API 原生 Schema + 新增主動流向表)
# -----------------------------
TABLE_REGISTRY = {
    "ohlcv": """
        CREATE TABLE IF NOT EXISTS stock_ohlcv_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            Trading_Volume INTEGER, Trading_money INTEGER,
            open REAL, max REAL, min REAL, close REAL,
            spread REAL, Trading_turnover REAL,
            PRIMARY KEY (date, stock_id)
        );""",
    "institutional": """
        CREATE TABLE IF NOT EXISTS institutional_investors_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            buy INTEGER, sell INTEGER, name TEXT,
            PRIMARY KEY (date, stock_id, name)
        );""",
    "branch": """
        CREATE TABLE IF NOT EXISTS branch_price_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            securities_trader_id TEXT NOT NULL, securities_trader TEXT,
            price REAL, buy INTEGER, sell INTEGER,
            PRIMARY KEY (date, stock_id, securities_trader_id, price)
        );""",
    "branch_trader_daily_detail": """
        CREATE TABLE IF NOT EXISTS branch_trader_daily_detail (
            date TEXT NOT NULL,
            securities_trader_id TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            securities_trader TEXT,
            price REAL,
            buy INTEGER,
            sell INTEGER,
            PRIMARY KEY (date, securities_trader_id, stock_id, price)
        );""",
    "securities_trader_info": """
        CREATE TABLE IF NOT EXISTS securities_trader_info (
            securities_trader_id TEXT NOT NULL,
            securities_trader TEXT,
            date TEXT,
            address TEXT,
            phone TEXT,
            PRIMARY KEY (securities_trader_id)
        );""",
    "per_pbr": """
        CREATE TABLE IF NOT EXISTS stock_per_pbr_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            dividend_yield REAL, PER REAL, PBR REAL,
            PRIMARY KEY (date, stock_id)
        );""",
    "margin_short": """
        CREATE TABLE IF NOT EXISTS margin_short_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            MarginPurchaseBuy INTEGER, MarginPurchaseCashRepayment INTEGER,
            MarginPurchaseLimit INTEGER, MarginPurchaseSell INTEGER,
            MarginPurchaseTodayBalance INTEGER, MarginPurchaseYesterdayBalance INTEGER,
            Note TEXT, OffsetLoanAndShort INTEGER,
            ShortSaleBuy INTEGER, ShortSaleCashRepayment INTEGER,
            ShortSaleLimit INTEGER, ShortSaleSell INTEGER,
            ShortSaleTodayBalance INTEGER, ShortSaleYesterdayBalance INTEGER,
            PRIMARY KEY (date, stock_id)
        );""",
    "day_trading": """
        CREATE TABLE IF NOT EXISTS stock_day_trading_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            BuyAfterSale TEXT, Volume INTEGER,
            BuyAmount INTEGER, SellAmount INTEGER,
            PRIMARY KEY (date, stock_id)
        );""",
    "holding_shares": """
        CREATE TABLE IF NOT EXISTS stock_holding_shares_per_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            HoldingSharesLevel TEXT NOT NULL,
            people INTEGER, percent REAL, unit INTEGER,
            PRIMARY KEY (date, stock_id, HoldingSharesLevel)
        );""",
    "securities_lending": """
        CREATE TABLE IF NOT EXISTS stock_securities_lending_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            transaction_type TEXT, volume INTEGER, fee_rate REAL,
            close REAL, original_return_date TEXT, original_lending_period INTEGER,
            PRIMARY KEY (date, stock_id, transaction_type, volume, original_return_date)
        );""",
    "month_revenue": """
        CREATE TABLE IF NOT EXISTS stock_month_revenue_monthly (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            country TEXT, revenue INTEGER,
            revenue_month INTEGER, revenue_year INTEGER,
            PRIMARY KEY (date, stock_id)
        );""",
    "financial_statements": """
        CREATE TABLE IF NOT EXISTS stock_financial_statements (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            type TEXT NOT NULL, value REAL, origin_name TEXT,
            PRIMARY KEY (date, stock_id, type)
        );""",
    "dividend": """
        CREATE TABLE IF NOT EXISTS stock_dividend (
            date TEXT NOT NULL, stock_id TEXT NOT NULL, year TEXT NOT NULL,
            StockEarningsDistribution REAL, StockStatutorySurplus REAL,
            StockExDividendTradingDate TEXT, TotalEmployeeStockDividend REAL,
            TotalEmployeeStockDividendAmount REAL, RatioOfEmployeeStockDividendOfTotal REAL,
            RatioOfEmployeeStockDividend REAL, CashEarningsDistribution REAL,
            CashStatutorySurplus REAL, CashExDividendTradingDate TEXT,
            CashDividendPaymentDate TEXT, TotalEmployeeCashDividend REAL,
            TotalNumberOfCashCapitalIncrease REAL, CashIncreaseSubscriptionRate REAL,
            CashIncreaseSubscriptionpRrice REAL, RemunerationOfDirectorsAndSupervisors REAL,
            ParticipateDistributionOfTotalShares REAL, AnnouncementDate TEXT,
            AnnouncementTime TEXT,
            PRIMARY KEY (date, stock_id, year)
        );""",
    "market_value": """
        CREATE TABLE IF NOT EXISTS stock_market_value_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            market_value INTEGER,
            PRIMARY KEY (date, stock_id)
        );""",
    "industry_chain": """
        CREATE TABLE IF NOT EXISTS stock_industry_chain (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            industry TEXT, sub_industry TEXT,
            PRIMARY KEY (date, stock_id)
        );""",
    "active_flow_daily": """
        CREATE TABLE IF NOT EXISTS stock_active_flow_daily (
            date TEXT NOT NULL, stock_id TEXT NOT NULL,
            active_buy_vol INTEGER,   -- 主動買總量 (TickType=2)
            active_sell_vol INTEGER,  -- 主動賣總量 (TickType=1)
            PRIMARY KEY (date, stock_id)
        );""",
    "ohlcv_minute": """
        CREATE TABLE IF NOT EXISTS stock_ohlcv_minute (
            date_time TEXT NOT NULL, stock_id TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, 
            volume INTEGER, active_buy_vol INTEGER, active_sell_vol INTEGER,
            PRIMARY KEY (date_time, stock_id)
        );""",
    # ✅ 新增：分點加權成本中間表 (用於加速需求 1 & 3)
    "branch_weighted_cost": """
        CREATE TABLE IF NOT EXISTS branch_weighted_cost (
            stock_id TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            avg_cost REAL,            -- 加權平均成本
            total_net_volume INTEGER, -- 區間淨張數
            concentration REAL,       -- 籌碼集中度%
            window_type TEXT DEFAULT 'legacy', -- rolling / user_custom / legacy
            window_days INTEGER,      -- 5/20/60 或自訂天數
            PRIMARY KEY (stock_id, start_date, end_date)
        );""",    
    "ingest_log": """
        CREATE TABLE IF NOT EXISTS ingest_log (
            dataset TEXT NOT NULL, trade_date TEXT NOT NULL, stock_id TEXT NOT NULL,
            fetched_at TEXT, row_count INTEGER, status TEXT, message TEXT,
            PRIMARY KEY (dataset, trade_date, stock_id)
        );""",
    "data_ingest_log": """
        CREATE TABLE IF NOT EXISTS data_ingest_log (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            api_name TEXT NOT NULL,
            api_count INTEGER DEFAULT 0,
            db_count INTEGER DEFAULT 0,
            status TEXT,
            updated_at TEXT,
            PRIMARY KEY (date, stock_id, api_name)
        );""",
    "branch_sync_log": """
        CREATE TABLE IF NOT EXISTS branch_sync_log (
            date TEXT NOT NULL,
            securities_trader_id TEXT NOT NULL,
            api_name TEXT NOT NULL DEFAULT 'taiwan_stock_trading_daily_report',
            row_count INTEGER DEFAULT 0,
            status TEXT,
            message TEXT,
            updated_at TEXT,
            PRIMARY KEY (date, securities_trader_id, api_name)
        );"""
    ,
    "stock_info": """
        CREATE TABLE IF NOT EXISTS stock_info (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            stock_name TEXT,
            type TEXT,
            industry_category TEXT,
            PRIMARY KEY (date, stock_id)
        );""",
    "stock_info_sync_log": """
        CREATE TABLE IF NOT EXISTS stock_info_sync_log (
            date TEXT NOT NULL,
            api_name TEXT NOT NULL DEFAULT 'taiwan_stock_info_with_warrant',
            row_count INTEGER DEFAULT 0,
            status TEXT,
            message TEXT,
            updated_at TEXT,
            PRIMARY KEY (date, api_name)
        );"""
}

# 索引配置
INDEX_REGISTRY = {
    "ohlcv": ["CREATE INDEX IF NOT EXISTS idx_ohlcv_stock_date ON stock_ohlcv_daily(stock_id, date);"],
    "institutional": ["CREATE INDEX IF NOT EXISTS idx_inst_stock_date ON institutional_investors_daily(stock_id, date);"],
    "margin_short": ["CREATE INDEX IF NOT EXISTS idx_ms_stock_date ON margin_short_daily(stock_id, date);"],
    "financial_statements": ["CREATE INDEX IF NOT EXISTS idx_fs_stock_date ON stock_financial_statements(stock_id, date);"],
    "ohlcv_minute": ["CREATE INDEX IF NOT EXISTS idx_min_stock_date ON stock_ohlcv_minute(stock_id, date_time);"],
    "active_flow_daily": ["CREATE INDEX IF NOT EXISTS idx_flow_stock_date ON stock_active_flow_daily(stock_id, date);"],
    "branch_weighted_cost": ["CREATE INDEX IF NOT EXISTS idx_cost_lookup ON branch_weighted_cost(stock_id);"],
    "branch_sync_log": [
        "CREATE INDEX IF NOT EXISTS idx_branch_sync_status_date ON branch_sync_log(status, date);"
    ],
    "branch_trader_daily_detail": [
        "CREATE INDEX IF NOT EXISTS idx_branch_trader_date ON branch_trader_daily_detail(securities_trader_id, date);",
        "CREATE INDEX IF NOT EXISTS idx_branch_trader_stock_date ON branch_trader_daily_detail(stock_id, date);"
    ],
    "securities_trader_info": [
        "CREATE INDEX IF NOT EXISTS idx_trader_info_name ON securities_trader_info(securities_trader);"
    ],
    "stock_info": [
        "CREATE INDEX IF NOT EXISTS idx_stock_info_stock_date ON stock_info(stock_id, date);",
        "CREATE INDEX IF NOT EXISTS idx_stock_info_industry ON stock_info(industry_category);"
    ],
    "stock_info_sync_log": [
        "CREATE INDEX IF NOT EXISTS idx_stock_info_sync_status_date ON stock_info_sync_log(status, date);"
    ],
}

# -----------------------------
# 2. 核心功能函式
# -----------------------------

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_db_connection(cfg: dict) -> sqlite3.Connection:
    path = Path(cfg["storage"]["sqlite_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))

# --- 🚀 中繼資料驅動工具 (Metadata-Driven Tools) ---

def get_table_columns(conn, table_name):
    """ 取得指定表格的所有真實欄位名稱 """
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    except Exception:
        return []

def match_column(available_cols, keywords):
    """ 
    自動在欄位清單中尋找符合關鍵字的欄位 (不分大小寫)。
    例如：keywords=['Margin', 'Balance'] 會匹配到 'MarginPurchaseTodayBalance'
    """
    for col in available_cols:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None


def ensure_data_ingest_log_schema(conn):
    """確保 data_ingest_log 有完整欄位，並支援以 API 維度紀錄狀態。"""
    conn.execute(TABLE_REGISTRY["data_ingest_log"])
    cols = get_table_columns(conn, "data_ingest_log")
    if not cols:
        conn.commit()
        return

    required_cols = ["date", "stock_id", "api_name", "api_count", "db_count", "status", "updated_at"]

    # 若已為新結構僅需補欄位
    if set(required_cols).issubset(set(cols)):
        conn.commit()
        return

    # 舊版主鍵是 (date, stock_id)，需要重建表格主鍵為 (date, stock_id, api_name)
    conn.execute("ALTER TABLE data_ingest_log RENAME TO data_ingest_log_old")
    conn.execute(
        """
        CREATE TABLE data_ingest_log (
            date TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            api_name TEXT NOT NULL,
            api_count INTEGER DEFAULT 0,
            db_count INTEGER DEFAULT 0,
            status TEXT,
            updated_at TEXT,
            PRIMARY KEY (date, stock_id, api_name)
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO data_ingest_log(date, stock_id, api_name, api_count, db_count, status, updated_at)
        SELECT date, stock_id, 'legacy',
               COALESCE(api_count, 0), COALESCE(db_count, 0), status, updated_at
        FROM data_ingest_log_old
        """
    )
    conn.execute("DROP TABLE data_ingest_log_old")
    conn.commit()


def ensure_branch_sync_log_schema(conn):
    """確保 branch_sync_log 表存在（供分點明細同步任務使用）。"""
    conn.execute(TABLE_REGISTRY["branch_sync_log"])
    for idx in INDEX_REGISTRY.get("branch_sync_log", []):
        conn.execute(idx)
    conn.commit()


def ensure_stock_info_sync_log_schema(conn):
    """確保 stock_info / stock_info_sync_log 表存在（供股票資訊同步任務使用）。"""
    conn.execute(TABLE_REGISTRY["stock_info"])
    conn.execute(TABLE_REGISTRY["stock_info_sync_log"])
    for idx in INDEX_REGISTRY.get("stock_info", []):
        conn.execute(idx)
    for idx in INDEX_REGISTRY.get("stock_info_sync_log", []):
        conn.execute(idx)
    conn.commit()

# -----------------------------
# 3. 初始化邏輯
# -----------------------------
def init_database():
    cfg = load_config()
    conn = get_db_connection(cfg)
    cur = conn.cursor()
    enabled = (cfg.get("datasets") or {}).get("enabled", [])
    
    print(f"🚀 初始化資料庫，重建 15 項指標與分析擴充表...")
    for key in enabled:
        if key in TABLE_REGISTRY:
            cur.execute(TABLE_REGISTRY[key])
            for idx in INDEX_REGISTRY.get(key, []):
                cur.execute(idx)
                
    conn.commit()
    conn.close()
    print("✅ 資料庫結構初始化完成。")

if __name__ == "__main__":
    init_database()
