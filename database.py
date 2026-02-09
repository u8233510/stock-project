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
            PRIMARY KEY (stock_id, start_date, end_date)
        );""",    
    "ingest_log": """
        CREATE TABLE IF NOT EXISTS ingest_log (
            dataset TEXT NOT NULL, trade_date TEXT NOT NULL, stock_id TEXT NOT NULL,
            fetched_at TEXT, row_count INTEGER, status TEXT, message TEXT,
            PRIMARY KEY (dataset, trade_date, stock_id)
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
    "branch_weighted_cost": ["CREATE INDEX IF NOT EXISTS idx_cost_lookup ON branch_weighted_cost(stock_id);"]
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
                
    # 強制初始化分析擴充表與日誌
    cur.execute(TABLE_REGISTRY["active_flow_daily"])
    cur.execute(TABLE_REGISTRY["ohlcv_minute"])
    cur.execute(TABLE_REGISTRY["ingest_log"])
    
    conn.commit()
    conn.close()
    print("✅ 資料庫結構初始化完成。")

if __name__ == "__main__":
    init_database()