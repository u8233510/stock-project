import streamlit as st

from modules import (
    branch_accumulation_scan,
    branch_analysis,
    branch_anomaly,
    branch_net_detector_entry,
    data_browser,
    data_management,
    fundamental_analysis,
    prediction,
    stock_health,
    tech_analysis,
    winner_branch_system,
)

st.set_page_config(page_title="台股 AI 自動化分析平台", layout="wide")

st.sidebar.title("🚀 功能選單")
main_menu = st.sidebar.radio(
    "請選擇功能模組：",
    [
        "📊 資料瀏覽器",
        "📈 技術分析",
        "📈 分點分析",
        "🧮 分點買賣超偵測",
        "🚨 分點異常偵測",
        "🕵️ 低檔潛伏分點掃描",
        "🧠 AI 贏家分點追蹤",
        "🏥 全方位籌碼診斷",
        "🔮 股價趨勢預測",
        "💎 基本面分析",
        "⚙️ 資料同步管理",
        "🧪 策略回測 (開發中)",
        "⚙️ 系統設定",
    ],
    index=3,
)

if main_menu == "📊 資料瀏覽器":
    data_browser.show_data_browser()
elif main_menu == "📈 技術分析":
    tech_analysis.show_tech_analysis()
elif main_menu == "📈 分點分析":
    branch_analysis.show_branch_analysis()
elif main_menu == "🧮 分點買賣超偵測":
    branch_net_detector_entry.show_branch_net_detector_entry()
elif main_menu == "🏥 全方位籌碼診斷":
    stock_health.show_stock_health()
elif main_menu == "🚨 分點異常偵測":
    branch_anomaly.show_branch_anomaly()
elif main_menu == "🕵️ 低檔潛伏分點掃描":
    branch_accumulation_scan.show_branch_accumulation_scan()
elif main_menu == "🧠 AI 贏家分點追蹤":
    winner_branch_system.show_winner_branch_system()
elif main_menu == "⚙️ 資料同步管理":
    data_management.show_data_management()
elif main_menu == "💎 基本面分析":
    fundamental_analysis.show_fundamental_analysis()
elif main_menu == "⚙️ 系統設定":
    st.header("⚙️ 系統組態與狀態")
    import database

    st.json(database.load_config())
    st.button("🔄 重新載入設定檔")
elif main_menu == "🔮 股價趨勢預測":
    prediction.show_prediction()
else:
    st.info(f"🛠️ 功能模組 `{main_menu}` 正在密集開發中...")
