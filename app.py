import streamlit as st
# 匯入所有功能模組，包含新加入的 data_management
from modules import data_browser, tech_analysis, branch_analysis, stock_health, data_management, prediction, fundamental_analysis, branch_anomaly, winner_branch_system
##--- 頁面配置 ---
st.set_page_config(page_title="台股 AI 自動化分析平台", layout="wide")

# --- 側邊導覽選單 ---
st.sidebar.title("🚀 功能選單")
main_menu = st.sidebar.radio(
    "請選擇功能模組：",
    [
        "📊 資料瀏覽器",
        "📈 技術分析",
        "📈 分點分析",
        "🚨 分點異常偵測",
        "🧠 AI 贏家分點追蹤",
        "🏥 全方位籌碼診斷",
        "🔮 股價趨勢預測",
        "💎 基本面分析",
        "⚙️ 資料同步管理", # 這裡是執行 ingest 案件的入口
        "🧪 策略回測 (開發中)",
        "⚙️ 系統設定"
    ],
    index=3 # 預設選中診斷頁面
)

# --- 根據選單切換模組 ---
if main_menu == "📊 資料瀏覽器":
    data_browser.show_data_browser()

elif main_menu == "📈 技術分析":
    tech_analysis.show_tech_analysis()

elif main_menu == "📈 分點分析":
    branch_analysis.show_branch_analysis()

elif main_menu == "🏥 全方位籌碼診斷":
    stock_health.show_stock_health()

elif main_menu == "🚨 分點異常偵測":
    branch_anomaly.show_branch_anomaly()

elif main_menu == "🧠 AI 贏家分點追蹤":
    winner_branch_system.show_winner_branch_system()

elif main_menu == "⚙️ 資料同步管理":
    # 呼叫管理中心，讓您選擇要執行的 Ingest 腳本
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
