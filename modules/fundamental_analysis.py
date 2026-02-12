from datetime import datetime

import streamlit as st
try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS

import database


def render_stock_news(sid: str, sname: str):
    """
    主要渲染函數：搜尋並顯示指定股票的最新 10 則新聞
    :param sid: 股票代碼 (例如: 2330)
    :param sname: 股票名稱 (例如: 台積電)
    """
    st.subheader(f"🌐 {sname} ({sid}) 最新相關新聞")

    # 1. 建立搜尋關鍵字（多組 fallback，避免單一 query 無資料）
    queries = [
        f"{sname} {sid} 股票 新聞",
        f"{sname} 股票 新聞",
        f"{sid} 股票 新聞",
    ]

    # timelimit 直接使用「年」
    timelimit = "y"

    try:
        with st.spinner("正在從網路搜尋最新動態..."):
            news_list = []
            # 2. 使用 DuckDuckGo 進行新聞搜尋（逐步放寬條件）
            with DDGS() as ddgs:
                for query in queries:
                    results = ddgs.news(
                        keywords=query,
                        region="wt-wt",
                        safesearch="off",
                        timelimit=timelimit,
                        max_results=10,
                    )
                    fetched = list(results)
                    if fetched:
                        news_list = fetched
                        break

        # 3. 呈現搜尋結果
        if not news_list:
            st.warning("目前找不到相關新聞（已嘗試代碼/名稱、近一年範圍），請稍後再試。")
            return

        for news in news_list[:10]:
            # 建立一個美觀的容器顯示每一則新聞
            with st.container():
                col1, col2 = st.columns([1, 4])

                # 顯示來源與日期
                with col1:
                    source = news.get("source", "新聞來源")
                    date_str = news.get("date", "")
                    # 格式化日期顯示
                    if date_str:
                        try:
                            # 簡化日期格式
                            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            st.caption(f"📅 {dt.strftime('%m/%d %H:%M')}")
                        except Exception:
                            st.caption(date_str)
                    st.info(f"📍 {source}")

                # 顯示標題與連結
                with col2:
                    title = news.get("title", "(無標題)")
                    url = news.get("url") or news.get("href", "")
                    snippet = news.get("body", "點擊標題查看完整內容...")
                    if url:
                        st.markdown(f"#### [{title}]({url})")
                    else:
                        st.markdown(f"#### {title}")
                    st.write(f"{snippet[:150]}...")

                st.divider()

    except Exception as e:
        st.error(f"搜尋新聞時發生錯誤：{str(e)}")
        st.info("建議檢查網路連線，或稍後再試。")


def show_fundamental_analysis():
    """保持與 app.py 相容的入口函數。"""
    st.markdown("### 💎 基本面分析（新聞）")

    cfg = database.load_config()
    universe = cfg.get("universe", [])
    if not universe:
        st.error("universe 未設定，請先在設定檔配置標的。")
        return

    stock_options = {f"{s['stock_id']} {s['name']}": (s["stock_id"], s["name"]) for s in universe}
    selected_label = st.selectbox("選擇股票", list(stock_options.keys()))
    sid, sname = stock_options[selected_label]

    if st.button("🔍 搜尋最新新聞", use_container_width=True):
        render_stock_news(sid, sname)


# 如果此程式被當作主程式執行 (測試用)
if __name__ == "__main__":
    # 這裡的 sid 與 sname 通常由您的 app.py 選取後傳入
    # 範例測試：
    st.set_page_config(page_title="股票新聞搜尋", layout="wide")
    test_sid = "2330"
    test_sname = "台積電"
    render_stock_news(test_sid, test_sname)
