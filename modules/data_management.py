import streamlit as st

import database
import ingest_manager
import ingest_minute


def show_data_management():
    st.header("⚙️ 資料同步管理中心")
    cfg = database.load_config()

    task_type = st.radio(
        "請選擇要啟動的執行案件：",
        ["📅 每日 13 項指標 (原 Ingest Manager)", "⏱️ 分鐘與主動力度 (新 Ingest Minute)"],
        horizontal=True,
    )

    st.divider()

    if task_type == "📅 每日 13 項指標 (原 Ingest Manager)":
        st.subheader("📋 案件：標準日線指標同步")
        if st.button("🔥 啟動全方位數據同步", use_container_width=True):
            with st.spinner("同步進行中..."):
                try:
                    log_area = st.empty()
                    failed_log = ingest_manager.start_ingest(st_placeholder=log_area)

                    if failed_log:
                        st.warning(f"同步完成，但有 {len(failed_log)} 個錯誤。")
                        with st.expander("查看錯誤明細"):
                            for msg in failed_log:
                                st.write(f"❌ {msg}")
                    else:
                        st.success("✅ 所有日線指標同步成功！")
                except Exception as e:
                    st.error(f"💥 程式執行中斷 (嚴重錯誤)：{e}")

    elif task_type == "⏱️ 分鐘與主動力度 (新 Ingest Minute)":
        st.subheader("📋 案件：分鐘 K 線與主動力度補洞")
        if st.button("🚀 啟動分鐘級數據補洞 (含 A/B 對帳)", use_container_width=True):
            ingest_minute.run_minute_task(cfg)
