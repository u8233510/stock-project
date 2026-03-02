import streamlit as st
from pathlib import Path
from datetime import date, timedelta

import database
import ingest_manager
import ingest_minute
from utility.finmind_branch_collector import run_collection


def _parse_iso_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return fallback


def show_data_management():
    st.header("⚙️ 資料同步管理中心")
    cfg = database.load_config()

    task_type = st.radio(
        "請選擇要啟動的執行案件：",
        [
            "📅 每日 13 項指標 (原 Ingest Manager)",
            "⏱️ 分鐘與主動力度 (新 Ingest Minute)",
            "🏦 FinMind 分點明細同步 (分點+日期)",
        ],
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

    elif task_type == "🏦 FinMind 分點明細同步 (分點+日期)":
        st.subheader("📋 案件：依分點代碼 + 日期下載交易明細")

        default_db = (cfg.get("storage") or {}).get("sqlite_path", "data/stock.db")
        branch_sync_cfg = cfg.get("branch_sync") or {}
        default_end = _parse_iso_date(branch_sync_cfg.get("end_date"), date.today())
        default_start = _parse_iso_date(
            branch_sync_cfg.get("start_date"),
            default_end - timedelta(days=7),
        )
        default_sleep = float(branch_sync_cfg.get("sleep_seconds", 0.2) or 0.2)

        col1, col2 = st.columns(2)
        with col1:
            token = st.text_input("FinMind Token", type="password")
            branch_ids_raw = st.text_area(
                "分點代碼 (逗號分隔)",
                value="1102,1160",
                help="例如: 1102,1160,7000",
            )
            raw_dir = st.text_input("Raw 資料目錄", value="data/branch_raw")
        with col2:
            end_d = st.date_input("結束日期", value=default_end)
            start_d = st.date_input("開始日期", value=default_start)
            sqlite_path = st.text_input("SQLite 路徑", value=default_db)
            sleep_sec = st.number_input(
                "每次呼叫間隔(秒)",
                min_value=0.0,
                max_value=5.0,
                value=max(0.0, min(5.0, default_sleep)),
                step=0.1,
            )

        if st.button("🚀 啟動分點明細同步", use_container_width=True):
            branch_ids = [x.strip() for x in branch_ids_raw.split(",") if x.strip()]
            if not token.strip():
                st.error("請先輸入 FinMind Token。")
                return
            if not branch_ids:
                st.error("請至少輸入一個分點代碼。")
                return
            if start_d > end_d:
                st.error("開始日期不可晚於結束日期。")
                return

            progress = st.empty()
            with st.spinner("分點明細同步中..."):
                stats = run_collection(
                    token=token.strip(),
                    branch_ids=branch_ids,
                    start_date=start_d.isoformat(),
                    end_date=end_d.isoformat(),
                    raw_dir=Path(raw_dir),
                    sqlite_path=Path(sqlite_path) if sqlite_path.strip() else None,
                    sleep_sec=float(sleep_sec),
                    progress_callback=lambda msg: progress.info(msg),
                )

            ok = int((stats["status"] == "ok").sum()) if not stats.empty else 0
            err = int((stats["status"] == "error").sum()) if not stats.empty else 0
            st.success(f"同步完成：成功 {ok} 筆，失敗 {err} 筆")
            st.dataframe(stats.tail(200), use_container_width=True)
