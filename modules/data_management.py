import streamlit as st
from pathlib import Path
from datetime import date, timedelta
import sqlite3

import database
import ingest_manager
import ingest_minute
from utility.finmind_branch_collector import run_collection, refresh_trader_info


def _parse_iso_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return fallback


def _get_finmind_token(cfg: dict) -> str:
    return str(((cfg.get("finmind") or {}).get("api_token") or "")).strip()


def _load_branch_ids_from_db(sqlite_path: str) -> list[str]:
    if not sqlite_path or not sqlite_path.strip():
        return []

    conn = sqlite3.connect(sqlite_path.strip())
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT securities_trader_id
            FROM securities_trader_info
            WHERE securities_trader_id IS NOT NULL
              AND TRIM(securities_trader_id) <> ''
            ORDER BY securities_trader_id
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def show_data_management():
    st.header("⚙️ 資料同步管理中心")
    cfg = database.load_config()
    finmind_token = _get_finmind_token(cfg)

    task_type = st.radio(
        "請選擇要啟動的執行案件：",
        [
            "📅 每日 13 項指標 (原 Ingest Manager)",
            "⏱️ 分鐘與主動力度 (新 Ingest Minute)",
            "🏦 FinMind 分點基本資料下載",
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

    elif task_type == "🏦 FinMind 分點基本資料下載":
        st.subheader("📋 案件：手動下載分點基本資料")

        default_db = (cfg.get("storage") or {}).get("sqlite_path", "data/stock.db")
        branch_sync_cfg = cfg.get("branch_sync") or {}
        default_sleep = float(branch_sync_cfg.get("sleep_seconds", 0.2) or 0.2)
        default_max_retries = int(branch_sync_cfg.get("max_retries", 2) or 2)
        default_retry_sleep = float(branch_sync_cfg.get("retry_sleep_seconds", 1.0) or 1.0)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("FinMind Token 由 config.json 的 finmind.api_token 自動帶入")
            download_mode = st.radio(
                "下載模式",
                ["全部分點", "指定分點"],
                horizontal=True,
                key="trader_info_mode",
            )
            branch_ids_raw = st.text_area(
                "指定分點代碼 (逗號分隔)",
                value="1102,1160",
                disabled=(download_mode != "指定分點"),
                key="trader_info_branch_ids",
            )
            raw_dir = st.text_input("Raw 資料目錄", value="data/branch_raw", key="trader_info_raw_dir")

        with col2:
            sqlite_path = st.text_input("SQLite 路徑", value=default_db, key="trader_info_sqlite")
            sleep_sec = st.number_input(
                "每次呼叫間隔(秒)",
                min_value=0.0,
                max_value=5.0,
                value=max(0.0, min(5.0, default_sleep)),
                step=0.1,
                key="trader_info_sleep",
            )
            max_retries = st.number_input(
                "API 失敗重試次數",
                min_value=0,
                max_value=10,
                value=max(0, min(10, default_max_retries)),
                step=1,
                key="trader_info_retries",
            )
            retry_sleep_sec = st.number_input(
                "API 重試等待(秒)",
                min_value=0.0,
                max_value=30.0,
                value=max(0.0, min(30.0, default_retry_sleep)),
                step=0.5,
                key="trader_info_retry_sleep",
            )

        if st.button("🚀 啟動分點基本資料下載", use_container_width=True):
            if not finmind_token:
                st.error("config.json 缺少 finmind.api_token，請先設定。")
                return

            branch_ids = None
            if download_mode == "指定分點":
                branch_ids = [x.strip() for x in branch_ids_raw.split(",") if x.strip()]
                if not branch_ids:
                    st.error("指定分點模式下，請至少輸入一個分點代碼。")
                    return

            progress = st.empty()
            with st.spinner("分點基本資料下載中..."):
                trader_df = refresh_trader_info(
                    token=finmind_token,
                    sqlite_path=Path(sqlite_path) if sqlite_path.strip() else None,
                    raw_dir=Path(raw_dir),
                    branch_ids=branch_ids,
                    sleep_sec=float(sleep_sec),
                    max_retries=int(max_retries),
                    retry_sleep_sec=float(retry_sleep_sec),
                    progress_callback=lambda msg: progress.info(msg),
                )

            if trader_df is None or trader_df.empty:
                st.warning("完成，但 API 回傳 0 筆分點基本資料。")
            else:
                st.success(f"完成：已下載/更新 {len(trader_df)} 筆分點基本資料")
                st.dataframe(trader_df.tail(500), use_container_width=True)

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
        default_max_retries = int(branch_sync_cfg.get("max_retries", 2) or 2)
        default_retry_sleep = float(branch_sync_cfg.get("retry_sleep_seconds", 1.0) or 1.0)
        default_commit_interval = int(branch_sync_cfg.get("commit_interval", 100) or 100)
        default_write_raw_csv = bool(branch_sync_cfg.get("write_raw_csv", False))
        retry_notrade_days = int(branch_sync_cfg.get("retry_notrade_days", (cfg.get("ingest") or {}).get("retry_notrade_days", 14)))
        default_refresh_info = bool(branch_sync_cfg.get("refresh_trader_info", False))
        default_recent_mode = bool(branch_sync_cfg.get("recent_only_mode", True))
        default_lookback_days = int(branch_sync_cfg.get("recent_lookback_days", 3) or 3)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("FinMind Token 由 config.json 的 finmind.api_token 自動帶入")
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
            refresh_trader_info_flag = st.checkbox(
                "執行前清空分點基本資料後重建",
                value=default_refresh_info,
                help="勾選後會先清空 securities_trader_info，再重新抓取分點基本資料。",
            )
            recent_only_mode = st.checkbox(
                "快速增量模式（僅同步今天 + 最近回補天數）",
                value=default_recent_mode,
                help="開啟後會忽略上方開始/結束日期，改為自動同步 [今天-回補天數, 今天]，避免每次都從很早日期全量掃描。",
            )
            lookback_days = st.number_input(
                "快速增量模式回補天數",
                min_value=0,
                max_value=30,
                value=max(0, min(30, default_lookback_days)),
                step=1,
                disabled=not recent_only_mode,
                help="例如填 3 代表同步今天與前 3 天（共 4 天）。",
            )
            max_retries = st.number_input(
                "API 失敗重試次數",
                min_value=0,
                max_value=10,
                value=max(0, min(10, default_max_retries)),
                step=1,
                key="branch_detail_retries",
            )
            retry_sleep_sec = st.number_input(
                "API 重試等待(秒)",
                min_value=0.0,
                max_value=30.0,
                value=max(0.0, min(30.0, default_retry_sleep)),
                step=0.5,
                key="branch_detail_retry_sleep",
            )
            commit_interval = st.number_input(
                "資料庫批次提交筆數",
                min_value=1,
                max_value=1000,
                value=max(1, min(1000, default_commit_interval)),
                step=10,
                help="每累積 N 筆再 commit，一般可大幅減少 SQLite I/O 時間。",
            )
            write_raw_csv = st.checkbox(
                "同步時輸出 raw csv（較慢）",
                value=default_write_raw_csv,
                help="預設關閉。若僅需同步進 SQLite，建議關閉以減少磁碟 I/O 並加速。",
            )

        branch_ids = _load_branch_ids_from_db(sqlite_path)
        if branch_ids:
            st.info(f"將使用資料庫中的全部分點，共 {len(branch_ids)} 個。")
        else:
            st.warning("目前資料庫查無分點代碼，請先下載分點基本資料。")

        if st.button("🚀 啟動分點明細同步", use_container_width=True):
            if not finmind_token:
                st.error("config.json 缺少 finmind.api_token，請先設定。")
                return
            if not branch_ids:
                st.error("資料庫查無分點代碼，請先下載分點基本資料。")
                return
            if start_d > end_d:
                st.error("開始日期不可晚於結束日期。")
                return

            if recent_only_mode:
                effective_end_d = date.today()
                effective_start_d = effective_end_d - timedelta(days=int(lookback_days))
                st.info(
                    f"已啟用快速增量模式：同步區間 {effective_start_d.isoformat()} ~ {effective_end_d.isoformat()}"
                )
            else:
                effective_start_d = start_d
                effective_end_d = end_d

            progress = st.empty()
            with st.spinner("分點明細同步中..."):
                stats = run_collection(
                    token=finmind_token,
                    branch_ids=branch_ids,
                    start_date=effective_start_d.isoformat(),
                    end_date=effective_end_d.isoformat(),
                    raw_dir=Path(raw_dir),
                    sqlite_path=Path(sqlite_path) if sqlite_path.strip() else None,
                    sleep_sec=float(sleep_sec),
                    retry_notrade_days=retry_notrade_days,
                    refresh_trader_info=refresh_trader_info_flag,
                    max_retries=int(max_retries),
                    retry_sleep_sec=float(retry_sleep_sec),
                    commit_interval=int(commit_interval),
                    write_raw_csv=write_raw_csv,
                    progress_callback=lambda msg: progress.info(msg),
                )

            ok = int((stats["status"] == "Success").sum()) if not stats.empty else 0
            err = int((stats["status"] == "Failed").sum()) if not stats.empty else 0
            st.success(f"同步完成：成功 {ok} 筆，失敗 {err} 筆")
            st.dataframe(stats.tail(200), use_container_width=True)
