import pandas as pd
import streamlit as st

import database


COLUMN_RENAME = {
    "date": "日期",
    "securities_trader_id": "分點代號",
    "securities_trader": "分點名稱",
    "stock_id": "股票代號",
    "buy": "買進量",
    "sell": "賣出量",
    "net": "淨買賣量",
    "gross": "總成交量",
    "buy_score": "買進強度分數",
    "sell_score": "賣出強度分數",
    "net_ratio": "淨量占比",
    "gross_z": "量能異常分數",
    "is_buy_streak_3": "連3日買超同股",
    "is_sell_streak_3": "連3日賣超同股",
}


def _load_detail(conn, start_date: str, end_date: str) -> pd.DataFrame:
    sql = """
    SELECT
        date,
        securities_trader_id,
        COALESCE(securities_trader, '') AS securities_trader,
        stock_id,
        SUM(COALESCE(buy, 0)) AS buy,
        SUM(COALESCE(sell, 0)) AS sell
    FROM branch_trader_daily_detail
    WHERE date >= ?
      AND date <= ?
    GROUP BY date, securities_trader_id, securities_trader, stock_id
    ORDER BY date, securities_trader_id, stock_id
    """
    return pd.read_sql(sql, conn, params=[start_date, end_date])


def _compute_streak_flags(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    target_ts = pd.to_datetime(target_date)

    def _max_consecutive(sub_df: pd.DataFrame, direction: str) -> int:
        direction_series = (sub_df["net"] > 0) if direction == "buy" else (sub_df["net"] < 0)
        streak = 0
        for is_match in direction_series.iloc[::-1]:
            if is_match:
                streak += 1
            else:
                break
        return streak

    grouped = out[out["date"] <= target_ts].sort_values("date").groupby(["securities_trader_id", "stock_id"], as_index=False)

    streak_rows = []
    for _, g in grouped:
        latest_date = g["date"].max()
        if latest_date != target_ts:
            continue
        streak_rows.append(
            {
                "securities_trader_id": g.iloc[-1]["securities_trader_id"],
                "stock_id": g.iloc[-1]["stock_id"],
                "is_buy_streak_3": _max_consecutive(g, "buy") >= 3,
                "is_sell_streak_3": _max_consecutive(g, "sell") >= 3,
            }
        )

    if not streak_rows:
        return pd.DataFrame(columns=["securities_trader_id", "stock_id", "is_buy_streak_3", "is_sell_streak_3"])
    return pd.DataFrame(streak_rows)


def _score_by_baseline(hist_df: pd.DataFrame, target_date: str, min_gross: int) -> pd.DataFrame:
    work = hist_df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["net"] = work["buy"] - work["sell"]
    work["gross"] = work["buy"] + work["sell"]

    target_ts = pd.to_datetime(target_date)
    daily = work[work["date"] == target_ts].copy()
    if daily.empty:
        return daily

    baseline = work[work["date"] < target_ts].copy()

    if baseline.empty:
        daily["gross_z"] = 0.0
    else:
        stats = baseline.groupby(["securities_trader_id", "stock_id"])["gross"].agg(["median", "mad"]).reset_index()
        stats["mad"] = stats["mad"].replace(0, pd.NA)
        daily = daily.merge(stats, on=["securities_trader_id", "stock_id"], how="left")
        daily["gross_z"] = ((daily["gross"] - daily["median"].fillna(0)) / (1.4826 * daily["mad"]))
        daily["gross_z"] = daily["gross_z"].replace([float("inf"), float("-inf")], 0).fillna(0.0)
        daily.drop(columns=["median", "mad"], inplace=True)

    streak_df = _compute_streak_flags(work, target_date)
    daily = daily.merge(streak_df, on=["securities_trader_id", "stock_id"], how="left")
    daily["is_buy_streak_3"] = daily["is_buy_streak_3"].fillna(False)
    daily["is_sell_streak_3"] = daily["is_sell_streak_3"].fillna(False)

    daily["net_ratio"] = (daily["net"] / daily["gross"].replace(0, pd.NA)).fillna(0.0)
    gross_signal = daily["gross_z"].clip(lower=0, upper=5) / 5

    daily["buy_score"] = (
        (daily["net_ratio"].clip(lower=0, upper=1) * 60)
        + (gross_signal * 25)
        + (daily["is_buy_streak_3"].astype(int) * 15)
    )
    daily["sell_score"] = (
        (daily["net_ratio"].clip(lower=-1, upper=0).abs() * 60)
        + (gross_signal * 25)
        + (daily["is_sell_streak_3"].astype(int) * 15)
    )

    daily = daily[daily["gross"] >= min_gross].copy()
    return daily


def _render_table(df: pd.DataFrame, columns: list[str], top_n: int):
    view = df[columns].head(top_n).rename(columns=COLUMN_RENAME)
    fmt = {
        "買進強度分數": "{:.1f}",
        "賣出強度分數": "{:.1f}",
        "淨量占比": "{:.2%}",
        "量能異常分數": "{:.2f}",
    }
    return view.style.format({k: v for k, v in fmt.items() if k in view.columns})


def _build_branch_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby(["securities_trader_id", "securities_trader"], as_index=False)
        .agg(
            gross=("gross", "sum"),
            net=("net", "sum"),
            buy_signal_count=("buy_score", lambda s: int((s > 0).sum())),
            sell_signal_count=("sell_score", lambda s: int((s > 0).sum())),
            buy_score_total=("buy_score", "sum"),
            sell_score_total=("sell_score", "sum"),
        )
    )
    summary["net_ratio"] = (summary["net"] / summary["gross"].replace(0, pd.NA)).fillna(0.0)
    return summary


def show_branch_flow_scanner():
    st.markdown("### 🔍 分點每日買賣掃描（獨立於既有異常偵測）")
    st.caption("專注各個分點：先找出當日最積極的分點，再看這些分點正在買/賣哪些股票。")

    cfg = database.load_config()
    conn = database.get_db_connection(cfg)

    bounds = conn.execute("SELECT MIN(date), MAX(date) FROM branch_trader_daily_detail").fetchone()
    if not bounds or not bounds[0] or not bounds[1]:
        st.info("目前 branch_trader_daily_detail 尚無資料。")
        return

    min_date = pd.to_datetime(bounds[0]).date()
    max_date = pd.to_datetime(bounds[1]).date()

    c1, c2, c3 = st.columns(3)
    with c1:
        target_date = st.date_input("目標交易日", value=max_date, min_value=min_date, max_value=max_date)
    with c2:
        lookback_days = st.slider("基準回看天數", min_value=5, max_value=60, value=10, step=1)
    with c3:
        top_n = st.number_input("TopN（前幾大分點）", min_value=5, max_value=200, value=20, step=5)

    min_gross = st.number_input("最小總成交量過濾", min_value=0, max_value=200000, value=200, step=100)

    run = st.button("開始掃描", type="primary", use_container_width=True)
    if not run:
        return

    target_ts = pd.to_datetime(target_date)
    start_ts = target_ts - pd.Timedelta(days=int(lookback_days) * 2)

    hist_df = _load_detail(conn, start_ts.strftime("%Y-%m-%d"), target_ts.strftime("%Y-%m-%d"))
    if hist_df.empty:
        st.warning("查無符合條件的分點明細資料。")
        return

    scored = _score_by_baseline(hist_df, target_ts.strftime("%Y-%m-%d"), min_gross=int(min_gross))
    if scored.empty:
        st.info("目標日期無符合門檻的訊號。")
        return

    branch_summary = _build_branch_summary(scored)
    buy_branch_rank = branch_summary.sort_values(["buy_score_total", "gross"], ascending=[False, False]).head(int(top_n))
    sell_branch_rank = branch_summary.sort_values(["sell_score_total", "gross"], ascending=[False, False]).head(int(top_n))
    focus_branch_ids = sorted(set(buy_branch_rank["securities_trader_id"]).union(set(sell_branch_rank["securities_trader_id"])))

    st.success(
        f"掃描完成：{target_ts.date()} 共 {len(scored)} 筆分點-股票訊號，聚焦 {len(focus_branch_ids)} 個重點分點。"
    )

    t1, t2, t3, t4 = st.tabs(["買進重點分點", "賣出重點分點", "重點分點買入股票", "重點分點賣出股票"])

    with t1:
        st.dataframe(
            buy_branch_rank.rename(
                columns={
                    "securities_trader_id": "分點代號",
                    "securities_trader": "分點名稱",
                    "gross": "總成交量",
                    "net": "淨買賣量",
                    "net_ratio": "淨量占比",
                    "buy_signal_count": "買進訊號檔數",
                    "buy_score_total": "買進總分",
                }
            )[
                ["分點代號", "分點名稱", "總成交量", "淨買賣量", "淨量占比", "買進訊號檔數", "買進總分"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with t2:
        st.dataframe(
            sell_branch_rank.rename(
                columns={
                    "securities_trader_id": "分點代號",
                    "securities_trader": "分點名稱",
                    "gross": "總成交量",
                    "net": "淨買賣量",
                    "net_ratio": "淨量占比",
                    "sell_signal_count": "賣出訊號檔數",
                    "sell_score_total": "賣出總分",
                }
            )[
                ["分點代號", "分點名稱", "總成交量", "淨買賣量", "淨量占比", "賣出訊號檔數", "賣出總分"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    branch_scored = scored[scored["securities_trader_id"].isin(focus_branch_ids)].copy()
    buy_rank = branch_scored.sort_values(["buy_score", "gross", "net"], ascending=[False, False, False])
    sell_rank = branch_scored.sort_values(["sell_score", "gross", "net"], ascending=[False, False, True])

    buy_cols = [
        "date",
        "securities_trader_id",
        "securities_trader",
        "stock_id",
        "buy",
        "sell",
        "net",
        "gross",
        "net_ratio",
        "gross_z",
        "buy_score",
        "is_buy_streak_3",
    ]
    sell_cols = [
        "date",
        "securities_trader_id",
        "securities_trader",
        "stock_id",
        "buy",
        "sell",
        "net",
        "gross",
        "net_ratio",
        "gross_z",
        "sell_score",
        "is_sell_streak_3",
    ]

    with t3:
        st.caption("已加上：同一分點連續 3 天買超同一檔股票標記。")
        st.dataframe(_render_table(buy_rank, buy_cols, int(top_n)), use_container_width=True, hide_index=True)

    with t4:
        st.caption("已加上：同一分點連續 3 天賣超同一檔股票標記。")
        st.dataframe(_render_table(sell_rank, sell_cols, int(top_n)), use_container_width=True, hide_index=True)
