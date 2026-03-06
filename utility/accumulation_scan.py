from dataclasses import dataclass

import pandas as pd


@dataclass
class AccumulationScanConfig:
    min_consecutive_days: int = 3
    min_buy_sell_ratio: float = 10.0
    min_volume_share: float = 0.05


def _prepare_scan_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in ("buy", "sell", "Trading_Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["net_buy"] = df["buy"] - df["sell"]

    # NOTE:
    # pandas nullable dtypes + pd.NA can raise TypeError on `.astype("float")`
    # in some versions. Keep arithmetic as numeric and coerce invalid values.
    sell_nonzero = df["sell"].replace(0, float("nan"))
    df["buy_sell_ratio"] = pd.to_numeric(df["buy"] / sell_nonzero, errors="coerce")
    df["buy_sell_ratio"] = df["buy_sell_ratio"].fillna(float("inf"))

    vol_nonzero = df["Trading_Volume"].replace(0, float("nan"))
    df["volume_share"] = pd.to_numeric(df["net_buy"] / vol_nonzero, errors="coerce").fillna(0.0)
    return df


def summarize_accumulation_filters(raw_df: pd.DataFrame, cfg: AccumulationScanConfig) -> dict:
    """Return filter-stage diagnostics so UI can explain why results are empty."""
    df = _prepare_scan_frame(raw_df)
    if df.empty:
        return {
            "raw_rows": 0,
            "stock_count": 0,
            "branch_count": 0,
            "trade_days": 0,
            "net_buy_rows": 0,
            "ratio_rows": 0,
            "share_rows": 0,
            "all_rules_rows": 0,
            "missing_ohlcv_rows": 0,
        }

    net_buy_mask = df["net_buy"] > 0
    ratio_mask = df["buy_sell_ratio"] > float(cfg.min_buy_sell_ratio)
    share_mask = df["volume_share"] >= float(cfg.min_volume_share)
    all_mask = net_buy_mask & ratio_mask & share_mask

    return {
        "raw_rows": int(len(df)),
        "stock_count": int(df["stock_id"].nunique()) if "stock_id" in df.columns else 0,
        "branch_count": int(df["branch_id"].nunique()) if "branch_id" in df.columns else 0,
        "trade_days": int(df["date"].nunique()),
        "net_buy_rows": int(net_buy_mask.sum()),
        "ratio_rows": int(ratio_mask.sum()),
        "share_rows": int(share_mask.sum()),
        "all_rules_rows": int(all_mask.sum()),
        "missing_ohlcv_rows": int(df.get("missing_ohlcv", pd.Series(dtype=int)).fillna(0).sum()),
    }


def run_accumulation_scan(raw_df: pd.DataFrame, cfg: AccumulationScanConfig) -> pd.DataFrame:
    df = _prepare_scan_frame(raw_df)
    if df.empty:
        return pd.DataFrame()

    cond = (
        (df["net_buy"] > 0)
        & (df["buy_sell_ratio"] > float(cfg.min_buy_sell_ratio))
        & (df["volume_share"] >= float(cfg.min_volume_share))
    )
    df["is_accumulation_day"] = cond

    candidates = []
    grouped = df.sort_values(["stock_id", "branch_id", "date"]).groupby(["stock_id", "branch_id", "branch_name"], dropna=False)

    for (stock_id, branch_id, branch_name), g in grouped:
        g = g.reset_index(drop=True)
        mask = g["is_accumulation_day"].astype(bool)
        if not mask.any():
            continue

        streak_id = (mask != mask.shift(fill_value=False)).cumsum()
        runs = g[mask].groupby(streak_id[mask])

        valid_runs = []
        for _, r in runs:
            streak_days = int(len(r))
            if streak_days < cfg.min_consecutive_days:
                continue
            valid_runs.append(
                {
                    "start_date": r["date"].min(),
                    "end_date": r["date"].max(),
                    "consecutive_days": streak_days,
                    "net_buy_total": float(r["net_buy"].sum()),
                    "avg_buy_sell_ratio": float(r["buy_sell_ratio"].replace(float("inf"), pd.NA).mean(skipna=True) or 9999.0),
                    "avg_volume_share": float(r["volume_share"].mean()),
                }
            )

        if not valid_runs:
            continue

        best_run = max(valid_runs, key=lambda x: (x["consecutive_days"], x["net_buy_total"]))
        candidates.append(
            {
                "stock_id": stock_id,
                "branch_id": branch_id,
                "branch_name": branch_name,
                **best_run,
                "signal_count": len(valid_runs),
                "latest_signal_end": max(x["end_date"] for x in valid_runs),
            }
        )

    if not candidates:
        return pd.DataFrame()

    out = pd.DataFrame(candidates).sort_values(
        ["consecutive_days", "avg_volume_share", "net_buy_total", "latest_signal_end"],
        ascending=[False, False, False, False],
    )
    return out.reset_index(drop=True)
