"""Branch-level anomaly detection utilities.

This module converts branch buy/sell/price records into:
1) anomaly-ranked events,
2) tradable watchlist,
3) weekly validation report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BranchAnomalyConfig:
    short_window: int = 20
    long_window: int = 60
    z_threshold: float = 3.0
    gross_z_threshold: float = 3.0
    vol_share_threshold: float = 0.10
    major_score_threshold: float = 80.0
    medium_score_threshold: float = 60.0


def _validate_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_branch_daily_features(
    raw_df: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    branch_col: str = "branch_id",
    price_col: str = "price",
    buy_col: str = "buy",
    sell_col: str = "sell",
    close_col: str = "close",
) -> pd.DataFrame:
    """Build daily features at (stock, date, branch) granularity.

    Expected columns in raw_df:
    - stock_id/date/branch_id/price/buy/sell/close
    """
    _validate_columns(raw_df, [stock_col, date_col, branch_col, price_col, buy_col, sell_col, close_col])

    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[buy_col] = pd.to_numeric(df[buy_col], errors="coerce").fillna(0.0)
    df[sell_col] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0.0)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce").fillna(0.0)

    df["trade_volume"] = df[buy_col] + df[sell_col]
    df["price_x_volume"] = df[price_col] * df["trade_volume"]

    grouped = (
        df.groupby([stock_col, date_col, branch_col], as_index=False)
        .agg(
            buy=(buy_col, "sum"),
            sell=(sell_col, "sum"),
            gross_vol=("trade_volume", "sum"),
            price_x_volume=("price_x_volume", "sum"),
            close=(close_col, "last"),
        )
    )
    grouped["net_vol"] = grouped["buy"] - grouped["sell"]
    grouped["buy_ratio"] = _safe_div(grouped["buy"], grouped["gross_vol"])
    grouped["avg_price"] = _safe_div(grouped["price_x_volume"], grouped["gross_vol"])
    grouped["price_impact"] = _safe_div(grouped["avg_price"] - grouped["close"], grouped["close"])

    stock_day_total = grouped.groupby([stock_col, date_col], as_index=False)["gross_vol"].sum().rename(
        columns={"gross_vol": "stock_day_gross_vol"}
    )
    grouped = grouped.merge(stock_day_total, on=[stock_col, date_col], how="left")
    grouped["vol_share"] = _safe_div(grouped["gross_vol"], grouped["stock_day_gross_vol"])
    grouped["net_share"] = _safe_div(grouped["net_vol"], grouped["stock_day_gross_vol"])

    return grouped.sort_values([stock_col, branch_col, date_col]).reset_index(drop=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=max(5, window // 2)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_rolling_baselines(
    features_df: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    branch_col: str = "branch_id",
    windows: Iterable[int] = (20, 60),
) -> pd.DataFrame:
    """Add rolling z-score baselines for key signals per (stock, branch)."""
    _validate_columns(features_df, [stock_col, date_col, branch_col, "net_vol", "gross_vol", "price_impact"])
    df = features_df.copy().sort_values([stock_col, branch_col, date_col])

    for window in windows:
        grouped = df.groupby([stock_col, branch_col], group_keys=False)
        df[f"z_net_{window}"] = grouped["net_vol"].apply(lambda s: _rolling_zscore(s, window))
        df[f"z_gross_{window}"] = grouped["gross_vol"].apply(lambda s: _rolling_zscore(s, window))
        df[f"z_price_impact_{window}"] = grouped["price_impact"].apply(lambda s: _rolling_zscore(s, window))

    return df


def detect_rule_flags(df: pd.DataFrame, cfg: BranchAnomalyConfig = BranchAnomalyConfig()) -> pd.DataFrame:
    """Apply rule-based anomaly flags."""
    _validate_columns(df, ["z_net_20", "z_gross_20", "vol_share", "net_vol", "avg_price", "close"])
    out = df.copy()

    out["flag_net"] = out["z_net_20"].abs() >= cfg.z_threshold
    out["flag_gross"] = out["z_gross_20"] >= cfg.gross_z_threshold
    out["flag_share"] = out["vol_share"] >= cfg.vol_share_threshold

    out["flag_accumulation"] = (
        (out["z_net_20"] > cfg.z_threshold) & (out["vol_share"] > cfg.vol_share_threshold) & (out["net_vol"] > 0)
    )
    out["flag_distribution"] = (
        (out["z_net_20"] < -cfg.z_threshold)
        & (out["vol_share"] > cfg.vol_share_threshold)
        & (out["avg_price"] < out["close"])
    )

    out["rule_hit_count"] = out[["flag_net", "flag_gross", "flag_share"]].sum(axis=1)
    out["rule_triggered"] = out["rule_hit_count"] >= 2
    return out


def compute_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute anomaly score (0~100)."""
    _validate_columns(df, ["z_net_20", "z_gross_20", "z_price_impact_20", "vol_share"])
    out = df.copy()

    score_volume = np.clip((out["z_net_20"].abs() + out["z_gross_20"].clip(lower=0)) * 12.5, 0, 100)
    score_price = np.clip(out["z_price_impact_20"].abs() * 18.0, 0, 100)
    score_concentration = np.clip(out["vol_share"] * 500.0, 0, 100)

    # persistence proxy: combine direction consistency with volume share
    direction_consistency = np.where(out["net_vol"] >= 0, 1.0, 1.0)
    score_persistence = np.clip((out["z_net_20"].abs() * out["vol_share"] * 120.0 * direction_consistency), 0, 100)

    out["score_volume"] = score_volume
    out["score_price"] = score_price
    out["score_concentration"] = score_concentration
    out["score_persistence"] = score_persistence

    out["anomaly_score"] = (
        out["score_volume"] * 0.40
        + out["score_price"] * 0.25
        + out["score_concentration"] * 0.20
        + out["score_persistence"] * 0.15
    ).round(2)

    out["anomaly_level"] = pd.cut(
        out["anomaly_score"],
        bins=[-np.inf, 60, 80, np.inf],
        labels=["normal", "medium", "major"],
    ).astype(str)

    return out


def build_anomaly_outputs(
    raw_df: pd.DataFrame,
    cfg: BranchAnomalyConfig = BranchAnomalyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return 3 practical outputs:

    1) anomaly_ranked_events
    2) tradable_watchlist
    3) weekly_validation_report
    """
    features = prepare_branch_daily_features(raw_df)
    with_baseline = add_rolling_baselines(features, windows=(cfg.short_window, cfg.long_window))
    flagged = detect_rule_flags(with_baseline, cfg=cfg)
    scored = compute_anomaly_score(flagged)

    interval_summary = (
        scored.groupby(["stock_id", "branch_id"], as_index=False)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            observed_days=("date", "nunique"),
            event_days=("rule_triggered", "sum"),
            avg_score=("anomaly_score", "mean"),
            peak_score=("anomaly_score", "max"),
            z_net_20=("z_net_20", "mean"),
            z_gross_20=("z_gross_20", "mean"),
            vol_share=("vol_share", "mean"),
            flag_accumulation=("flag_accumulation", "max"),
            flag_distribution=("flag_distribution", "max"),
        )
        .reset_index(drop=True)
    )

    interval_summary["anomaly_score"] = (
        interval_summary["avg_score"] * 0.7 + interval_summary["peak_score"] * 0.3
    ).round(2)
    interval_summary["anomaly_level"] = pd.cut(
        interval_summary["anomaly_score"],
        bins=[-np.inf, 60, 80, np.inf],
        labels=["normal", "medium", "major"],
    ).astype(str)

    anomaly_ranked_events = interval_summary.sort_values(
        ["anomaly_score", "event_days", "vol_share"], ascending=[False, False, False]
    ).reset_index(drop=True)

    tradable_watchlist = anomaly_ranked_events.loc[
        (anomaly_ranked_events["event_days"] >= 1)
        & (anomaly_ranked_events["anomaly_score"] >= cfg.medium_score_threshold)
    ].copy()

    # weekly summary by stock and anomaly level
    weekly = scored.copy()
    weekly["week"] = pd.to_datetime(weekly["date"]).dt.to_period("W").dt.start_time
    weekly_validation_report = (
        weekly.groupby(["week", "stock_id", "anomaly_level"], as_index=False)
        .agg(
            events=("anomaly_score", "count"),
            avg_score=("anomaly_score", "mean"),
            avg_vol_share=("vol_share", "mean"),
        )
        .sort_values(["week", "stock_id", "anomaly_level"])
        .reset_index(drop=True)
    )

    return anomaly_ranked_events, tradable_watchlist, weekly_validation_report
