"""Winner-branch tracking + strategy-mining utilities.

This module turns branch-level buy/sell records into a practical two-layer output:
1) point layer: winner branch rating/ranking and daily alerts
2) surface layer: feature-mined strategy candidates (HHI/Entropy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from utility.branch_anomaly_detection import prepare_branch_daily_features


@dataclass(frozen=True)
class WinnerBranchConfig:
    signal_windows: tuple[int, ...] = (5, 10, 20)
    hhi_rise_window: int = 10
    compression_window: int = 10
    compression_threshold: float = 0.02
    top_quantile: float = 0.85


REQUIRED_COLUMNS = ("stock_id", "date", "branch_id", "price", "buy", "sell", "close")


def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_div(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    out = numerator / denominator
    if not isinstance(out, pd.Series):
        out = pd.Series(out)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_stock_close_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    stock_close = (
        raw_df[["stock_id", "date", "close"]]
        .copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values(["stock_id", "date"])
        .drop_duplicates(["stock_id", "date"], keep="last")
    )
    return stock_close.reset_index(drop=True)


def _attach_forward_returns(stock_close: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = stock_close.copy()
    grouped = out.groupby("stock_id", group_keys=False)
    for w in windows:
        out[f"future_close_{w}"] = grouped["close"].shift(-w)
        out[f"fwd_ret_{w}"] = _safe_div(out[f"future_close_{w}"] - out["close"], out["close"]) * 100
    return out


def _longest_positive_streak(values: pd.Series) -> int:
    best, cur = 0, 0
    for v in values.fillna(0):
        if v > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _compute_winner_rating(branch_daily: pd.DataFrame, cfg: WinnerBranchConfig) -> pd.DataFrame:
    # only evaluate signal days where branch is net buyer
    sig = branch_daily[branch_daily["net_vol"] > 0].copy()
    if sig.empty:
        return pd.DataFrame(columns=["stock_id", "branch_id", "winner_rating", "winner_type"])

    grouped = sig.groupby(["stock_id", "branch_id"], as_index=False)
    rows = []
    for (stock_id, branch_id), g in grouped:
        g = g.sort_values("date").copy()

        # hit rate / alpha: averaged across configured windows
        hit_rates = []
        alphas = []
        for w in cfg.signal_windows:
            col = f"fwd_ret_{w}"
            valid = g[col].dropna()
            if valid.empty:
                continue
            hit_rates.append(float((valid > 0).mean()))
            stock_baseline = float(branch_daily.loc[branch_daily["stock_id"] == stock_id, col].mean())
            alphas.append(float(valid.mean() - (stock_baseline if np.isfinite(stock_baseline) else 0.0)))

        hit_rate = float(np.mean(hit_rates)) if hit_rates else 0.0
        avg_alpha = float(np.mean(alphas)) if alphas else 0.0

        # pnl ratio and sharpe use 5D return as short horizon proxy
        r = g.get("fwd_ret_5", pd.Series(dtype=float)).dropna()
        pos = r[r > 0]
        neg = r[r < 0]
        avg_win = float(pos.mean()) if not pos.empty else 0.0
        avg_loss = float(abs(neg.mean())) if not neg.empty else 0.0
        pnl_ratio = float(avg_win / avg_loss) if avg_loss > 0 else (2.0 if avg_win > 0 else 0.0)
        sharpe = float(r.mean() / r.std(ddof=0)) if (not r.empty and r.std(ddof=0) > 0) else 0.0

        # timing score: closer to rolling low (20d band) gets higher score
        t = g[["date", "close"]].drop_duplicates("date").set_index("date").sort_index()
        low20 = t["close"].rolling(20, min_periods=5).min()
        high20 = t["close"].rolling(20, min_periods=5).max()
        band_pos = _safe_div(t["close"] - low20, (high20 - low20).replace(0, np.nan)).clip(0, 1)
        timing_score = float((1 - band_pos).mean()) if not band_pos.empty else 0.0

        # holding power and risk penalty
        all_days = branch_daily[(branch_daily["stock_id"] == stock_id) & (branch_daily["branch_id"] == branch_id)]
        longest_streak = _longest_positive_streak(all_days.sort_values("date")["net_vol"])
        holding_power = float(min(longest_streak / 10.0, 1.0))

        r_std = float(r.std(ddof=0)) if len(r) > 1 else 0.0
        left_tail = float(abs(np.percentile(r, 10))) if len(r) > 4 else 0.0
        risk_penalty = float(min((r_std + left_tail) / 20.0, 1.0))

        # normalize major inputs
        hit_n = np.clip(hit_rate, 0, 1)
        pnl_n = np.clip(pnl_ratio / 2.5, 0, 1)
        sharpe_n = np.clip((sharpe + 1.0) / 2.0, 0, 1)
        alpha_n = np.clip((avg_alpha + 5.0) / 10.0, 0, 1)

        winner_rating = float(
            100
            * (
                0.25 * hit_n
                + 0.20 * pnl_n
                + 0.20 * sharpe_n
                + 0.20 * timing_score
                + 0.15 * alpha_n
                - 0.10 * risk_penalty
                + 0.10 * holding_power
            )
        )

        turnover_proxy = float(all_days["gross_vol"].sum() / (all_days["net_vol"].abs().sum() + 1e-9))
        winner_type = "長線價值型" if (holding_power >= 0.5 and turnover_proxy <= 3.0) else "短線交易型"

        rows.append(
            {
                "stock_id": stock_id,
                "branch_id": branch_id,
                "hit_rate": round(hit_rate, 4),
                "avg_alpha": round(avg_alpha, 4),
                "pnl_ratio": round(pnl_ratio, 4),
                "sharpe": round(sharpe, 4),
                "timing_score": round(timing_score, 4),
                "holding_power": round(holding_power, 4),
                "risk_penalty": round(risk_penalty, 4),
                "winner_rating": round(max(winner_rating, 0.0), 2),
                "winner_type": winner_type,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["winner_rating", "stock_id"], ascending=[False, True]).reset_index(drop=True)


def _compute_concentration_features(branch_daily: pd.DataFrame, cfg: WinnerBranchConfig) -> pd.DataFrame:
    x = branch_daily.copy()

    # Per stock-day concentration based on branch volume share
    day = (
        x.groupby(["stock_id", "date"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "hhi": float((g["vol_share"] ** 2).sum()),
                    "entropy": float(-(g["vol_share"].clip(lower=1e-12) * np.log(g["vol_share"].clip(lower=1e-12))).sum()),
                    "close": float(g["close"].iloc[-1]),
                }
            )
        )
        .reset_index(drop=True)
        .sort_values(["stock_id", "date"])
    )

    day["date"] = pd.to_datetime(day["date"])
    grouped = day.groupby("stock_id", group_keys=False)
    day["hhi_delta_10"] = grouped["hhi"].diff(cfg.hhi_rise_window).fillna(0.0)

    min_periods = max(2, min(5, cfg.compression_window))
    roll_max = grouped["close"].rolling(cfg.compression_window, min_periods=min_periods).max().reset_index(level=0, drop=True)
    roll_min = grouped["close"].rolling(cfg.compression_window, min_periods=min_periods).min().reset_index(level=0, drop=True)
    day["price_compression"] = _safe_div(roll_max - roll_min, day["close"]).clip(lower=0)

    return day.reset_index(drop=True)


def _build_strategy_candidates(conc_df: pd.DataFrame, cfg: WinnerBranchConfig) -> pd.DataFrame:
    if conc_df.empty:
        return conc_df

    out = conc_df.copy()
    out["rule_hhi_rise"] = out["hhi_delta_10"] > 0.03
    out["rule_compression"] = out["price_compression"] <= cfg.compression_threshold
    out["strategy_candidate"] = out["rule_hhi_rise"] & out["rule_compression"]

    out["candidate_score"] = (
        np.clip(out["hhi_delta_10"] * 1000, 0, 60)
        + np.clip((cfg.compression_threshold - out["price_compression"]) * 1000, 0, 40)
    ).round(2)

    return out.sort_values(["strategy_candidate", "candidate_score"], ascending=[False, False]).reset_index(drop=True)


def _build_daily_alerts(branch_daily: pd.DataFrame, rating_df: pd.DataFrame, cfg: WinnerBranchConfig) -> pd.DataFrame:
    if branch_daily.empty or rating_df.empty:
        return pd.DataFrame(columns=["stock_id", "date", "branch_id", "alert_level", "reason"]) 

    x = branch_daily.merge(rating_df[["stock_id", "branch_id", "winner_rating", "winner_type"]], on=["stock_id", "branch_id"], how="inner")
    threshold = x["winner_rating"].quantile(cfg.top_quantile)

    x["is_top_winner"] = x["winner_rating"] >= threshold
    x["is_accum"] = (x["net_vol"] > 0) & (x["vol_share"] >= 0.08)
    x["is_strong"] = x["is_top_winner"] & x["is_accum"]

    x["alert_level"] = np.select(
        [x["is_strong"] & (x["vol_share"] >= 0.12), x["is_strong"], x["is_top_winner"]],
        ["A", "B", "C"],
        default="-",
    )

    x = x[x["alert_level"] != "-"]
    x["reason"] = (
        "winner_rating="
        + x["winner_rating"].round(2).astype(str)
        + "; net_vol="
        + x["net_vol"].round(0).astype(int).astype(str)
        + "; vol_share="
        + (x["vol_share"] * 100).round(2).astype(str)
        + "%"
    )

    cols = ["stock_id", "date", "branch_id", "winner_type", "winner_rating", "net_vol", "vol_share", "alert_level", "reason"]
    return x[cols].sort_values(["date", "alert_level", "winner_rating"], ascending=[False, True, False]).reset_index(drop=True)


def build_winner_branch_outputs(
    raw_df: pd.DataFrame,
    cfg: WinnerBranchConfig = WinnerBranchConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build practical outputs for winner-branch tracking and strategy mining.

    Returns:
    1) winner_rating_table
    2) daily_alerts
    3) concentration_features
    4) strategy_candidates
    """
    _validate_columns(raw_df, REQUIRED_COLUMNS)

    branch_daily = prepare_branch_daily_features(raw_df)
    stock_close = _build_stock_close_table(raw_df)
    stock_close = _attach_forward_returns(stock_close, cfg.signal_windows)

    for w in cfg.signal_windows:
        branch_daily = branch_daily.merge(
            stock_close[["stock_id", "date", f"fwd_ret_{w}"]],
            on=["stock_id", "date"],
            how="left",
        )

    winner_rating = _compute_winner_rating(branch_daily, cfg)
    daily_alerts = _build_daily_alerts(branch_daily, winner_rating, cfg)
    concentration = _compute_concentration_features(branch_daily, cfg)
    strategy_candidates = _build_strategy_candidates(concentration, cfg)

    return winner_rating, daily_alerts, concentration, strategy_candidates
