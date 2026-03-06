from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AccumulationScanConfig:
    min_observed_days: int = 20
    day_trade_filter_ratio: float = 0.10
    recent_anomaly_days: int = 5
    trend_shift_window: int = 10
    high_stability_threshold: float = 0.70
    medium_stability_threshold: float = 0.50
    anomaly_sigma_threshold: float = 2.0


def _prepare_scan_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in ("buy", "sell", "Trading_Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["net_buy"] = df["buy"] - df["sell"]
    df["abs_trade"] = (df["buy"] + df["sell"]).clip(lower=1.0)
    df["buy_sell_gap_ratio"] = (df["net_buy"].abs() / df["abs_trade"]).fillna(0.0)

    volume_base = df["Trading_Volume"].replace(0, np.nan)
    df["volume_share"] = (df["net_buy"] / volume_base).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _detect_recent_anomaly(signal_df: pd.DataFrame, recent_days: int, sigma_threshold: float) -> bool:
    if signal_df.empty:
        return False

    ranked_dates = signal_df["date"].drop_duplicates().sort_values()
    tail_dates = ranked_dates.tail(max(int(recent_days), 1))
    recent = signal_df[signal_df["date"].isin(tail_dates)]

    buy_z = (signal_df["buy"] - signal_df["buy"].mean()) / (signal_df["buy"].std(ddof=0) or 1.0)
    share_z = (signal_df["volume_share"] - signal_df["volume_share"].mean()) / (signal_df["volume_share"].std(ddof=0) or 1.0)

    signal_df = signal_df.assign(_buy_z=buy_z.fillna(0.0), _share_z=share_z.fillna(0.0))
    recent = signal_df[signal_df["date"].isin(tail_dates)]

    return bool(((recent["_buy_z"] >= sigma_threshold) | (recent["_share_z"] >= sigma_threshold)).any())


def _detect_structure_shift(stock_df: pd.DataFrame, window: int) -> bool:
    daily_net = stock_df.groupby("date", as_index=False)["net_buy"].sum().sort_values("date")
    if len(daily_net) < max(window * 2, 10):
        return False

    cumulative = daily_net["net_buy"].cumsum()
    slope = cumulative.diff().fillna(0.0)
    recent = slope.tail(window)
    previous = slope.iloc[-(window * 2) : -window]

    if previous.empty:
        return False

    recent_mean = float(recent.mean())
    prev_mean = float(previous.mean())
    prev_std = float(previous.std(ddof=0) or 1.0)

    # 結構轉向：近期均值顯著高於過去，或由偏空轉偏多
    significantly_stronger = (recent_mean - prev_mean) >= (1.5 * prev_std)
    sign_flip_to_positive = prev_mean <= 0 < recent_mean
    return bool(significantly_stronger or sign_flip_to_positive)


def run_accumulation_scan(raw_df: pd.DataFrame, cfg: AccumulationScanConfig) -> pd.DataFrame:
    df = _prepare_scan_frame(raw_df)
    if df.empty:
        return pd.DataFrame()

    candidates: list[dict] = []

    for stock_id, stock_df in df.groupby("stock_id", dropna=False):
        observed_days = int(stock_df["date"].nunique())
        if observed_days < int(cfg.min_observed_days):
            continue

        filtered = stock_df[stock_df["buy_sell_gap_ratio"] > float(cfg.day_trade_filter_ratio)].copy()
        if filtered.empty:
            continue

        total_days = int(filtered["date"].nunique())
        if total_days == 0:
            continue

        branch_positive = filtered.assign(positive=(filtered["net_buy"] > 0).astype(int))
        stability_stats = (
            branch_positive.groupby(["branch_id", "branch_name"], dropna=False)["positive"].sum() / total_days
        ).sort_values(ascending=False)

        if stability_stats.empty:
            continue

        (dominant_branch_id, dominant_branch_name), max_stability = stability_stats.index[0], float(stability_stats.iloc[0])

        has_recent_anomaly = _detect_recent_anomaly(
            filtered,
            recent_days=int(cfg.recent_anomaly_days),
            sigma_threshold=float(cfg.anomaly_sigma_threshold),
        )
        is_shifting = _detect_structure_shift(filtered, window=int(cfg.trend_shift_window))

        triggered = (max_stability >= cfg.high_stability_threshold and is_shifting) or (
            max_stability >= cfg.medium_stability_threshold and has_recent_anomaly
        )
        if not triggered:
            continue

        signal_reason = "穩定吸籌 + 結構轉強" if (max_stability >= cfg.high_stability_threshold and is_shifting) else "中高穩定 + 異常吸籌"

        candidates.append(
            {
                "stock_id": stock_id,
                "branch_id": dominant_branch_id,
                "branch_name": dominant_branch_name,
                "buying_stability": round(max_stability, 4),
                "structure_shift": bool(is_shifting),
                "recent_anomaly": bool(has_recent_anomaly),
                "observed_days": observed_days,
                "signal_reason": signal_reason,
                "latest_signal_end": filtered["date"].max(),
            }
        )

    if not candidates:
        return pd.DataFrame()

    out = pd.DataFrame(candidates).sort_values(
        ["buying_stability", "structure_shift", "recent_anomaly", "latest_signal_end"],
        ascending=[False, False, False, False],
    )
    return out.reset_index(drop=True)
