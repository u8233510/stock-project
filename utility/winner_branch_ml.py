"""Model-oriented phase utilities for winner-branch strategy mining.

Phase-2 pipeline:
1) build labeled dataset (positive sample tagging)
2) train classifier (optional XGBoost)
3) simple parameter search for holding days / stop-loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from utility.branch_anomaly_detection import prepare_branch_daily_features


@dataclass(frozen=True)
class WinnerMLConfig:
    lookahead_days: int = 20
    rally_threshold: float = 0.08
    continuity_window: int = 5


REQUIRED_COLUMNS = ("stock_id", "date", "branch_id", "price", "buy", "sell", "close")


def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_div(a: pd.Series | float, b: pd.Series | float) -> pd.Series:
    out = a / b
    if not isinstance(out, pd.Series):
        out = pd.Series(out)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _stock_day_features(branch_daily: pd.DataFrame, continuity_window: int = 5) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        vol_share = g["vol_share"].clip(lower=1e-12)
        hhi = float((vol_share ** 2).sum())
        entropy = float(-(vol_share * np.log(vol_share)).sum())

        buy_side = g[g["net_vol"] > 0]
        if buy_side.empty:
            avg_buy_cost = float(g["close"].iloc[-1])
        else:
            avg_buy_cost = float(np.average(buy_side["avg_price"], weights=buy_side["net_vol"].abs()))

        net_buy_strength = float(g["net_vol"].sum())

        # retail_exit_ratio proxy: smaller-share branches net selling proportion
        small = g[g["vol_share"] <= 0.03]
        small_sell = float((-small["net_vol"].clip(upper=0)).sum())
        total_gross = float(g["gross_vol"].sum())
        retail_exit_ratio = float(small_sell / total_gross) if total_gross > 0 else 0.0

        return pd.Series(
            {
                "hhi": hhi,
                "entropy": entropy,
                "avg_buy_cost": avg_buy_cost,
                "net_buy_strength": net_buy_strength,
                "retail_exit_ratio": retail_exit_ratio,
                "close": float(g["close"].iloc[-1]),
            }
        )

    day = (
        branch_daily.groupby(["stock_id", "date"], as_index=False)
        .apply(_agg)
        .reset_index(drop=True)
        .sort_values(["stock_id", "date"])
    )

    day["date"] = pd.to_datetime(day["date"])
    day["cost_gap"] = _safe_div(day["close"] - day["avg_buy_cost"], day["avg_buy_cost"])

    grp = day.groupby("stock_id", group_keys=False)
    flow_sign = np.sign(day["net_buy_strength"])
    day["buy_continuity"] = (
        flow_sign.groupby(day["stock_id"]).rolling(continuity_window, min_periods=1).apply(lambda x: float((x > 0).mean()))
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # for candidate strategy optimization
    for h in (5, 10, 20):
        day[f"future_ret_{h}"] = grp["close"].shift(-h) / day["close"] - 1.0

    return day.reset_index(drop=True)


def _attach_positive_label(day_df: pd.DataFrame, lookahead_days: int, rally_threshold: float) -> pd.DataFrame:
    out = day_df.copy().sort_values(["stock_id", "date"])

    def _future_max_return(close_s: pd.Series) -> pd.Series:
        arr = close_s.to_numpy(dtype=float)
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(n):
            j = min(n, i + lookahead_days + 1)
            if i + 1 >= j:
                continue
            mx = np.max(arr[i + 1 : j])
            result[i] = (mx / arr[i]) - 1.0
        return pd.Series(result, index=close_s.index)

    out["future_max_ret"] = out.groupby("stock_id", group_keys=False)["close"].apply(_future_max_return)
    out["label_positive"] = (out["future_max_ret"] >= rally_threshold).astype(int)
    return out


def build_phase2_training_dataset(raw_df: pd.DataFrame, cfg: WinnerMLConfig = WinnerMLConfig()) -> pd.DataFrame:
    """Build labeled dataset for model-oriented strategy mining."""
    _validate_columns(raw_df, REQUIRED_COLUMNS)

    branch_daily = prepare_branch_daily_features(raw_df)
    day = _stock_day_features(branch_daily, continuity_window=cfg.continuity_window)
    day = _attach_positive_label(day, lookahead_days=cfg.lookahead_days, rally_threshold=cfg.rally_threshold)

    feature_cols = [
        "hhi",
        "entropy",
        "avg_buy_cost",
        "cost_gap",
        "net_buy_strength",
        "buy_continuity",
        "retail_exit_ratio",
    ]

    ds = day[["stock_id", "date", "close", *feature_cols, "future_max_ret", "label_positive", "future_ret_5", "future_ret_10", "future_ret_20"]].copy()
    return ds.dropna(subset=feature_cols + ["label_positive"]).reset_index(drop=True)


def train_xgboost_classifier(
    ds: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label_positive",
) -> dict:
    """Train optional XGBoost model.

    Returns a dict with status/metrics/model.
    """
    if feature_cols is None:
        feature_cols = [
            "hhi",
            "entropy",
            "avg_buy_cost",
            "cost_gap",
            "net_buy_strength",
            "buy_continuity",
            "retail_exit_ratio",
        ]

    try:
        from xgboost import XGBClassifier
    except Exception as e:  # dependency fallback
        return {"status": "missing_dependency", "message": f"xgboost unavailable: {e}"}

    x = ds[feature_cols].copy()
    y = ds[label_col].astype(int)

    # time-aware split by date quantile
    tmp = ds[["date"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    split_date = tmp["date"].quantile(0.8)
    train_idx = tmp["date"] <= split_date
    test_idx = tmp["date"] > split_date

    if train_idx.sum() < 50 or test_idx.sum() < 10:
        return {"status": "insufficient_data", "message": "Not enough samples for train/test split."}

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x.loc[train_idx], y.loc[train_idx])

    prob = model.predict_proba(x.loc[test_idx])[:, 1]
    pred = (prob >= 0.5).astype(int)
    yt = y.loc[test_idx].to_numpy()

    acc = float((pred == yt).mean())
    precision = float(((pred == 1) & (yt == 1)).sum() / max((pred == 1).sum(), 1))
    recall = float(((pred == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))

    return {
        "status": "ok",
        "split_date": str(split_date.date()),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "feature_importance": {c: float(v) for c, v in zip(feature_cols, model.feature_importances_)},
        "model": model,
    }


def optimize_trade_params(
    ds: pd.DataFrame,
    signal_col: str = "label_positive",
    hold_days_options: tuple[int, ...] = (5, 10, 20),
    stop_loss_options: tuple[float, ...] = (-0.03, -0.05, -0.08),
) -> pd.DataFrame:
    """Simple parameter sweep using future return columns as proxy backtest."""
    rows = []
    sig = ds[ds[signal_col] == 1].copy()
    if sig.empty:
        return pd.DataFrame(columns=["hold_days", "stop_loss", "sample_size", "win_rate", "avg_return", "expectancy"])

    for h in hold_days_options:
        col = f"future_ret_{h}"
        if col not in sig.columns:
            continue
        base = sig[col].dropna()
        if base.empty:
            continue
        for sl in stop_loss_options:
            r = base.clip(lower=sl)
            win_rate = float((r > 0).mean())
            avg_return = float(r.mean())
            avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
            avg_loss = float(abs(r[r <= 0].mean())) if (r <= 0).any() else 0.0
            loss_rate = 1.0 - win_rate
            expectancy = win_rate * avg_win - loss_rate * avg_loss

            rows.append(
                {
                    "hold_days": h,
                    "stop_loss": sl,
                    "sample_size": int(len(r)),
                    "win_rate": round(win_rate, 4),
                    "avg_return": round(avg_return, 4),
                    "expectancy": round(expectancy, 4),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["expectancy", "avg_return"], ascending=[False, False]).reset_index(drop=True)
