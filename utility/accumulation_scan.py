from dataclasses import dataclass

import pandas as pd

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover
    rpt = None

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover
    IsolationForest = None


@dataclass
class AccumulationScanConfig:
    lookback_days: int = 60
    min_stability: float = 0.5
    coord_threshold: float = 0.5
    final_score_threshold: int = 50
    anomaly_contamination: float = 0.05
    changepoint_penalty: int = 8
    changepoint_recent_days: int = 10
    min_shift_days: int = 10
    anomaly_recent_days: int = 3
    min_anomaly_samples: int = 10
    coordination_corr_threshold: float = 0.7
    coordination_stability_gate: float = 0.3
    net_buy_noise_ratio: float = 0.05
    score_stability: int = 40
    score_shifting: int = 20
    score_coordination: int = 20
    score_anomaly: int = 10
    score_participant_diff: int = 10


def _prepare_scan_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in ("buy", "sell", "price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["net_buy"] = pd.to_numeric(df["buy"] - df["sell"], errors="coerce").fillna(0)
    return df


def _detect_changepoint(cumulative_net: pd.Series, penalty: int, recent_days: int, min_shift_days: int) -> bool:
    if len(cumulative_net) <= int(min_shift_days):
        return False

    if rpt is None:
        return False

    algo = rpt.Pelt(model="rbf").fit(cumulative_net.values)
    points = algo.predict(pen=int(penalty))
    return any(p > (len(cumulative_net) - int(recent_days)) for p in points[:-1])


def _detect_anomaly(g: pd.DataFrame, contamination: float, recent_days: int, min_anomaly_samples: int) -> bool:
    if len(g) < int(min_anomaly_samples):
        return False

    if IsolationForest is None:
        return False

    feature_cols = ["buy", "price"]
    if "price" not in g.columns:
        return False

    feat = g[feature_cols].fillna(0).values
    iso = IsolationForest(contamination=float(contamination), random_state=42)
    g = g.copy()
    g["anomaly"] = iso.fit_predict(feat)
    return bool((g.tail(int(recent_days))["anomaly"] == -1).any())


def _coordination_score(g: pd.DataFrame, min_stability: float, corr_threshold: float) -> float:
    if g.empty:
        return 0.0

    total_days = max(g["date"].nunique(), 1)
    branch_stability = g.groupby("branch_id").apply(lambda x: (x["net_buy"] > 0).sum() / total_days)
    if branch_stability.empty or branch_stability.max() <= float(min_stability):
        return 0.0

    pivoted = g.pivot_table(index="date", columns="branch_id", values="net_buy", fill_value=0)
    if pivoted.shape[1] < 2:
        return 0.0

    if nx is None:
        return 0.0

    corr_matrix = pivoted.corr()
    G = nx.from_pandas_adjacency((corr_matrix > float(corr_threshold)).astype(int))
    centrality = nx.degree_centrality(G)
    return float(max(centrality.values())) if centrality else 0.0


def run_accumulation_scan(raw_df: pd.DataFrame, cfg: AccumulationScanConfig) -> pd.DataFrame:
    df = _prepare_scan_frame(raw_df)
    if df.empty:
        return pd.DataFrame()

    sorted_dates = sorted(df["date"].dropna().unique())
    if sorted_dates:
        effective_lookback = min(int(cfg.lookback_days), len(sorted_dates))
        selected_dates = set(sorted_dates[-effective_lookback:])
        df = df[df["date"].isin(selected_dates)]

    # 過濾當沖雜訊：淨買賣需超過買量的 5%
    df = df[abs(df["net_buy"]) > (df["buy"] * float(cfg.net_buy_noise_ratio))]
    if df.empty:
        return pd.DataFrame()

    final_candidates = []
    for stock_id, g in df.groupby("stock_id"):
        total_days = max(g["date"].nunique(), 1)
        stability = g.groupby("branch_id").apply(lambda x: (x["net_buy"] > 0).sum() / total_days)
        stability_score = float(stability.max()) if not stability.empty else 0.0

        buyers = g[g["net_buy"] > 0]["branch_id"].nunique()
        sellers = g[g["net_buy"] < 0]["branch_id"].nunique()
        participant_diff = int(buyers - sellers)

        cumulative_net = g.groupby("date")["net_buy"].sum().sort_index().cumsum()
        is_shifting = _detect_changepoint(
            cumulative_net,
            cfg.changepoint_penalty,
            cfg.changepoint_recent_days,
            cfg.min_shift_days,
        )
        has_anomaly = _detect_anomaly(
            g.sort_values("date"),
            cfg.anomaly_contamination,
            cfg.anomaly_recent_days,
            cfg.min_anomaly_samples,
        )
        coordination_score = _coordination_score(
            g,
            min_stability=cfg.coordination_stability_gate,
            corr_threshold=cfg.coordination_corr_threshold,
        )

        final_score = 0
        if stability_score >= float(cfg.min_stability):
            final_score += int(cfg.score_stability)
        if is_shifting:
            final_score += int(cfg.score_shifting)
        if coordination_score >= float(cfg.coord_threshold):
            final_score += int(cfg.score_coordination)
        if has_anomaly:
            final_score += int(cfg.score_anomaly)
        if participant_diff < 0:
            final_score += int(cfg.score_participant_diff)

        if final_score >= int(cfg.final_score_threshold):
            final_candidates.append(
                {
                    "stock_id": stock_id,
                    "stock_name": g["stock_name"].dropna().iloc[0] if "stock_name" in g.columns and not g["stock_name"].dropna().empty else "",
                    "final_score": int(final_score),
                    "stability": round(stability_score, 2),
                    "is_shifting": bool(is_shifting),
                    "has_anomaly": bool(has_anomaly),
                    "coordination_score": round(coordination_score, 2),
                    "participant_diff": participant_diff,
                }
            )

    if not final_candidates:
        return pd.DataFrame()

    out = pd.DataFrame(final_candidates).sort_values(
        ["final_score", "stability", "coordination_score", "has_anomaly"],
        ascending=[False, False, False, False],
    )
    return out.reset_index(drop=True)
