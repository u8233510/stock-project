import pandas as pd


def compute_interval_metrics(df, top_n=15, trader_col_candidates=("securities_trader_id", "securities_trader")):
    """Compute avg_cost, total_net_volume, concentration(%) from interval branch rows."""
    if df is None or df.empty:
        return 0.0, 0, 0.0

    work = df.copy()
    for col in ["buy", "sell", "price"]:
        if col not in work.columns:
            return 0.0, 0, 0.0

    work["buy"] = pd.to_numeric(work["buy"], errors="coerce").fillna(0)
    work["sell"] = pd.to_numeric(work["sell"], errors="coerce").fillna(0)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work.dropna(subset=["price"])
    if work.empty:
        return 0.0, 0, 0.0

    total_buy_volume = float(work["buy"].sum())
    total_net_volume = int((work["buy"] - work["sell"]).sum())
    avg_cost = round(float((work["price"] * work["buy"]).sum() / total_buy_volume), 2) if total_buy_volume > 0 else 0.0

    trader_col = next((c for c in trader_col_candidates if c in work.columns), None)
    if trader_col is None or total_buy_volume <= 0:
        return avg_cost, total_net_volume, 0.0

    trader_net_df = work.groupby(trader_col, dropna=False)[["buy", "sell"]].sum()
    trader_net = trader_net_df["buy"] - trader_net_df["sell"]
    if trader_net.empty:
        return avg_cost, total_net_volume, 0.0

    n = max(1, int(top_n or 15))
    top_buy = float(trader_net.nlargest(n).clip(lower=0).sum())
    top_sell_abs = float(abs(trader_net.nsmallest(n).clip(upper=0).sum()))
    concentration = round(((top_buy - top_sell_abs) / total_buy_volume) * 100, 2)
    return avg_cost, total_net_volume, concentration
