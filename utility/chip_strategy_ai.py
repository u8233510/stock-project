"""AI 贏家分點整合入口（Facade）。

重點：
- 與既有 AI 贏家分點流程整合（不重複造輪子）
- 支援資料來源：DB 查詢後 DataFrame（主要）或 CSV（相容）
- 明確拆分兩個需求：贏家追蹤 / 策略挖掘
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from utility.winner_branch_ai_system import WinnerBranchConfig, build_winner_branch_outputs
from utility.winner_branch_ml import (
    WinnerMLConfig,
    build_today_candidate_list,
    build_phase2_training_dataset,
    optimize_trade_params,
    train_xgboost_classifier,
)

_REQUIRED_SOURCE_COLUMNS_CN = ["日期", "分點名稱", "買進張數", "賣出張數", "成交均價"]
_REQUIRED_SOURCE_COLUMNS_STD = ["stock_id", "date", "branch_id", "price", "buy", "sell", "close"]


@dataclass(frozen=True)
class ChipStrategyConfig:
    winner_cfg: WinnerBranchConfig = WinnerBranchConfig()
    ml_cfg: WinnerMLConfig = WinnerMLConfig()


class ChipStrategyAI:
    """兩條獨立流程：
    1) track_winner_branches()：贏家分點自動化追蹤
    2) mine_trading_strategies()：AI 策略挖掘
    """

    def __init__(
        self,
        filepath: str | None = None,
        data: pd.DataFrame | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        config: ChipStrategyConfig = ChipStrategyConfig(),
    ) -> None:
        if filepath is None and data is None:
            raise ValueError("請提供 filepath 或 data（二選一）。")

        print(f"[{datetime.now()}] 載入分點資料中...")
        if data is not None:
            self.df = data.copy()
        else:
            self.df = pd.read_csv(filepath)

        self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        self.end_date = pd.to_datetime(end_date) if end_date is not None else None
        self.config = config
        self._normalized_df = self._normalize_input(self.df)

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        config: ChipStrategyConfig = ChipStrategyConfig(),
    ) -> "ChipStrategyAI":
        return cls(data=data, start_date=start_date, end_date=end_date, config=config)

    def _slice_by_date(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        start = pd.to_datetime(start_date) if start_date is not None else self.start_date
        end = pd.to_datetime(end_date) if end_date is not None else self.end_date

        data = self._normalized_df
        if start is not None:
            data = data[data["date"] >= start]
        if end is not None:
            data = data[data["date"] <= end]
        return data.sort_values(["stock_id", "date", "branch_id"]).reset_index(drop=True)

    @staticmethod
    def _normalize_input(raw: pd.DataFrame) -> pd.DataFrame:
        cols = set(raw.columns)

        # 情境 A: DB/既有系統標準欄位（優先）
        if all(c in cols for c in _REQUIRED_SOURCE_COLUMNS_STD):
            out = raw.copy()
            out["date"] = pd.to_datetime(out["date"])
            return out[_REQUIRED_SOURCE_COLUMNS_STD].copy()

        # 情境 B: 中文 CSV 欄位（相容）
        miss_cn = [c for c in _REQUIRED_SOURCE_COLUMNS_CN if c not in cols]
        if miss_cn:
            raise ValueError(f"缺少必要欄位: {miss_cn}（或提供標準欄位 {_REQUIRED_SOURCE_COLUMNS_STD}）")

        out = raw.copy()
        out["日期"] = pd.to_datetime(out["日期"])
        if "stock_id" in out.columns:
            stock_id = out["stock_id"].astype(str)
        elif "股票代號" in out.columns:
            stock_id = out["股票代號"].astype(str)
        else:
            stock_id = pd.Series(["UNKNOWN"] * len(out), index=out.index)

        mapped = pd.DataFrame(
            {
                "stock_id": stock_id,
                "date": out["日期"],
                "branch_id": out["分點名稱"].astype(str),
                "price": out["成交均價"].astype(float),
                "buy": out["買進張數"].astype(float),
                "sell": out["賣出張數"].astype(float),
                "close": out.get("收盤價", out["成交均價"]).astype(float),
            }
        )
        return mapped

    def run_winner_pipeline(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = self._slice_by_date(start_date=start_date, end_date=end_date)
        if data.empty:
            empty = pd.DataFrame()
            return empty, empty, empty, empty
        return build_winner_branch_outputs(data, cfg=self.config.winner_cfg)

    def track_winner_branches(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        top_n: int = 20,
    ) -> dict[str, pd.DataFrame]:
        winner_rating, daily_alerts, concentration, strategy_candidates = self.run_winner_pipeline(
            start_date=start_date,
            end_date=end_date,
        )

        if winner_rating.empty:
            top_winners = winner_rating
        else:
            top_winners = winner_rating.sort_values("winner_rating", ascending=False).head(top_n).reset_index(drop=True)

        return {
            "winner_rating": winner_rating,
            "daily_alerts": daily_alerts,
            "concentration": concentration,
            "strategy_candidates": strategy_candidates,
            "top_winners": top_winners,
        }

    def build_ml_dataset(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        ml_cfg: WinnerMLConfig | None = None,
    ) -> pd.DataFrame:
        data = self._slice_by_date(start_date=start_date, end_date=end_date)
        if data.empty:
            return pd.DataFrame()
        return build_phase2_training_dataset(data, cfg=ml_cfg or self.config.ml_cfg)

    def mine_trading_strategies(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        ml_cfg: WinnerMLConfig | None = None,
    ) -> dict:
        dataset = self.build_ml_dataset(start_date=start_date, end_date=end_date, ml_cfg=ml_cfg)
        if dataset.empty:
            raise ValueError("策略挖掘資料不足，請擴大日期區間或檢查資料品質。")

        model_result = train_xgboost_classifier(dataset)
        param_scan = optimize_trade_params(dataset, signal_col="label_positive")
        today_candidates = build_today_candidate_list(
            dataset,
            model_result,
            score_threshold=0.55,
            top_n=20,
        )
        return {
            "dataset": dataset,
            "model_result": model_result,
            "param_scan": param_scan,
            "today_candidates": today_candidates,
        }

    # backward-compatible wrappers
    def get_winner_list(self, start_date=None, end_date=None, top_n: int = 20) -> pd.DataFrame:
        return self.track_winner_branches(start_date=start_date, end_date=end_date, top_n=top_n)["top_winners"]

    def train_strategy_model(self, dataset: pd.DataFrame | None = None) -> dict:
        ds = dataset if dataset is not None else self.build_ml_dataset()
        if ds.empty:
            raise ValueError("訓練資料不足，請擴大日期區間或確認資料品質。")
        return train_xgboost_classifier(ds)

    def optimize_params(self, dataset: pd.DataFrame | None = None) -> pd.DataFrame:
        ds = dataset if dataset is not None else self.build_ml_dataset()
        if ds.empty:
            return pd.DataFrame()
        return optimize_trade_params(ds, signal_col="label_positive")


if __name__ == "__main__":
    print("ChipStrategyAI 整合入口可用。")
