"""AI 贏家分點整合入口（Facade）。

目的：
- 不重做一套重複邏輯
- 直接整合既有 `winner_branch_ai_system.py` + `winner_branch_ml.py`
- 額外提供日期區間與中文欄位相容
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from utility.winner_branch_ai_system import WinnerBranchConfig, build_winner_branch_outputs
from utility.winner_branch_ml import (
    WinnerMLConfig,
    build_phase2_training_dataset,
    optimize_trade_params,
    train_xgboost_classifier,
)


_REQUIRED_SOURCE_COLUMNS = ["日期", "分點名稱", "買進張數", "賣出張數", "成交均價"]


@dataclass(frozen=True)
class ChipStrategyConfig:
    """整合設定。"""

    winner_cfg: WinnerBranchConfig = WinnerBranchConfig()
    ml_cfg: WinnerMLConfig = WinnerMLConfig()


class ChipStrategyAI:
    """整合入口：明確提供兩個需求的獨立流程。

    需求 1: 贏家分點自動化追蹤（winner tracking）
    需求 2: AI 從海量數據挖掘交易策略（strategy mining）
    """

    def __init__(
        self,
        filepath: str,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        config: ChipStrategyConfig = ChipStrategyConfig(),
    ) -> None:
        print(f"[{datetime.now()}] 載入分點資料中...")
        self.df = pd.read_csv(filepath)
        self.df["日期"] = pd.to_datetime(self.df["日期"])

        self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        self.end_date = pd.to_datetime(end_date) if end_date is not None else None
        self.config = config

        self._normalized_df = self._normalize_input(self.df)

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
        """把中文欄位轉成現有 winner 模組需要的標準欄位。"""
        miss = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in raw.columns]
        if miss:
            raise ValueError(f"缺少必要欄位: {miss}")

        out = raw.copy()
        out["日期"] = pd.to_datetime(out["日期"])

        # 若資料檔沒有股票代號，視為單一標的資料
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
                # 若有收盤價欄位優先使用；否則 fallback 成交均價
                "close": out.get("收盤價", out["成交均價"]).astype(float),
            }
        )
        return mapped


    def track_winner_branches(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        top_n: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """需求 1：針對「贏家分點」進行自動化追蹤。

        回傳：
        - winner_rating: 分點評級
        - daily_alerts: 每日警示
        - concentration: 集中度特徵
        - strategy_candidates: 規則型候選（供觀察）
        - top_winners: 依 winner_rating 取前 top_n
        """
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

    def mine_trading_strategies(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        ml_cfg: WinnerMLConfig | None = None,
    ) -> dict:
        """需求 2：從海量數據挖掘交易策略（模型導向）。

        流程：
        1) 產生訓練資料集
        2) 訓練分類模型
        3) 參數掃描（持有天數/停損）
        """
        dataset = self.build_ml_dataset(start_date=start_date, end_date=end_date, ml_cfg=ml_cfg)
        if dataset.empty:
            raise ValueError("策略挖掘資料不足，請擴大日期區間或檢查輸入欄位/品質。")

        model_result = train_xgboost_classifier(dataset)
        params = optimize_trade_params(dataset, signal_col="label_positive")

        return {
            "dataset": dataset,
            "model_result": model_result,
            "param_scan": params,
        }
    def run_winner_pipeline(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """執行既有 AI 贏家分點輸出（評級/警示/集中度/候選）。"""
        data = self._slice_by_date(start_date=start_date, end_date=end_date)
        if data.empty:
            empty = pd.DataFrame()
            return empty, empty, empty, empty
        return build_winner_branch_outputs(data, cfg=self.config.winner_cfg)

    def get_winner_list(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """相容舊 API：回傳 top N 贏家分點。

        注意：此結果來自現有 winner rating，不再重做另一套『金額總和』演算法。
        """
        winner_rating, _, _, _ = self.run_winner_pipeline(start_date=start_date, end_date=end_date)
        if winner_rating.empty:
            return winner_rating
        return winner_rating.sort_values("winner_rating", ascending=False).head(top_n).reset_index(drop=True)

    def build_ml_dataset(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        ml_cfg: WinnerMLConfig | None = None,
    ) -> pd.DataFrame:
        """建立第二階段訓練資料集（沿用既有 `winner_branch_ml`）。"""
        data = self._slice_by_date(start_date=start_date, end_date=end_date)
        if data.empty:
            return pd.DataFrame()
        return build_phase2_training_dataset(data, cfg=ml_cfg or self.config.ml_cfg)

    def train_strategy_model(self, dataset: pd.DataFrame | None = None) -> dict:
        """訓練模型（沿用既有 XGBoost 訓練流程）。"""
        ds = dataset if dataset is not None else self.build_ml_dataset()
        if ds.empty:
            raise ValueError("訓練資料不足，請擴大日期區間或確認資料品質。")
        return train_xgboost_classifier(ds)

    def optimize_params(self, dataset: pd.DataFrame | None = None) -> pd.DataFrame:
        """策略參數掃描（沿用既有 optimize_trade_params）。"""
        ds = dataset if dataset is not None else self.build_ml_dataset()
        if ds.empty:
            return pd.DataFrame()
        return optimize_trade_params(ds, signal_col="label_positive")


if __name__ == "__main__":
    print("ChipStrategyAI 整合入口可用。")
