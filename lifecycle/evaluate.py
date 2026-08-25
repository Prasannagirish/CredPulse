"""Score a champion and a challenger on the same held-out window, never one either was
trained on (spec Phase 3)."""
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from config import NON_FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class ChampionChallengerComparison:
    champion_auc: float
    challenger_auc: float
    champion_f1: float
    challenger_f1: float
    champion_brier: float
    challenger_brier: float
    challenger_wins: bool


def _score(pipeline: Pipeline, df: pd.DataFrame) -> tuple[float, float, float]:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    y_true = df[TARGET_COLUMN]
    y_prob = pipeline.predict_proba(df[feature_cols])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return (
        roc_auc_score(y_true, y_prob),
        f1_score(y_true, y_pred),
        brier_score_loss(y_true, y_prob),
    )


def compare_champion_challenger(
    champion: Pipeline, challenger: Pipeline, holdout_df: pd.DataFrame
) -> ChampionChallengerComparison:
    champion_auc, champion_f1, champion_brier = _score(champion, holdout_df)
    challenger_auc, challenger_f1, challenger_brier = _score(challenger, holdout_df)
    return ChampionChallengerComparison(
        champion_auc=champion_auc,
        challenger_auc=challenger_auc,
        champion_f1=champion_f1,
        challenger_f1=challenger_f1,
        champion_brier=champion_brier,
        challenger_brier=challenger_brier,
        challenger_wins=challenger_auc > champion_auc,
    )
