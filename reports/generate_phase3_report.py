"""Phase 3 deliverable: retrain a challenger on a trailing window, compare it against the
Phase 1 champion on a genuinely held-out slice, register both in MLflow, and demonstrate the
confirm-gated promote/rollback lifecycle end-to-end. All numbers are a backtest on public
historical Lending Club data — not a live production decision."""
import mlflow
import mlflow.sklearn
import pandas as pd

from config import MLFLOW_TRACKING_URI, PROCESSED_DIR
from lifecycle.evaluate import compare_champion_challenger
from lifecycle.registry import (
    ConfirmationRequired,
    get_champion_version,
    promote_challenger,
    register_model,
    rollback,
    set_challenger,
)
from lifecycle.retrain import retrain_challenger, select_recent_window
from model.train import train_baseline

# Chosen deliberately, not arbitrarily: Phase 1/2 established that quarters within ~9
# months of the eval window's Sept-2020 data cutoff are right-censored (too few loans have
# had time to actually default yet), which makes threshold-based metrics like F1 read as
# near-zero regardless of model quality. 2019-11 keeps the 2-month holdout inside the
# reliably-labeled region (2019Q4, >100 observed defaults per Phase 1's own criterion) while
# still landing after Phase 2's drift breach (macro features breached 2019-06).
RETRAIN_AS_OF = pd.Timestamp("2019-11-01")
HOLDOUT_MONTHS = 2

# MLflow's sklearn flavor serializes with skops by default, which refuses to load a
# pipeline containing an XGBClassifier as "untrusted" unless explicitly whitelisted — a
# legitimate security default from skops, not an MLflow bug. We trust our own model here.
_SKOPS_TRUSTED_TYPES = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("creditpulse-phase3-lifecycle")

    print("=== BACKTEST ON HISTORICAL DATA — not a live production decision ===")

    champion, _, _ = train_baseline(reference_df)

    # A challenger retrained "as of" a given date should draw on all resolved loans
    # available by then, not just the eval-window partition — reference_df/eval_df is a
    # split for measuring the original champion's staleness, not a boundary retraining is
    # supposed to respect.
    all_loans_df = pd.concat([reference_df, eval_df], ignore_index=True)
    training_window = select_recent_window(all_loans_df, RETRAIN_AS_OF, months=12)
    holdout_start = RETRAIN_AS_OF
    holdout_end = RETRAIN_AS_OF + pd.DateOffset(months=HOLDOUT_MONTHS)
    holdout_df = all_loans_df[
        (all_loans_df["issue_d"] >= holdout_start) & (all_loans_df["issue_d"] < holdout_end)
    ]

    print(f"Retrain-as-of: {RETRAIN_AS_OF.date()}, trailing window: {len(training_window)} loans "
          f"({training_window['issue_d'].min().date()} to {training_window['issue_d'].max().date()})")
    print(f"Held-out comparison slice: {len(holdout_df)} loans "
          f"({holdout_start.date()} to {holdout_end.date()}, not used in either model's training)")

    challenger = retrain_challenger(training_window)
    comparison = compare_champion_challenger(champion, challenger, holdout_df)

    print(f"\nChampion   — AUC: {comparison.champion_auc:.4f}  F1: {comparison.champion_f1:.4f}  "
          f"Brier: {comparison.champion_brier:.4f}")
    print(f"Challenger — AUC: {comparison.challenger_auc:.4f}  F1: {comparison.challenger_f1:.4f}  "
          f"Brier: {comparison.challenger_brier:.4f}")
    print(f"Challenger wins on AUC: {comparison.challenger_wins}")

    with mlflow.start_run(run_name="champion-reference") as champion_run:
        mlflow.sklearn.log_model(champion, "model", skops_trusted_types=_SKOPS_TRUSTED_TYPES)
        mlflow.log_metric("auc", comparison.champion_auc)
        champion_uri = f"runs:/{champion_run.info.run_id}/model"
    with mlflow.start_run(run_name=f"challenger-as-of-{RETRAIN_AS_OF.date()}") as challenger_run:
        mlflow.sklearn.log_model(challenger, "model", skops_trusted_types=_SKOPS_TRUSTED_TYPES)
        mlflow.log_metric("auc", comparison.challenger_auc)
        challenger_uri = f"runs:/{challenger_run.info.run_id}/model"

    champion_version = register_model(champion_uri)
    if get_champion_version() is None:
        # Bootstrap: promote_challenger requires an existing challenger alias, so run the
        # very first champion through the same gated path rather than special-casing it —
        # there's simply no "previous champion" to protect on the first-ever registration.
        set_challenger(champion_version)
        promote_challenger(confirm=True)
    print(f"\nRegistered champion as version {champion_version} (bootstrapped as registry champion)")

    challenger_version = register_model(challenger_uri)
    set_challenger(challenger_version)
    print(f"Registered challenger as version {challenger_version}")

    print("\n--- Demonstrating confirm-gating (spec §2: no silent auto-promotion) ---")
    try:
        promote_challenger(confirm=False)
        raise AssertionError("promote_challenger should have refused without confirm=True")
    except ConfirmationRequired as e:
        print(f"promote_challenger(confirm=False) correctly refused: {e}")

    if comparison.challenger_wins:
        new_champion_version = promote_challenger(confirm=True)
        print(f"promote_challenger(confirm=True) succeeded — new champion is version {new_champion_version}")

        try:
            rollback(confirm=False)
            raise AssertionError("rollback should have refused without confirm=True")
        except ConfirmationRequired as e:
            print(f"rollback(confirm=False) correctly refused: {e}")

        reverted_version = rollback(confirm=True)
        print(f"rollback(confirm=True) succeeded — reverted to version {reverted_version}")
    else:
        print(
            "Challenger did not outperform champion on this window — in a real system, "
            "trigger_retrain (Phase 4) would report this and a human/agent would decide "
            "not to promote. No promotion attempted here for the same reason."
        )


if __name__ == "__main__":
    main()
