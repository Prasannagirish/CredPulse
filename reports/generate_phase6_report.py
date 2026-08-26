"""Phase 6 deliverable: hyperparameter tuning, a logistic regression + WOE baseline
comparison, and calibration analysis for the champion model. All numbers are a backtest on
public historical Lending Club data — not a live production evaluation.

Note: unlike the Phase 2/3/4/5 report scripts, this one never imports torch (no drift or
mcp_server modules are used here), so the OMP_NUM_THREADS=1 guard those scripts need to
avoid the PyTorch/XGBoost OpenMP conflict is not needed here — omitting it lets XGBoost use
all available cores, which matters a lot across 30 real tuning trials.
"""
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, REPORTS_DIR, TUNING_AUC_IMPROVEMENT_THRESHOLD
from model.baseline_woe import evaluate_logistic_auc, train_logistic_baseline
from model.calibration import expected_calibration_error, reliability_diagram_data
from model.train import evaluate_auc, temporal_holdout_split, train_baseline
from model.tune import tune_hyperparameters


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")

    print("=== BACKTEST ON HISTORICAL DATA — not a live production evaluation ===")

    default_pipeline, _, test_slice = train_baseline(reference_df)
    default_auc = evaluate_auc(default_pipeline, test_slice)
    print(f"Current default-config holdout AUC: {default_auc:.4f}")

    print("\nRunning Optuna hyperparameter search (30 trials, ~15-30 min on this dataset)...")
    best_params, tuned_auc = tune_hyperparameters(reference_df)
    print(f"Best tuned holdout AUC: {tuned_auc:.4f}")
    print(f"Best params: {best_params}")

    improvement = tuned_auc - default_auc
    print(f"Improvement over default: {improvement:+.4f} "
          f"(threshold for adopting: {TUNING_AUC_IMPROVEMENT_THRESHOLD})")
    if improvement > TUNING_AUC_IMPROVEMENT_THRESHOLD:
        print("DECISION: tuning found a meaningful improvement — "
              "update model/train.py's build_model_pipeline with these params.")
    else:
        print("DECISION: tuning did not clear the improvement threshold — "
              "keeping the current default configuration.")
    champion_pipeline = default_pipeline  # report always evaluates the CURRENT default;
    # if the decision above is to adopt tuned params, model/train.py is edited afterward
    # and this report is re-run to reflect the new champion.

    logistic_model, woe_bins = train_logistic_baseline(reference_df)
    _, _, logistic_test_slice = train_baseline(reference_df)  # same split, for a fair comparison
    logistic_auc = evaluate_logistic_auc(logistic_model, woe_bins, logistic_test_slice)
    xgboost_auc = evaluate_auc(champion_pipeline, logistic_test_slice)

    print("\n=== Baseline comparison (same held-out slice) ===")
    print(f"Logistic regression + WOE — AUC: {logistic_auc:.4f}")
    print(f"XGBoost champion          — AUC: {xgboost_auc:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Logistic + WOE", "XGBoost (champion)"], [logistic_auc, xgboost_auc], color=["tab:orange", "tab:blue"])
    ax.set_ylabel("AUC")
    ax.set_title("Baseline Comparison — Backtest on Historical Lending Club Data")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase6_tuning_comparison.png", dpi=150)

    _, _, calibration_slice = train_baseline(reference_df)
    ece = expected_calibration_error(champion_pipeline, calibration_slice)
    diagram = reliability_diagram_data(champion_pipeline, calibration_slice)
    print("\n=== Calibration (champion, held-out slice) ===")
    print(f"Expected Calibration Error: {ece:.4f}")
    print(diagram.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(diagram["mean_predicted"], diagram["observed_rate"], marker="o", label="champion")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability Diagram — Backtest on Historical Lending Club Data")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase6_calibration.png", dpi=150)
    print(f"\nCharts saved to {REPORTS_DIR}/phase6_tuning_comparison.png and phase6_calibration.png")


if __name__ == "__main__":
    main()
