"""Phase 1 deliverable: baseline AUC on a held-out pre-2019 slice, and the quarterly AUC
decline through 2019-2020 without retraining. All numbers below are a backtest on public
historical Lending Club data — not a live production evaluation."""
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from config import MIN_RELIABLE_DEFAULTS, MLFLOW_TRACKING_URI, PROCESSED_DIR, REPORTS_DIR
from model.train import evaluate_auc, quarterly_auc, train_baseline


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("creditpulse-phase1-baseline")

    with mlflow.start_run(run_name="baseline-champion"):
        pipeline, _, test_slice = train_baseline(reference_df)
        holdout_auc = evaluate_auc(pipeline, test_slice)
        quarterly = quarterly_auc(pipeline, eval_df)

        # Sample sizes per quarter, for transparency: quarters near the data cutoff
        # (2020-09) have had little time for loans to resolve, so their "default" label
        # is heavily right-censored — very few defaults have had time to materialize yet,
        # independent of the model's actual accuracy. AUC on those quarters is unreliable,
        # not evidence of extreme drift, and must be labeled as such rather than presented
        # as a clean headline number (spec §2).
        eval_df_q = eval_df.copy()
        eval_df_q["quarter"] = pd.PeriodIndex(eval_df_q["issue_d"], freq="Q").astype(str)
        counts = eval_df_q.groupby("quarter")[["default_flag"]].agg(
            n_loans=("default_flag", "count"), n_defaults=("default_flag", "sum")
        )
        quarterly = quarterly.merge(counts, on="quarter")
        # A quarter's default label is trustworthy once it has a reasonable number of
        # observed defaults; below that, the labels are still mostly "hasn't happened yet".
        quarterly["reliable"] = quarterly["n_defaults"] >= MIN_RELIABLE_DEFAULTS

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("reference_holdout_auc", holdout_auc)
        for _, row in quarterly.iterrows():
            mlflow.log_metric(f"auc_{row['quarter']}", row["auc"])

    print("=== BACKTEST ON HISTORICAL DATA — not a live evaluation ===")
    print(f"Held-out pre-2019 test slice AUC: {holdout_auc:.4f}")
    print("\nQuarterly AUC on 2019-2020 data, static model (no retraining):")
    print(quarterly.to_string(index=False))
    print(
        f"\nNote: quarters with fewer than {MIN_RELIABLE_DEFAULTS} observed defaults are "
        "right-censored (loans issued close to the data cutoff haven't had time to "
        "default yet) and are marked unreliable, not evidence of drift beyond that point."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    reliable = quarterly[quarterly["reliable"]]
    unreliable = quarterly[~quarterly["reliable"]]
    ax.plot(reliable["quarter"], reliable["auc"], marker="o", color="tab:blue", label="reliable (n_defaults >= 100)")
    if len(unreliable):
        ax.plot(
            quarterly["quarter"], quarterly["auc"], linestyle=":", color="tab:blue", alpha=0.5,
        )
        ax.plot(
            unreliable["quarter"], unreliable["auc"], marker="x", color="tab:red",
            label="right-censored, unreliable (n_defaults < 100)", linestyle="none",
        )
    ax.axhline(holdout_auc, color="gray", linestyle="--", label=f"pre-2019 holdout AUC ({holdout_auc:.3f})")
    ax.set_ylabel("AUC")
    ax.set_xlabel("Quarter")
    ax.set_title("Static Model AUC Decline — Backtest on Historical Lending Club Data")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase1_auc_decline.png", dpi=150)
    print(f"\nChart saved to {REPORTS_DIR / 'phase1_auc_decline.png'}")


if __name__ == "__main__":
    main()
