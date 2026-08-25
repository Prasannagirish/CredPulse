"""Phase 5 deliverable: monthly walk-forward backtest of static vs scheduled vs monitored
retraining strategies through the reliably-labeled 2019 window, plus the dollar-impact
estimate. All numbers are a backtest on public historical Lending Club data — not a live
production outcome.

Restricted to the reliably-labeled window: Phase 1/2/3 established that eval-window months
within roughly 9 months of the Sept-2020 data cutoff are right-censored (too few loans have
had time to actually default yet), which would make both the AUC trajectory and the
dollar-impact calculation meaningless for that period regardless of which strategy is
actually better. Computed from real data below, not hardcoded.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import matplotlib.pyplot as plt

from backtest.arms import ArmState, maybe_retrain_monitored, maybe_retrain_scheduled
from backtest.impact import dollar_impact
from config import (
    APPROVAL_RATE,
    CATEGORICAL_FEATURES,
    LOSS_GIVEN_DEFAULT,
    MIN_RELIABLE_DEFAULTS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from drift.autoencoder import reconstruction_error, train_autoencoder
from drift.engine import evaluate_drift
from drift.statistical import fit_statistical_reference
from mcp_server.state import transform_for_drift
from model.train import evaluate_auc, train_baseline


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")
    all_loans_df = pd.concat([reference_df, eval_df], ignore_index=True)

    print("=== BACKTEST ON HISTORICAL DATA — not a live production outcome ===")

    eval_df_m = eval_df.copy()
    eval_df_m["month"] = eval_df_m["issue_d"].dt.to_period("M").astype(str)
    defaults_by_month = eval_df_m.groupby("month")["default_flag"].sum()
    reliable_months = sorted(defaults_by_month[defaults_by_month >= MIN_RELIABLE_DEFAULTS].index)
    print(f"Reliably-labeled months (>= {MIN_RELIABLE_DEFAULTS} observed defaults): {reliable_months}")

    initial_pipeline, _, _ = train_baseline(reference_df)
    start_date = pd.Period(reliable_months[0], freq="M").start_time

    static = ArmState(name="static", pipeline=initial_pipeline, last_retrain_date=start_date)
    scheduled = ArmState(name="scheduled", pipeline=initial_pipeline, last_retrain_date=start_date)
    monitored = ArmState(name="monitored", pipeline=initial_pipeline, last_retrain_date=start_date)

    # Drift reference: fit once on the pre-2019 reference window, frozen for the whole walk.
    # Always transform through initial_pipeline's preprocessor below, never an arm's current
    # (possibly retrained) one — the frozen reference must stay in the same feature space it
    # was fit in.
    statistical_reference = fit_statistical_reference(reference_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    reference_X = transform_for_drift(initial_pipeline, reference_df)
    autoencoder = train_autoencoder(reference_X, epochs=10, batch_size=1024)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    rows = []
    for month_label in reliable_months:
        month_start = pd.Period(month_label, freq="M").start_time
        batch_df = eval_df_m[eval_df_m["month"] == month_label]

        # Retrain decisions use only data strictly before this month (select_recent_window's
        # [as_of - months, as_of) semantics) — no leak into the batch being evaluated.
        scheduled = maybe_retrain_scheduled(scheduled, month_start, all_loans_df)

        comparison_X = transform_for_drift(initial_pipeline, batch_df)
        report = evaluate_drift(
            window=month_label,
            statistical_reference=statistical_reference,
            autoencoder=autoencoder,
            reference_errors=reference_errors,
            comparison_df=batch_df,
            comparison_X=comparison_X,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        monitored = maybe_retrain_monitored(monitored, month_start, all_loans_df, report)

        for arm in (static, scheduled, monitored):
            rows.append({
                "month": month_label,
                "arm": arm.name,
                "auc": evaluate_auc(arm.pipeline, batch_df),
                "retrain_count": arm.retrain_count,
            })

    trajectory = pd.DataFrame(rows)
    print("\nMonthly AUC by arm:")
    print(trajectory.pivot(index="month", columns="arm", values="auc").to_string())

    summary = trajectory.groupby("arm")["auc"].mean()
    retrain_counts = {arm.name: arm.retrain_count for arm in (static, scheduled, monitored)}
    print(f"\nMean AUC over the window — static: {summary['static']:.4f} "
          f"({retrain_counts['static']} retrains), "
          f"scheduled: {summary['scheduled']:.4f} ({retrain_counts['scheduled']} retrains), "
          f"monitored: {summary['monitored']:.4f} ({retrain_counts['monitored']} retrains)")
    print(f"AUC points recovered, monitored vs static: {summary['monitored'] - summary['static']:.4f}")
    print(f"AUC points recovered, scheduled vs static: {summary['scheduled'] - summary['static']:.4f}")

    final_month = reliable_months[-1]
    final_batch = eval_df_m[eval_df_m["month"] == final_month]
    static_impact = dollar_impact(static.pipeline, final_batch)
    monitored_impact = dollar_impact(monitored.pipeline, final_batch)

    print(f"\n=== Dollar impact, {final_month}, approval rate={APPROVAL_RATE:.0%}, "
          f"loss_given_default={LOSS_GIVEN_DEFAULT} (full principal, no recovery modeling) ===")
    print(f"Static    — accepted {static_impact['n_accepted']} loans, "
          f"{static_impact['n_defaults']} defaults ({static_impact['default_rate']:.2%}), "
          f"${static_impact['dollar_loss_per_10k_loans']:,.0f} loss per 10,000 loans")
    print(f"Monitored — accepted {monitored_impact['n_accepted']} loans, "
          f"{monitored_impact['n_defaults']} defaults ({monitored_impact['default_rate']:.2%}), "
          f"${monitored_impact['dollar_loss_per_10k_loans']:,.0f} loss per 10,000 loans")
    savings_per_10k = static_impact["dollar_loss_per_10k_loans"] - monitored_impact["dollar_loss_per_10k_loans"]
    print(f"Estimated savings from monitoring: ${savings_per_10k:,.0f} per 10,000 loans")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for arm_name in ("static", "scheduled", "monitored"):
        arm_data = trajectory[trajectory["arm"] == arm_name]
        ax.plot(arm_data["month"], arm_data["auc"], marker="o", label=arm_name)
    ax.set_xlabel("Month")
    ax.set_ylabel("AUC")
    ax.set_title("Static vs Scheduled vs Monitored — Backtest on Historical Lending Club Data")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase5_backtest_auc_trajectory.png", dpi=150)
    print(f"\nChart saved to {REPORTS_DIR / 'phase5_backtest_auc_trajectory.png'}")


if __name__ == "__main__":
    main()
