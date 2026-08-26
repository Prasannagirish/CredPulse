"""Phase 7 deliverable: a disparate-impact screening check and an approval-rate sensitivity
sweep. All numbers are a backtest on public historical Lending Club data — not a live
production evaluation.

LIMITATION: Lending Club's public dataset contains no protected-class labels (race, sex,
age, national origin, marital status) — genuine fair-lending analysis is not possible on
this data. The fairness section below uses two literature-grounded PROXIES instead —
home_ownership and income quartiles — and its results are a screening heuristic, not a
legal or regulatory determination.

Note: like Phase 6, this script never imports torch (no drift or mcp_server modules are
used here), so it does not need the OMP_NUM_THREADS=1 guard the Phase 2/3/4/5 report
scripts require to avoid the PyTorch/XGBoost OpenMP conflict.
"""
import pandas as pd
import matplotlib.pyplot as plt

from backtest.fairness import add_income_quartile_column, disparate_impact_ratio, group_approval_rates
from backtest.sensitivity import approval_rate_sensitivity
from config import APPROVAL_RATE, MIN_RELIABLE_DEFAULTS, PROCESSED_DIR, REPORTS_DIR
from model.train import train_baseline


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")

    print("=== BACKTEST ON HISTORICAL DATA — not a live production evaluation ===")
    print(
        "LIMITATION: Lending Club's public dataset has no protected-class labels (race, "
        "sex, age, national origin, marital status). The fairness check below uses "
        "documented proxies (home_ownership, income quartile) — it is a screening "
        "heuristic, not a genuine fair-lending or legal determination.\n"
    )

    champion_pipeline, _, _ = train_baseline(reference_df)

    eval_df_m = eval_df.copy()
    eval_df_m["month"] = eval_df_m["issue_d"].dt.to_period("M").astype(str)
    defaults_by_month = eval_df_m.groupby("month")["default_flag"].sum()
    reliable_months = sorted(defaults_by_month[defaults_by_month >= MIN_RELIABLE_DEFAULTS].index)
    final_month = reliable_months[-1]
    final_batch = eval_df_m[eval_df_m["month"] == final_month].copy()
    final_batch = add_income_quartile_column(final_batch)

    print(f"Evaluating on {final_month} ({len(final_batch)} loans), approval rate={APPROVAL_RATE:.0%}\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for group_col in ["home_ownership", "income_quartile"]:
        rates = group_approval_rates(champion_pipeline, final_batch, group_col)
        impact = disparate_impact_ratio(rates)
        print(f"--- Disparate impact by {group_col} ---")
        print(rates.to_string(index=False))
        flag_text = "FLAGGED (<0.8)" if impact["flagged"] else "not flagged (>=0.8)"
        print(f"Disparate impact ratio (min/max approval rate): {impact['ratio']:.3f} "
              f"({impact['min_group']} vs {impact['max_group']}) {flag_text}\n")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(rates["group"].astype(str), rates["approval_rate"], color="tab:blue")
        ax.axhline(APPROVAL_RATE, color="gray", linestyle="--",
                   label=f"overall approval rate ({APPROVAL_RATE:.0%})")
        ax.set_ylabel("Approval rate")
        ax.set_title(f"Approval Rate by {group_col} — Backtest, Screening Heuristic Only")
        ax.legend()
        plt.tight_layout()
        fig.savefig(REPORTS_DIR / f"phase7_fairness_{group_col}.png", dpi=150)

    sensitivity = approval_rate_sensitivity(champion_pipeline, final_batch)
    print("=== Approval-rate sensitivity ===")
    print(sensitivity[
        ["approval_rate", "n_accepted", "n_defaults", "default_rate", "dollar_loss_per_10k_loans"]
    ].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sensitivity["approval_rate"], sensitivity["dollar_loss_per_10k_loans"], marker="o")
    ax.set_xlabel("Approval rate")
    ax.set_ylabel("Loss per 10,000 loans ($)")
    ax.set_title("Approval-Rate Sensitivity — Backtest on Historical Lending Club Data")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase7_approval_sensitivity.png", dpi=150)
    print(f"\nCharts saved to {REPORTS_DIR}/phase7_fairness_*.png and phase7_approval_sensitivity.png")


if __name__ == "__main__":
    main()
