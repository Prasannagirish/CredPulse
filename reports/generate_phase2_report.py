"""Phase 2 deliverable: monthly drift status through 2019-2020, and the lead time between
the first drift warning/breach and the point where Phase 1's champion AUC actually craters.
All numbers below are a backtest on public historical Lending Club data."""
import os

# Must be set before numpy/xgboost/torch are imported: on macOS, XGBoost and PyTorch each
# bundle their own OpenMP runtime, and running both in one process without this segfaults
# (SIGSEGV, exit 139) as soon as XGBoost tries to fit — this is the first script in the
# project to use both libraries together.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CATEGORICAL_FEATURES,
    MIN_RELIABLE_DEFAULTS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    PSI_BREACH,
    REPORTS_DIR,
)
from drift.autoencoder import reconstruction_error, train_autoencoder
from drift.engine import evaluate_drift
from drift.statistical import fit_statistical_reference
from model.features import build_preprocessor
from model.train import quarterly_auc, train_baseline

_STATUS_RANK = {"ok": 0, "warning": 1, "breach": 2}

# grade/sub_grade reflect Lending Club's own risk-grading methodology, which the platform
# revises independently of the economy — drift here can mean "the platform changed its
# grading system", not "the economy shifted". Splitting them out from the other (economic)
# features lets the report distinguish the two rather than reporting one conflated number.
_GRADE_FEATURES = ["grade", "sub_grade"]
_MACRO_FEATURES = [f for f in NUMERIC_FEATURES if f not in _GRADE_FEATURES]


def main() -> None:
    reference_df = pd.read_parquet(PROCESSED_DIR / "reference.parquet")
    eval_df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")

    # --- Find the AUC trough from Phase 1, restricted to reliably-labeled quarters ---
    pipeline, _, _ = train_baseline(reference_df)
    quarterly = quarterly_auc(pipeline, eval_df)
    eval_df_q = eval_df.copy()
    eval_df_q["quarter"] = pd.PeriodIndex(eval_df_q["issue_d"], freq="Q").astype(str)
    counts = eval_df_q.groupby("quarter")["default_flag"].agg(n_defaults="sum").reset_index()
    quarterly = quarterly.merge(counts, on="quarter")
    reliable = quarterly[quarterly["n_defaults"] >= MIN_RELIABLE_DEFAULTS]
    trough_quarter = reliable.loc[reliable["auc"].idxmin(), "quarter"]
    trough_date = pd.Period(trough_quarter, freq="Q").start_time
    print(f"AUC trough (reliable window): {trough_quarter} (AUC={reliable['auc'].min():.4f}), "
          f"starts {trough_date.date()}")

    # --- Fit the drift reference on the full reference window ---
    preprocessor = build_preprocessor()
    reference_X = preprocessor.fit_transform(reference_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    if hasattr(reference_X, "toarray"):
        reference_X = reference_X.toarray()
    reference_X = reference_X.astype(np.float32)

    statistical_reference = fit_statistical_reference(reference_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    autoencoder = train_autoencoder(reference_X, epochs=10, batch_size=1024)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    # --- Walk forward monthly through the eval window ---
    eval_df = eval_df.copy()
    eval_df["month"] = eval_df["issue_d"].dt.to_period("M").astype(str)

    rows = []
    first_alert_month = None
    first_grade_breach_month = None
    first_macro_breach_month = None
    for month, group in sorted(eval_df.groupby("month")):
        comparison_X = preprocessor.transform(group[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        if hasattr(comparison_X, "toarray"):
            comparison_X = comparison_X.toarray()
        comparison_X = comparison_X.astype(np.float32)

        report = evaluate_drift(
            window=month,
            statistical_reference=statistical_reference,
            autoencoder=autoencoder,
            reference_errors=reference_errors,
            comparison_df=group,
            comparison_X=comparison_X,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        top = report.feature_drift.iloc[0]
        rows.append({
            "month": month, "status": report.status, "n_loans": len(group),
            "top_feature": top["feature"], "top_psi": top["psi"],
        })
        if first_alert_month is None and report.status != "ok":
            first_alert_month = month

        grade_psi = report.feature_drift[report.feature_drift["feature"].isin(_GRADE_FEATURES)]["psi"]
        if first_grade_breach_month is None and (grade_psi > PSI_BREACH).any():
            first_grade_breach_month = month

        macro_psi = report.feature_drift[report.feature_drift["feature"].isin(_MACRO_FEATURES)]["psi"]
        if first_macro_breach_month is None and (macro_psi > PSI_BREACH).any():
            first_macro_breach_month = month

    monthly = pd.DataFrame(rows)
    print("\n=== BACKTEST ON HISTORICAL DATA — not a live evaluation ===")
    print("Monthly drift status, 2019-2020 (reference = pre-2019 window, never retrained):")
    print(monthly.to_string(index=False))

    if first_alert_month is not None:
        alert_date = pd.Period(first_alert_month, freq="M").start_time
        lead_time_weeks = (trough_date - alert_date).days / 7
        print(f"\nFirst drift alert (any feature, any signal): {first_alert_month} "
              f"(starts {alert_date.date()})")
        print(f"Lead time vs AUC trough ({trough_quarter}): {lead_time_weeks:.1f} weeks")
    else:
        print("\nNo drift alert fired at any point in the eval window.")

    print(
        "\nAttribution — these two drivers are conflated in the single number above, and "
        "shouldn't be: grade/sub_grade reflect Lending Club's own risk-grading methodology, "
        "which the platform revises for business reasons unrelated to the economy, while the "
        "other numeric features (income, DTI, revolving utilization, FICO, etc.) are closer "
        "to a genuine macroeconomic/borrower-behavior signal."
    )
    if first_grade_breach_month is not None:
        print(f"  First grade/sub_grade PSI breach (platform methodology shift): {first_grade_breach_month}")
    else:
        print("  grade/sub_grade PSI never breached in the eval window.")
    if first_macro_breach_month is not None:
        macro_date = pd.Period(first_macro_breach_month, freq="M").start_time
        macro_lead_weeks = (trough_date - macro_date).days / 7
        print(f"  First macro-feature PSI breach (economic signal): {first_macro_breach_month} "
              f"— {macro_lead_weeks:.1f} weeks lead time vs AUC trough")
    else:
        print("  No macro (non-grade) feature PSI breached in the eval window.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    status_numeric = monthly["status"].map(_STATUS_RANK)
    ax.step(monthly["month"], status_numeric, where="mid", marker="o")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["ok", "warning", "breach"])
    ax.set_xlabel("Month")
    ax.set_title("Monthly Drift Status — Backtest on Historical Lending Club Data")
    if first_grade_breach_month is not None:
        ax.axvline(first_grade_breach_month, color="orange", linestyle="--",
                   label=f"grade/sub_grade breach ({first_grade_breach_month}, platform methodology)")
    if first_macro_breach_month is not None:
        ax.axvline(first_macro_breach_month, color="red", linestyle="--",
                   label=f"macro-feature breach ({first_macro_breach_month}, economic signal)")
    trough_month = trough_date.strftime("%Y-%m")
    if trough_month in monthly["month"].values:
        ax.axvline(trough_month, color="gray", linestyle=":", label=f"AUC trough ({trough_quarter})")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "phase2_drift_lead_time.png", dpi=150)
    print(f"\nChart saved to {REPORTS_DIR / 'phase2_drift_lead_time.png'}")


if __name__ == "__main__":
    main()
