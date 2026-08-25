# CreditPulse Phase 5 — Quantified Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Answer the project's headline question with real numbers: how much did monitoring
+ retraining save versus a static, never-touched model? Walk three strategies (static,
scheduled, drift-monitored) forward monthly through the reliably-labeled 2019 window, and
compute a transparent dollar-impact estimate.

**Architecture:** `backtest/arms.py` holds the three strategies' retrain-decision logic as
small, independently-testable functions operating on a shared `ArmState`. `backtest/
impact.py` computes the fixed-approval-rate dollar-impact calculation. `backtest/
run_backtest.py` — the exact path spec §6 names — orchestrates the monthly walk, reusing
Phase 2's frozen drift reference, Phase 3's retraining, and Phase 1's AUC evaluation, and is
both the importable module and the report-generating entrypoint.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, xgboost, torch (via existing
`drift`/`lifecycle`/`model` modules), matplotlib, pytest.

## Global Constraints

- The walk-forward and dollar-impact calculation are restricted to **reliably-labeled
  months only** — the same `MIN_RELIABLE_DEFAULTS` bar Phase 1 established (>= 100 observed
  defaults). Empirically this is 2019-01 through 2019-12; months near the eval window's
  Sept-2020 data cutoff are right-censored (loans haven't had time to resolve), which would
  make both the AUC trajectory and the dollar figure meaningless for that period regardless
  of which strategy is actually better. This restriction is computed from real data and
  printed explicitly in the report — never silently applied.
- The drift reference (`StatisticalDriftReference`, `DriftAutoencoder`, reference errors) is
  fit **once**, on the pre-2019 reference window, using the **original initial champion's**
  fitted preprocessor — and reused unchanged for every month's drift check throughout the
  walk. Never refit per month or per arm. Critically: every call to `transform_for_drift`
  during the walk must pass the **original** initial pipeline, never an arm's current
  (possibly already-retrained) pipeline — a retrained arm's `ColumnTransformer` is freshly
  fit on a different window and produces a different feature space, which would silently
  make the frozen reference internally inconsistent with what it's being compared against.
- Retraining reuses `lifecycle.retrain.retrain_challenger` + `select_recent_window` exactly
  as Phase 3 built them — no new training logic.
- Dollar impact: a fixed approval rate (`config.APPROVAL_RATE`), each arm's own model picks
  its own threshold to accept exactly that fraction of applicants — this is what keeps the
  comparison fair; a model can't "win" by simply rejecting more loans than its competitor.
  Loss given default is `config.LOSS_GIVEN_DEFAULT * loan_amnt` — a stated, conservative
  simplification (no partial-recovery modeling, since recovery amounts are excluded from
  features as leaky post-origination fields, spec §3 — they're genuinely not available here
  either). The calculation is printed step by step, never collapsed into an unexplained
  headline number (spec §2).
- `backtest/run_backtest.py` is the literal path spec §6 names for "the full
  monthly-walk-forward simulation + $-impact calc" — not a `reports/generate_phaseN_report.py`
  file like Phases 1-3 used, since spec explicitly names this one.
- Every printed/plotted result is labeled a backtest on historical, resolved-outcome data.
- Reuse Phase 2's already-computed drift-detection lead time (17.4 weeks, macro-feature
  signal) in the README — do not recompute it in this phase.

---

### Task 1: Backtest arm strategies

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/arms.py`
- Create: `tests/test_backtest.py`
- Modify: `config.py` — add `SCHEDULED_RETRAIN_MONTHS`, `MONITORED_RETRAIN_COOLDOWN_MONTHS`

**Interfaces:**
- Produces:
  - `ArmState` (dataclass) — fields `name: str`, `pipeline: Pipeline`, `last_retrain_date:
    pd.Timestamp`, `retrain_count: int = 0`.
  - `maybe_retrain_scheduled(arm: ArmState, month: pd.Timestamp, all_loans_df: pd.DataFrame)
    -> ArmState` — blindly retrains every `config.SCHEDULED_RETRAIN_MONTHS` months,
    regardless of drift.
  - `maybe_retrain_monitored(arm: ArmState, month: pd.Timestamp, all_loans_df: pd.DataFrame,
    report: DriftReport) -> ArmState` — retrains only when `report.status != "ok"` **and**
    at least `config.MONITORED_RETRAIN_COOLDOWN_MONTHS` months have passed since the arm's
    last retrain (a persistent breach would otherwise trigger a retrain every single month,
    re-fitting on nearly-identical overlapping windows for no benefit). The static arm needs
    no function — it's just an `ArmState` whose pipeline is never touched after
    initialization.

  Task 3 (`backtest/run_backtest.py`) calls both functions once per simulated month.

- [ ] **Step 1: Add cadence constants to `config.py`**

Append to `config.py`:

```python
# Backtest arm cadence (Phase 5). Scheduled retrains on a blind calendar cadence; monitored
# retrains only on drift breach, with a cooldown so a persistent breach signal doesn't
# trigger a retrain every single month.
SCHEDULED_RETRAIN_MONTHS = 3
MONITORED_RETRAIN_COOLDOWN_MONTHS = 2
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_backtest.py
import numpy as np
import pandas as pd

from backtest.arms import ArmState, maybe_retrain_monitored, maybe_retrain_scheduled
from drift.engine import DriftReport
from lifecycle.retrain import retrain_challenger


def _synthetic_df(n: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="3D")
    risk = rng.normal(size=n)
    default_prob = 1 / (1 + np.exp(-risk))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)
    return pd.DataFrame({
        "issue_d": dates,
        "loan_amnt": rng.uniform(1000, 30000, size=n),
        "term": rng.choice([36, 60], size=n),
        "int_rate": rng.uniform(5, 25, size=n),
        "emp_length": rng.integers(0, 11, size=n),
        "annual_inc": rng.uniform(20000, 150000, size=n),
        "dti": rng.uniform(0, 40, size=n),
        "revol_util": rng.uniform(0, 100, size=n),
        "fico_range_low": rng.integers(660, 800, size=n),
        "fico_range_high": rng.integers(664, 804, size=n),
        "grade": rng.choice(list("ABCDEFG"), size=n),
        "sub_grade": rng.choice([f"{g}{i}" for g in "ABC" for i in range(1, 6)], size=n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], size=n),
        "verification_status": rng.choice(["Verified", "Not Verified"], size=n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], size=n),
        "default_flag": default_flag,
    })


def _dummy_report(status: str) -> DriftReport:
    return DriftReport(
        window="test", status=status, feature_drift=pd.DataFrame(),
        autoencoder_fraction_above_threshold=0.0, autoencoder_status=status, statistical_status=status,
    )


def test_maybe_retrain_scheduled_retrains_after_interval_elapses():
    df = _synthetic_df(600, "2017-01-01")
    initial = retrain_challenger(df)
    arm = ArmState(name="scheduled", pipeline=initial, last_retrain_date=pd.Timestamp("2019-01-01"))

    still_waiting = maybe_retrain_scheduled(arm, pd.Timestamp("2019-02-01"), df)
    assert still_waiting.retrain_count == 0

    retrained = maybe_retrain_scheduled(arm, pd.Timestamp("2019-04-01"), df)
    assert retrained.retrain_count == 1
    assert retrained.last_retrain_date == pd.Timestamp("2019-04-01")


def test_maybe_retrain_monitored_only_retrains_on_breach_after_cooldown():
    df = _synthetic_df(600, "2017-01-01")
    initial = retrain_challenger(df)
    arm = ArmState(name="monitored", pipeline=initial, last_retrain_date=pd.Timestamp("2019-01-01"))

    ok_result = maybe_retrain_monitored(arm, pd.Timestamp("2019-02-01"), df, _dummy_report("ok"))
    assert ok_result.retrain_count == 0

    too_soon = maybe_retrain_monitored(arm, pd.Timestamp("2019-02-01"), df, _dummy_report("breach"))
    assert too_soon.retrain_count == 0  # cooldown not elapsed yet

    retrained = maybe_retrain_monitored(arm, pd.Timestamp("2019-04-01"), df, _dummy_report("breach"))
    assert retrained.retrain_count == 1
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_backtest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backtest'`.

- [ ] **Step 4: Create the `backtest` package and implement `backtest/arms.py`**

```bash
cd ~/IdeaProjects/mlops
mkdir -p backtest
touch backtest/__init__.py
```

```python
# backtest/arms.py
"""Three backtest strategies for the monthly walk-forward: static (never retrains,
represented as an ArmState nothing ever touches), scheduled (blind periodic retrain), and
monitored (drift-triggered retrain)."""
from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from config import MONITORED_RETRAIN_COOLDOWN_MONTHS, RETRAIN_WINDOW_MONTHS, SCHEDULED_RETRAIN_MONTHS
from drift.engine import DriftReport
from lifecycle.retrain import retrain_challenger, select_recent_window


@dataclass
class ArmState:
    name: str
    pipeline: Pipeline
    last_retrain_date: pd.Timestamp
    retrain_count: int = 0


def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def maybe_retrain_scheduled(arm: ArmState, month: pd.Timestamp, all_loans_df: pd.DataFrame) -> ArmState:
    """Blindly retrain every SCHEDULED_RETRAIN_MONTHS months, regardless of drift."""
    if _months_between(month, arm.last_retrain_date) < SCHEDULED_RETRAIN_MONTHS:
        return arm
    training_window = select_recent_window(all_loans_df, month, months=RETRAIN_WINDOW_MONTHS)
    new_pipeline = retrain_challenger(training_window)
    return ArmState(arm.name, new_pipeline, month, arm.retrain_count + 1)


def maybe_retrain_monitored(
    arm: ArmState, month: pd.Timestamp, all_loans_df: pd.DataFrame, report: DriftReport
) -> ArmState:
    """Retrain only when this month's drift status is not "ok", and only if the cooldown
    since the last retrain has elapsed — otherwise a persistent breach would trigger a
    retrain every single month, re-fitting on nearly-identical overlapping windows."""
    if report.status == "ok":
        return arm
    if _months_between(month, arm.last_retrain_date) < MONITORED_RETRAIN_COOLDOWN_MONTHS:
        return arm
    training_window = select_recent_window(all_loans_df, month, months=RETRAIN_WINDOW_MONTHS)
    new_pipeline = retrain_challenger(training_window)
    return ArmState(arm.name, new_pipeline, month, arm.retrain_count + 1)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_backtest.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add backtest/__init__.py backtest/arms.py tests/test_backtest.py config.py
git commit -m "Add backtest arm strategies: static, scheduled, monitored retraining"
```

---

### Task 2: Dollar-impact calculation

**Files:**
- Create: `backtest/impact.py`
- Test: `tests/test_backtest.py` (append)
- Modify: `config.py` — add `APPROVAL_RATE`, `LOSS_GIVEN_DEFAULT`

**Interfaces:**
- Consumes: `config.NON_FEATURE_COLUMNS`, `config.TARGET_COLUMN`, `config.APPROVAL_RATE`,
  `config.LOSS_GIVEN_DEFAULT`.
- Produces: `dollar_impact(pipeline: Pipeline, batch_df: pd.DataFrame, approval_rate: float =
  config.APPROVAL_RATE) -> dict` — keys `n_accepted: int`, `n_defaults: int`, `default_rate:
  float`, `dollar_loss: float`, `dollar_loss_per_10k_loans: float`. Task 3
  (`backtest/run_backtest.py`) calls this once per arm on the final reliable month.

- [ ] **Step 1: Add dollar-impact assumptions to `config.py`**

Append to `config.py`:

```python
# Dollar-impact assumptions (spec §8 Phase 5) — shown explicitly in the report, never a
# hidden headline number (spec §2).
APPROVAL_RATE = 0.80  # accept the lowest-predicted-risk 80% of applicants
LOSS_GIVEN_DEFAULT = 1.0  # simplification: a defaulted loan's full principal is the loss —
# no partial-recovery modeling, since recovery amounts are excluded from features as
# leaky post-origination fields (spec §3) and genuinely aren't available here either.
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_backtest.py`:

```python
from backtest.impact import dollar_impact


def test_dollar_impact_accepts_fixed_fraction_and_computes_loss_per_10k():
    df = _synthetic_df(500, "2019-01-01")
    pipeline = retrain_challenger(df)

    result = dollar_impact(pipeline, df, approval_rate=0.5)

    assert abs(result["n_accepted"] - 250) <= 5
    assert result["n_defaults"] >= 0
    assert result["dollar_loss"] >= 0.0
    assert result["dollar_loss_per_10k_loans"] >= 0.0
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_backtest.py -v -k dollar_impact
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.impact'`.

- [ ] **Step 4: Implement `backtest/impact.py`**

```python
# backtest/impact.py
"""Dollar-impact estimate: at a fixed approval rate, how much expected loss does each
arm's own model accept among its approved loans? Backtest metric only."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config


def dollar_impact(
    pipeline: Pipeline, batch_df: pd.DataFrame, approval_rate: float = config.APPROVAL_RATE
) -> dict:
    """Accept the approval_rate fraction of batch_df with the LOWEST predicted default risk
    — each model picks its own threshold to hit that rate. This is what keeps the
    comparison fair: a model can't "win" simply by rejecting more loans than another."""
    feature_cols = [c for c in batch_df.columns if c not in config.NON_FEATURE_COLUMNS]
    risk_scores = pipeline.predict_proba(batch_df[feature_cols])[:, 1]
    threshold = np.quantile(risk_scores, approval_rate)
    accepted = batch_df[risk_scores <= threshold]

    n_accepted = len(accepted)
    n_defaults = int(accepted[config.TARGET_COLUMN].sum())
    dollar_loss = float(
        accepted.loc[accepted[config.TARGET_COLUMN] == 1, "loan_amnt"].sum() * config.LOSS_GIVEN_DEFAULT
    )
    per_10k = (dollar_loss / n_accepted * 10_000) if n_accepted else 0.0

    return {
        "n_accepted": n_accepted,
        "n_defaults": n_defaults,
        "default_rate": (n_defaults / n_accepted) if n_accepted else 0.0,
        "dollar_loss": dollar_loss,
        "dollar_loss_per_10k_loans": per_10k,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_backtest.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add backtest/impact.py tests/test_backtest.py config.py
git commit -m "Add fixed-approval-rate dollar-impact calculation"
```

---

### Task 3: Real monthly walk-forward backtest

**Files:**
- Create: `backtest/run_backtest.py`

**Interfaces:**
- Consumes: `ArmState`, `maybe_retrain_scheduled`, `maybe_retrain_monitored` from
  `backtest.arms`; `dollar_impact` from `backtest.impact`; `fit_statistical_reference` from
  `drift.statistical`; `train_autoencoder`, `reconstruction_error` from `drift.autoencoder`;
  `evaluate_drift` from `drift.engine`; `transform_for_drift` from `mcp_server.state`;
  `train_baseline`, `evaluate_auc` from `model.train`; `config.MIN_RELIABLE_DEFAULTS`,
  `config.NUMERIC_FEATURES`, `config.CATEGORICAL_FEATURES`, `config.APPROVAL_RATE`,
  `config.LOSS_GIVEN_DEFAULT`, `config.PROCESSED_DIR`, `config.REPORTS_DIR`.
- Produces: a standalone script producing `reports/phase5_backtest_auc_trajectory.png` and a
  printed summary covering all four Phase 5 deliverables. No other module depends on this
  script's internals.

- [ ] **Step 1: Write the report script**

```python
# backtest/run_backtest.py
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
```

- [ ] **Step 2: Run the backtest against the real prepared data**

```bash
cd ~/IdeaProjects/mlops
uv run python -m backtest.run_backtest
```

Expected: prints the reliably-labeled month list (expect roughly 2019-01 through 2019-12,
matching Phase 1's finding), the monthly AUC-by-arm table, mean AUC per arm with retrain
counts, AUC points recovered (monitored vs static, scheduled vs static), the dollar-impact
comparison for the final reliable month, and estimated savings per 10,000 loans. Saves
`reports/phase5_backtest_auc_trajectory.png`.

**Stop here and inspect the result before writing it up.** Two things specifically worth
checking against real output rather than assuming: (1) does `monitored` actually retrain
fewer or more times than `scheduled`, and does that correspond to a sensible AUC story — a
higher retrain count alone isn't automatically "better," what matters is whether monitored's
AUC trajectory tracks closer to what a fully-retrained-every-month model would achieve than
scheduled's does; (2) is the dollar-savings figure a sensible order of magnitude given the
observed default-rate difference between static's and monitored's accepted loans at the
fixed 80% approval rate — a savings figure of $0 or a wildly large number both warrant
checking the accepted-loan counts and default rates printed above before trusting the
headline number.

- [ ] **Step 3: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all tests pass (this step doesn't add new tests, but confirms nothing broke).

- [ ] **Step 4: Commit**

```bash
cd ~/IdeaProjects/mlops
git add backtest/run_backtest.py reports/phase5_backtest_auc_trajectory.png
git commit -m "Add monthly walk-forward backtest: static vs scheduled vs monitored, dollar impact"
```

---

### Task 4: Finalize README with Phase 5 results

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the real printed output and chart from Task 3's run. No new code interfaces —
  this task is documentation only.

- [ ] **Step 1: Add a "Phase 5 — Quantified Backtest" section to `README.md`**

Using Task 3's actual printed output and the actual reliable-months list, actual mean-AUC
figures, actual retrain counts, and actual dollar-impact numbers (not placeholders — copy
the real values from the terminal output), add a section after the existing "What's built so
far" section containing:

- The reliably-labeled month range actually found, and one sentence on why the walk stops
  there (right-censoring near the Sept-2020 cutoff, per Phase 1's finding).
- The AUC trajectory chart, referenced as `![Static vs Scheduled vs Monitored AUC](reports/phase5_backtest_auc_trajectory.png)`.
- A short table: mean AUC and retrain count for each of the three arms, and AUC points
  recovered (monitored vs static, scheduled vs static).
- Phase 2's drift-detection lead time (17.4 weeks, macro-feature signal) — reused, not
  recomputed, with a one-line note that it's from Phase 2's report.
- The dollar-impact figure with the calculation shown: approval rate, loss-given-default
  assumption, accepted-loan counts and default rates for static vs monitored, and the
  resulting savings per 10,000 loans.
- A closing line restating that every number above is a backtest on public historical
  Lending Club data, not a live production outcome (spec §2, §9).

- [ ] **Step 2: Update the spec's Definition of Done checklist**

Add a "## Definition of Done" section near the end of `README.md` reproducing spec §9's
checklist, checking off every item that's now genuinely true based on what's been built and
verified in Phases 1-4 and this phase — do not check an item unless you can point to the
specific report, chart, or test that verifies it.

- [ ] **Step 3: Commit**

```bash
cd ~/IdeaProjects/mlops
git add README.md
git commit -m "Add Phase 5 results and Definition of Done checklist to README"
```

---

## Self-review notes

- **Spec coverage:** `backtest/run_backtest.py` at the exact spec §6 path ✓; static/scheduled
  never-retrain vs blind-cadence vs drift-triggered retraining strategies ✓; AUC tracked over
  time for all three (spec asks for at least static vs monitored — scheduled is included
  too, since it was already an approved design-doc addition, not scope creep) ✓; dollar
  impact using the dataset's own `loan_amnt` and realized outcomes, calculation shown, not a
  single unexplained number ✓; all four Phase 5 acceptance-criteria deliverables (AUC
  trajectory chart, drift lead time, AUC points recovered, $-per-10k-loans) present in the
  README ✓; every result labeled backtest ✓.
- **Placeholder scan:** none remaining — Task 4's instruction to "copy the real values from
  the terminal output" is a concrete action with a named source, not a vague hedge; it's
  documentation work that depends on Task 3's real output, which doesn't exist until
  execution time, so it can't be pre-filled at plan-writing time without fabricating numbers.
- **Type consistency:** `ArmState` (Task 1) fields (`name`, `pipeline`, `last_retrain_date`,
  `retrain_count`) are constructed and read identically in Task 3. `maybe_retrain_scheduled`/
  `maybe_retrain_monitored`'s signatures from Task 1 match their call sites in Task 3 exactly
  (positional `arm, month, all_loans_df[, report]`). `dollar_impact`'s return dict keys
  (Task 2) match every key accessed in Task 3's print statements
  (`n_accepted`, `n_defaults`, `default_rate`, `dollar_loss_per_10k_loans`).
