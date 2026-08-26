# CreditPulse Phase 7 — Fairness and Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Check the champion for disparate impact across documented proxy groups (Lending
Club's public data has no protected-class labels), and replace Phase 5's single-point
dollar-impact estimate with a full approval-rate sensitivity curve.

**Architecture:** `backtest/impact.py` gets a small extraction — `compute_accepted_mask`,
pulled out of `dollar_impact` so `backtest/fairness.py` can reuse the exact same
"accept the lowest-risk N%" mechanic instead of reimplementing it. `backtest/fairness.py`
adds per-group approval/default rates and the standard 4/5ths-rule disparate-impact ratio.
`backtest/sensitivity.py` sweeps `dollar_impact` across a range of approval rates.
`reports/generate_phase7_report.py` runs both against real data and updates the README.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, matplotlib, pytest.

## Global Constraints

- **Stated limitation, in the module docstring and every report's printed output:**
  Lending Club's public dataset contains no protected-class labels (race, sex, age,
  national origin, marital status) — genuine fair-lending analysis is not possible on this
  data. The fairness check uses two literature-grounded proxies instead — `home_ownership`
  and income quartiles of `annual_inc` — and its results are a **screening heuristic, not a
  legal or regulatory determination**. This sentence (or one materially identical to it)
  must appear in `backtest/fairness.py`'s docstring and in
  `reports/generate_phase7_report.py`'s printed output — never only in this plan.
- Income quartiles are computed from real data at report-run time (`pd.qcut` on the actual
  batch), never hardcoded dollar boundaries.
- The fairness check and the approval-rate sensitivity sweep both reuse
  `backtest.impact.dollar_impact`'s exact "accept the lowest-risk N% via the model's own
  threshold" mechanic — extracted into a shared `compute_accepted_mask` helper, not
  reimplemented.
- The disparate-impact ratio threshold is `config.DISPARATE_IMPACT_THRESHOLD` (0.8, the
  standard US "4/5ths rule" screen) — flagged below it, not flagged at or above it.
- Every printed/plotted result is labeled a backtest on historical, resolved-outcome data.

---

### Task 1: Extract shared accepted-mask helper

**Files:**
- Modify: `backtest/impact.py` — extract `compute_accepted_mask`
- Modify: `config.py` — add `DISPARATE_IMPACT_THRESHOLD`, `APPROVAL_RATE_SWEEP`
- Test: `tests/test_backtest.py` (existing `dollar_impact` test must still pass unchanged)

**Interfaces:**
- Produces: `backtest.impact.compute_accepted_mask(pipeline: Pipeline, df: pd.DataFrame,
  approval_rate: float = config.APPROVAL_RATE) -> np.ndarray` — a boolean array, `True` for
  rows accepted at the given approval rate (the same lowest-risk-N% mechanic `dollar_impact`
  already used inline). Task 2 (`backtest/fairness.py`) and Task 3
  (`backtest/sensitivity.py`, indirectly via `dollar_impact`) both rely on this.

- [ ] **Step 1: Add new constants to `config.py`**

Append to `config.py`:

```python
# Fairness screening (Phase 7). 0.8 is the standard US "4/5ths rule" disparate-impact
# screen — a heuristic starting point, not a legal determination.
DISPARATE_IMPACT_THRESHOLD = 0.8

# Approval-rate sensitivity sweep (Phase 7) — replaces Phase 5's single 80%-only estimate
# with a curve across a realistic range of approval policies.
APPROVAL_RATE_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
```

- [ ] **Step 2: Extract `compute_accepted_mask` in `backtest/impact.py`**

Find this block inside `dollar_impact`:

```python
    feature_cols = [c for c in batch_df.columns if c not in config.NON_FEATURE_COLUMNS]
    risk_scores = pipeline.predict_proba(batch_df[feature_cols])[:, 1]
    threshold = np.quantile(risk_scores, approval_rate)
    accepted = batch_df[risk_scores <= threshold]
```

Replace the whole `dollar_impact` function with:

```python
def compute_accepted_mask(
    pipeline: Pipeline, df: pd.DataFrame, approval_rate: float = config.APPROVAL_RATE
) -> np.ndarray:
    """Boolean mask, True for the approval_rate fraction of df with the LOWEST predicted
    default risk — the model picks its own threshold to hit that rate. Shared by
    dollar_impact (below) and backtest.fairness.group_approval_rates, so both analyses use
    the exact same accept/reject decision."""
    feature_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLUMNS]
    risk_scores = pipeline.predict_proba(df[feature_cols])[:, 1]
    threshold = np.quantile(risk_scores, approval_rate)
    return risk_scores <= threshold


def dollar_impact(
    pipeline: Pipeline, batch_df: pd.DataFrame, approval_rate: float = config.APPROVAL_RATE
) -> dict:
    """Accept the approval_rate fraction of batch_df with the LOWEST predicted default risk
    — each model picks its own threshold to hit that rate. This is what keeps the
    comparison fair: a model can't "win" simply by rejecting more loans than another."""
    accepted_mask = compute_accepted_mask(pipeline, batch_df, approval_rate)
    accepted = batch_df[accepted_mask]

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

- [ ] **Step 3: Run the existing backtest test suite to confirm the refactor didn't break anything**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_backtest.py -v
```

Expected: all 3 previously-passing tests (including
`test_dollar_impact_accepts_fixed_fraction_and_computes_loss_per_10k`) still pass — pure
extraction, identical behavior.

- [ ] **Step 4: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add backtest/impact.py config.py
git commit -m "Extract compute_accepted_mask from dollar_impact for reuse by fairness checks"
```

---

### Task 2: Disparate-impact screening

**Files:**
- Create: `backtest/fairness.py`
- Create: `tests/test_fairness.py`

**Interfaces:**
- Consumes: `compute_accepted_mask` from `backtest.impact`; `config.APPROVAL_RATE`,
  `config.DISPARATE_IMPACT_THRESHOLD`, `config.TARGET_COLUMN`.
- Produces:
  - `group_approval_rates(pipeline: Pipeline, df: pd.DataFrame, group_col: str,
    approval_rate: float = config.APPROVAL_RATE) -> pd.DataFrame` — columns `group`,
    `n_total`, `n_accepted`, `approval_rate`, `n_defaults`, `default_rate`. Uses **one
    shared threshold** for the whole `df` (from `compute_accepted_mask`), then breaks the
    result out by `group_col` — the group column plays no role in the accept/reject
    decision itself, only in how results are reported.
  - `disparate_impact_ratio(group_rates: pd.DataFrame) -> dict` — keys `ratio` (min group
    approval rate / max group approval rate), `min_group`, `max_group`, `flagged` (`True` if
    `ratio < config.DISPARATE_IMPACT_THRESHOLD`).
  - `add_income_quartile_column(df: pd.DataFrame) -> pd.DataFrame` — returns a copy of `df`
    with a new `income_quartile` column (`pd.qcut` on `annual_inc`, computed from `df`
    itself, never hardcoded boundaries).

  Task 4 (report script) calls all three directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fairness.py
import numpy as np
import pandas as pd
import pytest

from backtest.fairness import add_income_quartile_column, disparate_impact_ratio, group_approval_rates
from lifecycle.retrain import retrain_challenger


def _synthetic_df(n: int = 500, start: str = "2019-01-01") -> pd.DataFrame:
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


def test_group_approval_rates_breaks_out_by_group():
    df = _synthetic_df(n=500)
    pipeline = retrain_challenger(df)

    rates = group_approval_rates(pipeline, df, group_col="home_ownership")

    assert set(rates["group"]) == {"RENT", "OWN", "MORTGAGE"}
    assert (rates["approval_rate"] >= 0).all() and (rates["approval_rate"] <= 1).all()
    assert rates["n_total"].sum() == len(df)


def test_disparate_impact_ratio_flags_large_gap():
    group_rates = pd.DataFrame({"group": ["A", "B"], "approval_rate": [0.9, 0.5]})

    result = disparate_impact_ratio(group_rates)

    assert result["ratio"] == pytest.approx(0.5 / 0.9)
    assert result["flagged"] is True
    assert result["min_group"] == "B"
    assert result["max_group"] == "A"


def test_disparate_impact_ratio_does_not_flag_similar_rates():
    group_rates = pd.DataFrame({"group": ["A", "B"], "approval_rate": [0.85, 0.82]})

    result = disparate_impact_ratio(group_rates)

    assert result["flagged"] is False


def test_add_income_quartile_column_creates_four_groups_from_real_data():
    df = _synthetic_df(n=400)

    result = add_income_quartile_column(df)

    assert result["income_quartile"].nunique() == 4
    assert len(result) == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_fairness.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.fairness'`.

- [ ] **Step 3: Implement `backtest/fairness.py`**

```python
# backtest/fairness.py
"""Disparate-impact screening across documented proxy groups.

LIMITATION: Lending Club's public dataset contains no protected-class labels (race, sex,
age, national origin, marital status) — genuine fair-lending analysis is not possible on
this data. This module uses two literature-grounded PROXIES instead — home_ownership and
income quartiles of annual_inc — and its results are a screening heuristic, not a legal or
regulatory determination.
"""
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from backtest.impact import compute_accepted_mask


def group_approval_rates(
    pipeline: Pipeline, df: pd.DataFrame, group_col: str, approval_rate: float = config.APPROVAL_RATE
) -> pd.DataFrame:
    """Approval and default rates broken out by group_col, using ONE shared risk threshold
    for the whole df — group_col plays no role in the accept/reject decision itself, only
    in how the single decision's outcomes are reported per group."""
    df = df.reset_index(drop=True)
    accepted_mask = compute_accepted_mask(pipeline, df, approval_rate)

    rows = []
    for group_value in sorted(df[group_col].dropna().unique(), key=str):
        group_mask = (df[group_col] == group_value).to_numpy()
        n_total = int(group_mask.sum())
        group_accepted = group_mask & accepted_mask
        n_accepted = int(group_accepted.sum())
        n_defaults = int(df.loc[group_accepted, config.TARGET_COLUMN].sum())
        rows.append({
            "group": group_value,
            "n_total": n_total,
            "n_accepted": n_accepted,
            "approval_rate": (n_accepted / n_total) if n_total else 0.0,
            "n_defaults": n_defaults,
            "default_rate": (n_defaults / n_accepted) if n_accepted else 0.0,
        })
    return pd.DataFrame(rows)


def disparate_impact_ratio(group_rates: pd.DataFrame) -> dict:
    """Standard US "4/5ths rule" screen: minimum group approval rate divided by maximum.
    Below config.DISPARATE_IMPACT_THRESHOLD is flagged — a screening heuristic, not a legal
    determination."""
    min_row = group_rates.loc[group_rates["approval_rate"].idxmin()]
    max_row = group_rates.loc[group_rates["approval_rate"].idxmax()]
    ratio = (min_row["approval_rate"] / max_row["approval_rate"]) if max_row["approval_rate"] > 0 else 1.0
    return {
        "ratio": float(ratio),
        "min_group": min_row["group"],
        "max_group": max_row["group"],
        "flagged": ratio < config.DISPARATE_IMPACT_THRESHOLD,
    }


def add_income_quartile_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an income_quartile column computed from this df's own annual_inc distribution —
    never a hardcoded dollar boundary."""
    out = df.copy()
    out["income_quartile"] = pd.qcut(
        out["annual_inc"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_fairness.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add backtest/fairness.py tests/test_fairness.py
git commit -m "Add disparate-impact screening (4/5ths rule) across proxy groups"
```

---

### Task 3: Approval-rate sensitivity

**Files:**
- Create: `backtest/sensitivity.py`
- Create: `tests/test_sensitivity.py`

**Interfaces:**
- Consumes: `dollar_impact` from `backtest.impact`; `config.APPROVAL_RATE_SWEEP`.
- Produces: `approval_rate_sensitivity(pipeline: Pipeline, batch_df: pd.DataFrame,
  approval_rates: list[float] = config.APPROVAL_RATE_SWEEP) -> pd.DataFrame` — one row per
  approval rate, columns `approval_rate` plus every key `dollar_impact` returns
  (`n_accepted`, `n_defaults`, `default_rate`, `dollar_loss`, `dollar_loss_per_10k_loans`).

  Task 4 (report script) calls this directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sensitivity.py
import numpy as np
import pandas as pd

from backtest.sensitivity import approval_rate_sensitivity
from lifecycle.retrain import retrain_challenger


def _synthetic_df(n: int = 500, start: str = "2019-01-01") -> pd.DataFrame:
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


def test_approval_rate_sensitivity_sweeps_all_rates_and_accepts_more_at_higher_rates():
    df = _synthetic_df(n=500)
    pipeline = retrain_challenger(df)

    result = approval_rate_sensitivity(pipeline, df, approval_rates=[0.5, 0.8])

    assert list(result["approval_rate"]) == [0.5, 0.8]
    assert (result["n_accepted"] > 0).all()
    assert result.iloc[1]["n_accepted"] >= result.iloc[0]["n_accepted"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_sensitivity.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.sensitivity'`.

- [ ] **Step 3: Implement `backtest/sensitivity.py`**

```python
# backtest/sensitivity.py
"""Approval-rate sensitivity: sweeps the dollar-impact calculation across a range of
approval rates instead of a single point estimate."""
import pandas as pd
from sklearn.pipeline import Pipeline

from backtest.impact import dollar_impact
from config import APPROVAL_RATE_SWEEP


def approval_rate_sensitivity(
    pipeline: Pipeline, batch_df: pd.DataFrame, approval_rates: list[float] = APPROVAL_RATE_SWEEP
) -> pd.DataFrame:
    rows = []
    for rate in approval_rates:
        result = dollar_impact(pipeline, batch_df, approval_rate=rate)
        rows.append({"approval_rate": rate, **result})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_sensitivity.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Run the full test suite**

```bash
cd ~/IdeaProjects/mlops
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add backtest/sensitivity.py tests/test_sensitivity.py
git commit -m "Add approval-rate sensitivity sweep"
```

---

### Task 4: Real report — fairness screen, sensitivity curve, README

**Files:**
- Create: `reports/generate_phase7_report.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `add_income_quartile_column`, `disparate_impact_ratio`, `group_approval_rates`
  from `backtest.fairness`; `approval_rate_sensitivity` from `backtest.sensitivity`;
  `train_baseline` from `model.train`; `config.APPROVAL_RATE`, `config.MIN_RELIABLE_DEFAULTS`,
  `config.PROCESSED_DIR`, `config.REPORTS_DIR`.
- Produces: a standalone script and three charts
  (`reports/phase7_fairness_home_ownership.png`, `reports/phase7_fairness_income_quartile.png`,
  `reports/phase7_approval_sensitivity.png`). No other module depends on this script's
  internals.

- [ ] **Step 1: Write the report script**

```python
# reports/generate_phase7_report.py
"""Phase 7 deliverable: a disparate-impact screening check and an approval-rate sensitivity
sweep. All numbers are a backtest on public historical Lending Club data — not a live
production evaluation.

LIMITATION: Lending Club's public dataset contains no protected-class labels (race, sex,
age, national origin, marital status) — genuine fair-lending analysis is not possible on
this data. The fairness section below uses two literature-grounded PROXIES instead —
home_ownership and income quartiles — and its results are a screening heuristic, not a
legal or regulatory determination.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
```

- [ ] **Step 2: Run the report against the real prepared data**

```bash
cd ~/IdeaProjects/mlops
uv run python -m reports.generate_phase7_report
```

Expected: prints the limitation statement, the evaluation month and loan count, the
per-group approval/default rate tables for `home_ownership` and `income_quartile` with
each group's disparate-impact ratio and flag status, and the approval-rate sensitivity
table. Saves three charts. **Inspect the real numbers** — if either group breakdown is
flagged (ratio < 0.8), that's a real finding to report honestly in the README, not to
adjust the threshold to make disappear.

- [ ] **Step 3: Add a Phase 7 section to `README.md`**

Using the real printed numbers from Step 2, add a "## Phase 7 — Fairness & Sensitivity"
section after the Phase 6 section, covering:

- The stated limitation (no protected-class labels, proxies only, screening heuristic).
- The per-group approval-rate tables (or a summary of them) for `home_ownership` and
  `income_quartile`, with the real disparate-impact ratios and whether either was flagged.
- The approval-rate sensitivity chart
  (`reports/phase7_approval_sensitivity.png`) replacing Phase 5's single 80%-only dollar
  figure with the full curve, plus one sentence on what the curve shows (e.g., where the
  savings-per-10k-loans gap between models widens or narrows as approval policy changes).
- The backtest framing line, consistent with every other section.

- [ ] **Step 4: Commit**

```bash
cd ~/IdeaProjects/mlops
git add reports/generate_phase7_report.py reports/phase7_fairness_home_ownership.png reports/phase7_fairness_income_quartile.png reports/phase7_approval_sensitivity.png README.md
git commit -m "Add Phase 7 report: disparate-impact screening and approval-rate sensitivity"
```

---

## Self-review notes

- **Spec coverage:** disparate-impact check across `home_ownership`/income-quartile proxies
  with the standard 4/5ths-rule ratio ✓; explicit no-protected-class-labels limitation
  stated in both code and report output ✓; approval-rate sensitivity sweep replacing the
  single-point Phase 5 estimate ✓; both extend `backtest.impact.dollar_impact`'s existing
  mechanic via the extracted `compute_accepted_mask`, not a reimplementation ✓; income
  quartile boundaries computed from real data via `pd.qcut`, never hardcoded ✓; README
  section ✓.
- **Placeholder scan:** none remaining.
- **Type consistency:** `compute_accepted_mask` (Task 1) is called identically by
  `dollar_impact` (Task 1, refactored) and `group_approval_rates` (Task 2).
  `group_approval_rates`'s output columns (Task 2) match exactly what
  `disparate_impact_ratio` (Task 2) and the report script's printed table (Task 4) both
  consume. `approval_rate_sensitivity`'s output columns (Task 3) match exactly what the
  report script (Task 4) selects and plots.
