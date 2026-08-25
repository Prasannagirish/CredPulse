# CreditPulse — Design

Date: 2026-08-25
Status: Draft for review

## 1. Purpose

Implement the CreditPulse spec (`~/Downloads/creditpulse-mlops-spec.md`): an agentic MLOps
control plane that trains a credit-default model on real historical loan data, detects when
it has drifted, retrains/evaluates a challenger under human-gated promotion, exposes the
lifecycle as MCP tools, and produces a backtested, quantified answer to "did monitoring +
retraining help, and by how much?"

This document covers implementation decisions not fully pinned down by the spec itself —
where the spec is explicit (repo layout, MCP tool signatures, phase order), this doc doesn't
repeat it.

## 2. Environment

- Python 3.12 via a `uv`-managed venv (system default is 3.14, where `xgboost`/`torch`/`mlflow`
  wheel support is unreliable). `uv venv --python 3.12`.
- Dataset: Kaggle `wordsforthewise/lending-club`, `accepted_2007_to_2018Q4.csv` — already
  downloaded to `~/Downloads/creditpulse-data/`. 2,260,702 rows, 151 columns. Confirmed schema
  matches spec §3: `issue_d` (origination date), `loan_status` (target), full applicant/loan
  attribute set, and the post-origination fields (`total_pymnt`, `recoveries`, `last_pymnt_d`,
  etc.) that must be excluded from features.
  - Real `loan_status` value counts (via pandas, not naive CSV split — several free-text
    columns contain embedded commas): Fully Paid 1,059,885 / Current 871,037 / Charged Off
    266,014 / Default 39 / plus Late and Grace Period buckets, all excluded as non-terminal.
  - Row count (~2.26M) is below the spec's ~2.9M estimate — that estimate likely reflects a
    different mirror or includes rejected applications. This is still the correct
    accepted-loans dataset; not a blocker.
- MLflow backend store: **SQLite** (`sqlite:///mlflow.db`), not the default file store — file
  store does not support the model registry, which Phase 3/4 depend on. This is a hard
  constraint, not a style choice.

## 3. Repository layout

Spec §6 as written, plus:

```
creditpulse/
├── config.py              # paths, split dates, drift thresholds — single source of truth,
│                           # imported by data prep, drift engine, backtest, and MCP server
├── data/
│   ├── download.py         # (already done manually; kept for reproducibility)
│   └── prepare.py
├── model/
│   ├── train.py
│   └── features.py
├── drift/
│   ├── statistical.py
│   ├── autoencoder.py
│   └── engine.py
├── lifecycle/
│   ├── retrain.py
│   ├── evaluate.py
│   └── registry.py
├── mcp_server/
│   └── server.py
├── backtest/
│   ├── arms.py             # static / scheduled / monitored strategy implementations
│   └── run_backtest.py
├── reports/                 # generated charts + numbers, committed as build output
└── tests/
    ├── test_drift.py
    └── test_lifecycle_gating.py
```

`config.py` centralizes the temporal split dates and drift thresholds so the backtest can't
silently drift out of sync with the live drift-engine thresholds.

## 4. Data preparation and leak prevention

Pipeline: raw CSV → `prepare.py` (pandas, proper CSV quoting) → parquet, partitioned by
origination quarter → `features.py` → `train.py`.

Two leak points, handled explicitly:

- **Label leak.** Target = 1 if `loan_status` in `{Charged Off, Default}`, 0 if `Fully Paid`;
  all other statuses (`Current`, `Late (*)`, `In Grace Period`, `Does not meet the credit
  policy.*`) are dropped as non-terminal/unresolved, per spec §3. Feature set is built from an
  explicit **deny-list** of post-origination columns (payment history, recoveries, last credit
  pull, etc.) rather than an allow-list — an allow-list can silently admit a new leaky column
  if the schema changes; a deny-list fails loud instead.
- **Preprocessing leak.** The `ColumnTransformer` (imputation, encoding, scaling) is fit once
  on the pre-2019 reference window only, then persisted and reused unchanged for every
  evaluation window and by the autoencoder. Refitting per window would let drift leak into
  the preprocessing itself and invalidate champion-vs-challenger comparisons.

## 5. Drift engine

- PSI bin edges (for numeric features) are computed **once** from the reference (pre-2019)
  distribution and frozen for all subsequent windows. Recomputing quantile bins per window is
  a common bug that makes PSI read ~0 forever, since every window is compared to its own bins.
- Categorical features use frequency-based PSI with an explicit "unseen category" bucket.
- Thresholds: PSI > 0.10 = warning, > 0.25 = breach (standard credit-risk convention).
- Autoencoder: small MLP trained on standardized reference feature vectors. Drift signal is
  the reconstruction-error **percentile against the reference error distribution**, breaching
  at p99.
- `DriftReport.status` = the worse of the statistical and autoencoder signals, with
  per-feature PSI/KS attribution so `get_drift_report()` can name what's driving a breach.

## 6. Lifecycle and registry

- `registry.py` uses MLflow **aliases** (`champion` / `challenger`), not the deprecated stage
  API. `promote_challenger` reassigns the `champion` alias and records the displaced version so
  `rollback` has a well-defined target.
- Confirmation gating: `if confirm is not True: raise ConfirmationRequired(...)` — identity
  check, not truthiness, so `1`, `"true"`, `"yes"` are all refused, only the literal boolean
  `True` passes. `test_lifecycle_gating.py` asserts both that a call without `confirm=True`
  raises **and** that registry state is unchanged afterward — refusing is not enough if the
  refusal has side effects.

## 7. Backtest design

Three arms, walked monthly through 2019-01 → 2020-12, all starting from the same champion
trained on pre-2019 data:

1. **Static** — never retrains.
2. **Scheduled** — blind quarterly retrain, no drift awareness.
3. **Monitored** — retrains via the Phase 3 pipeline whenever the drift engine reports a
   breach.

The scheduled arm exists specifically to answer "would a dumb calendar job have gotten the
same benefit as monitoring?" — without it, an AUC-recovery number attributed to "monitoring"
could really just be attributable to "retraining eventually happened."

Each arm tracks AUC per window for the AUC-trajectory chart and lead-time measurement (spec
§8 Phase 5).

**Dollar-impact methodology.** To keep the comparison fair, all three arms accept the **same
loan volume** at each evaluation point — a fixed approval rate (accept the lowest-predicted-risk
N% of applicants), with each arm's own model choosing its own threshold to hit that rate. This
avoids the trivial failure mode where a model "prevents" more losses simply by rejecting more
loans. Realized defaults among each arm's accepted loans are compared using the dataset's
actual `loan_amnt` and outcome, normalized per 10,000 loans. The full calculation — approval
rate, per-arm threshold, accepted-loan default rate, dollar loss — is shown step-by-step in the
README, not collapsed into a single headline figure.

Every generated chart, printed summary, and README section states plainly that these are
backtest results on public historical data (spec §2).

## 8. Deliverables and review gates

- Phase 1 stops for review at the AUC-decline curve (pre-2019 test AUC vs. each 2019/2020
  quarter, no retraining). If the decline isn't visible, the project's premise is unsupported
  and we diagnose before building further phases on top of it.
- Phases 2–5 proceed per spec, `pytest` green before each phase handoff.
- Phase 5 final outputs: AUC trajectory chart (three arms), drift-detection lead time, AUC
  points recovered (monitored vs. static, and vs. scheduled), $-per-10,000-loans impact with
  calculation shown — written into `README.md` and also published as a shareable web Artifact
  with the interactive AUC trajectory and dollar-impact walkthrough.
- Phase 4 deliverable: a saved transcript of a full MCP lifecycle walkthrough (health → drift
  report → retrain → promote) driven from a real MCP client, embedded in the README.

## 9. Out of scope

- No live trading/lending integration of any kind — this is a backtest-only system, explicitly
  per spec §2.
- No auto-promotion path of any kind, even behind a feature flag — `confirm: true` is the only
  door, by design.
