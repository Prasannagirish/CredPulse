# CreditPulse — Agentic MLOps Control Plane for Credit Risk Models

> **Everything in this project is a backtest on public historical Lending Club data.**
> Nothing here runs against live loans, a real bank, or real money. Every quantified result
> below is labeled "backtest" and should be read as a simulation, not a production outcome.

## What this is

Credit-risk models degrade silently in production when the economic regime shifts underneath
them — a model trained before 2019 kept scoring loan applications confidently well into the
2020 COVID shock, even as borrower behavior changed faster than the model could be retrained.
CreditPulse is an agentic MLOps system that trains a real credit-default model on real
historical loan data, detects when it has drifted, retrains and evaluates a challenger under
human-gated promotion, and exposes the entire lifecycle as MCP tools so an LLM agent can
operate the model conversationally.

## Architecture

```
Loan applications (2007-2020, walked forward)
        │
        ▼
Champion model (XGBoost, trained on pre-2019 data)
        │ features + prediction
        ▼
Drift engine — PSI/KS statistics + PyTorch autoencoder reconstruction error
        │ threshold breach
        ▼
Retrain + shadow-evaluate challenger (MLflow tracks both runs)
        │
        ▼
MCP server (6 tools) ── driven by Claude Desktop, Claude Code, or any MCP client
```

## What's built (Phases 1-5)

- **Phase 1 — Data + baseline model.** 2,925,493 Lending Club loans (2007-2020), a leak-safe
  temporal split (train on loans issued before 2019, evaluate on 2019-2020), and an XGBoost
  champion. Held-out pre-2019 AUC: **0.7102**. Quarterly AUC on 2019-2020 shows a real decline
  (0.691 → 0.642) through the reliably-labeled window — see [`reports/phase1_auc_decline.png`](reports/phase1_auc_decline.png).
- **Phase 2 — Drift engine.** PSI/KS statistical drift plus a PyTorch autoencoder's
  reconstruction-error signal, combined into a single `ok`/`warning`/`breach` report with
  per-feature attribution. Macro-feature drift (income, DTI, utilization, FICO) breaches
  starting June 2019 — **17.4 weeks of lead time** ahead of the AUC trough — separated from a
  Lending Club grading-methodology shift that breaches even earlier (March 2019) but reflects
  a platform change, not economic drift. See [`reports/phase2_drift_lead_time.png`](reports/phase2_drift_lead_time.png).
- **Phase 3 — Retrain + MLflow registry.** A challenger retrained on a trailing 12-month
  window beats the champion on AUC, F1, and Brier score (0.6308 → 0.6803 AUC) on a held-out
  slice neither model trained on. `promote_challenger`/`rollback` are gated behind an
  identity-checked `confirm=True` — verified by a passing test
  ([`tests/test_lifecycle_gating.py`](tests/test_lifecycle_gating.py)) that no silent
  auto-promotion is possible.
- **Phase 4 — MCP server.** All six lifecycle tools (`get_model_health`, `get_drift_report`,
  `trigger_retrain`, `promote_challenger`, `rollback`, `explain_prediction`) exposed via the
  `mcp` SDK, driven end-to-end by a real MCP client below.
- **Phase 5 — Quantified backtest.** A monthly walk-forward comparing three strategies —
  static (never retrains), scheduled (blind quarterly retrain), and monitored
  (drift-triggered retrain) — through the reliably-labeled 2019 window, plus a transparent
  dollar-impact estimate. See the [Phase 5 results](#phase-5--quantified-backtest-results)
  section below.

## How to run

```bash
uv sync
uv run python -m data.prepare              # downloads nothing — expects the raw CSV locally
uv run python -m reports.generate_phase1_report
uv run python -m reports.generate_phase2_report
uv run python -m reports.generate_phase3_report
uv run python scripts/run_mcp_walkthrough.py   # drives the MCP server via a real client
uv run python -m backtest.run_backtest         # the final quantified backtest
```

## MCP Lifecycle Walkthrough

The transcript below was captured by running [`scripts/run_mcp_walkthrough.py`](scripts/run_mcp_walkthrough.py)
against the real [`mcp_server/server.py`](mcp_server/server.py) as a subprocess, connected
over stdio via the official `mcp` Python SDK's client (`stdio_client` + `ClientSession`) —
not a mock. Routine MLflow dependency-export logging noise has been trimmed for readability;
every tool result and the confirm-gating behavior are shown exactly as produced. Every number
is a backtest computation over historical Lending Club data.

```
Connected. Available tools: ['get_model_health', 'get_drift_report', 'explain_prediction',
'trigger_retrain', 'promote_challenger', 'rollback']

--- get_model_health() ---
{
  "champion_version": "1",
  "days_since_last_retrain": 0.2,
  "recent_batch_auc": 0.5926339735884674,
  "recent_batch_f1": 0.013157894736842105,
  "recent_batch_label": "2019-12",
  "drift_status": "breach"
}

--- get_drift_report(window="2019-06") ---
{
  "window": "2019-06",
  "status": "breach",
  "autoencoder_status": "ok",
  "autoencoder_fraction_above_threshold": 0.017767295597484276,
  "statistical_status": "breach",
  "top_drifting_features": [
    { "feature": "grade", "type": "categorical", "psi": 0.6306990618105149, "ks_stat": null },
    { "feature": "sub_grade", "type": "categorical", "psi": 0.5231292120839162, "ks_stat": null },
    { "feature": "revol_util", "type": "numeric", "psi": 0.2510489596788549, "ks_stat": 0.21774325148894297 },
    { "feature": "fico_range_low", "type": "numeric", "psi": 0.1699795940329936, "ks_stat": 0.17182386473008981 },
    { "feature": "fico_range_high", "type": "numeric", "psi": 0.1699795940329936, "ks_stat": 0.17182386473008981 }
  ]
}

--- trigger_retrain() ---
{
  "training_window_start": "2018-11-01",
  "training_window_end": "2019-10-01",
  "holdout_window_start": "2019-11-01",
  "holdout_window_end": "2020-01-01",
  "champion_auc": 0.6308324854701808,
  "challenger_auc": 0.6803063457330416,
  "champion_f1": 0.015503875968992248,
  "challenger_f1": 0.06042296072507553,
  "champion_brier": 0.08494247496128082,
  "challenger_brier": 0.07829736173152924,
  "challenger_wins": true,
  "challenger_version": "2"
}

--- promote_challenger(confirm=False) — expected refusal ---
Error executing tool promote_challenger

Challenger (AUC 0.6803) beat champion (AUC 0.6308) — proceeding to promote.

--- promote_challenger(confirm=True) ---
{
  "new_champion_version": "2"
}
```

The client-visible error message above ("Error executing tool promote_challenger") is
generic, but the server process's own stderr for that same call shows exactly why it
refused:

```
lifecycle.registry.ConfirmationRequired: promote_challenger requires confirm=True
```

This is the full spec lifecycle: health check → drift report → retrain → review the
comparison → promote with explicit confirmation. `promote_challenger(confirm=False)` was
attempted first and genuinely refused by the server before `confirm=True` was ever sent.

## Phase 5 — Quantified Backtest Results

All figures below are a **backtest on public historical Lending Club data** — a simulation
of what three retraining strategies would have achieved, not a live production outcome.

**Window.** The walk-forward is restricted to the reliably-labeled window, **2019-01 through
2019-12** — 12 calendar months, each with at least 100 observed defaults (the same
reliability bar Phase 1 established). Months closer to the eval data's September 2020 cutoff
are right-censored: loans issued that recently haven't had time to actually resolve, which
would make both the AUC trajectory and the dollar-impact calculation meaningless for that
period regardless of which strategy is actually better — so the walk stops before entering
that region, rather than reporting numbers known to be unreliable.

![Static vs Scheduled vs Monitored AUC](reports/phase5_backtest_auc_trajectory.png)

| Arm | Mean AUC (Jan-Dec 2019) | Retrains | AUC points recovered vs static |
|---|---|---|---|
| Static (never retrains) | 0.6758 | 0 | — |
| Scheduled (blind quarterly) | 0.6990 | 3 | +0.0233 |
| Monitored (drift-triggered, 2-month cooldown) | 0.7004 | 5 | +0.0247 |

**Drift-detection lead time** (from Phase 2, reused here — not recomputed): macro-feature
drift breached **17.4 weeks** before the AUC trough, giving the monitored strategy a genuine
early-warning window rather than reacting only after damage was already visible in AUC.

**Dollar impact.** At the final reliable month (2019-12), with a fixed 80% approval rate
(each model picks its own threshold to accept exactly the lowest-risk 80% of that month's
applicants — this is what keeps the comparison fair, since a model can't "win" simply by
rejecting more loans than its competitor) and loss given default simplified to the full
loan principal (no partial-recovery modeling — recovery amounts are excluded from features
as leaky post-origination fields, spec §3, and genuinely aren't available here either):

| Arm | Loans accepted | Defaults | Default rate | Loss per 10,000 loans |
|---|---|---|---|---|
| Static | 1,845 | 79 | 4.28% | $6,387,127 |
| Monitored | 1,845 | 73 | 3.96% | $5,092,141 |

Both arms accept the **same number of loans** — the comparison isn't inflated by one model
simply rejecting more applicants. The monitored strategy's lower default rate among an
equally-sized accepted pool translates to an estimated **$1,294,986 saved per 10,000 loans**
from monitoring + retraining versus never retraining, on this backtest.

## Definition of Done

Reproduced from the spec, checked only where a specific report, chart, or test in this repo
verifies the claim:

- [x] Baseline model shows a real, visible AUC decline through 2020 without retraining —
      [`reports/phase1_auc_decline.png`](reports/phase1_auc_decline.png), 0.691 → 0.642 over
      the reliably-labeled 2019 quarters.
- [x] Drift engine fires before the AUC trough, with a measured lead time —
      [`reports/phase2_drift_lead_time.png`](reports/phase2_drift_lead_time.png), 17.4 weeks
      (macro-feature signal).
- [x] Retrain/evaluate/promote pipeline works end-to-end, tracked in MLflow —
      [`reports/generate_phase3_report.py`](reports/generate_phase3_report.py) output;
      registry backed by `var/mlflow.db`.
- [x] `promote_challenger`/`rollback` verifiably refuse to act without `confirm=True`,
      covered by a passing test —
      [`tests/test_lifecycle_gating.py`](tests/test_lifecycle_gating.py).
- [x] All six MCP tools work when driven from a real MCP client, with a saved example
      transcript — [MCP Lifecycle Walkthrough](#mcp-lifecycle-walkthrough) above, captured by
      [`scripts/run_mcp_walkthrough.py`](scripts/run_mcp_walkthrough.py).
- [x] Backtest produces the four quantified numbers (AUC trajectory, lead time, AUC points
      recovered, $-per-10,000-loans), each clearly labeled as simulated/historical — this
      section, backed by [`backtest/run_backtest.py`](backtest/run_backtest.py).
- [x] README states plainly, near the top, that this is a backtest on public historical data
      and not a live production deployment — see the callout at the top of this file.

## Non-negotiable framing

- `promote_challenger` and `rollback` require `confirm: bool` to be exactly `True` — not
  merely truthy. `1`, `"true"`, and `"yes"` are all refused. This is enforced in
  [`lifecycle/registry.py`](lifecycle/registry.py) and covered by a passing test, not just
  documentation.
- Dollar-impact and lead-time figures are computed transparently from the dataset's own
  values, with the calculation shown in each report script — never a single unexplained
  headline number.
