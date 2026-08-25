# CreditPulse Phase 2 — Drift Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user is
> executing this plan to learn the system, one step at a time — narrate what each step is
> doing and why before running it, don't just execute silently.

**Goal:** Build a drift-detection engine that combines per-feature statistical drift (PSI,
KS-test) with a PyTorch autoencoder's multivariate reconstruction-error signal into a single
`DriftReport`, then prove it fires a warning/breach ahead of the point where the Phase 1
champion's AUC actually craters — with the lead time measured in real weeks.

**Architecture:** `drift/statistical.py` computes PSI/KS against a **frozen** reference
distribution fit once on the pre-2019 window. `drift/autoencoder.py` trains a small PyTorch
MLP on the same `ColumnTransformer` output the Phase 1 model uses, and turns reconstruction
error into a drift signal via percentile-against-reference. `drift/engine.py` combines both
into one `DriftReport` with a threshold-based `"ok"|"warning"|"breach"` status and per-feature
attribution. A monthly walk-forward report (`reports/generate_phase2_report.py`) runs this
against the real 2019-2020 eval data and measures lead time against Phase 1's AUC trough.

**Tech Stack:** Python 3.12, pandas, numpy, scipy (`ks_2samp`), scikit-learn
(`ColumnTransformer` from Phase 1), torch, matplotlib, pytest. Builds directly on
`config.py`, `model/features.py`, `model/train.py`, `data/prepare.py` from Phase 1.

## Global Constraints

- PSI bin edges (numeric features) are computed **once** from the reference window and
  frozen for every comparison window — never recomputed per window. This is the classic PSI
  bug: recomputing bins from each window's own distribution makes PSI read ~0 forever, since
  every window is compared against itself.
- Categorical PSI uses reference frequency proportions with an explicit `"__UNSEEN__"`
  bucket for any category not present in the reference window.
- Thresholds (design doc §5): PSI > 0.10 = warning, PSI > 0.25 = breach.
- Autoencoder: trained once on standardized reference feature vectors (the same fitted
  `ColumnTransformer` output the Phase 1 model consumes — `model.features.build_preprocessor`).
  Drift signal = reconstruction-error **percentile against the reference error distribution**,
  with the 99th percentile of reference error as the per-row breach threshold.
- `DriftReport.status` = the worse of the statistical status and the autoencoder status
  (`ok < warning < breach`), with per-feature PSI/KS values retained for attribution.
- Every printed/plotted result is a backtest on historical, resolved-outcome data — label
  every report output as such, consistent with Phase 1.
- Test file name follows the spec's repo layout (§6): `tests/test_drift.py`, covering all
  three `drift/` modules.

---

### Task 1: Statistical drift signals (PSI + KS)

**Files:**
- Create: `drift/__init__.py`
- Create: `drift/statistical.py`
- Test: `tests/test_drift.py`
- Modify: `config.py` — add `PSI_WARNING`, `PSI_BREACH`, `MIN_RELIABLE_DEFAULTS`
- Modify: `reports/generate_phase1_report.py` — use `config.MIN_RELIABLE_DEFAULTS` instead
  of the local constant defined in Phase 1

**Interfaces:**
- Produces:
  - `StatisticalDriftReference` (dataclass) — fields `numeric_bin_edges: dict[str, np.ndarray]`,
    `numeric_reference_values: dict[str, np.ndarray]`, `categorical_reference_frequencies:
    dict[str, pd.Series]`.
  - `fit_statistical_reference(reference_df: pd.DataFrame, numeric_features: list[str],
    categorical_features: list[str], n_bins: int = 10) -> StatisticalDriftReference`
  - `psi_numeric(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float`
  - `psi_categorical(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float`
  - `ks_statistic(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float`
  - `compute_feature_drift(reference: StatisticalDriftReference, comparison_df: pd.DataFrame,
    numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame` — columns
    `["feature", "type", "psi", "ks_stat"]`, sorted by `psi` descending. `ks_stat` is `NaN`
    for categorical features. Task 3 (`drift/engine.py`) consumes this directly.

- [ ] **Step 1: Add PSI thresholds and reliability constant to `config.py`**

Append to `config.py`:

```python
# Drift thresholds (design doc §5)
PSI_WARNING = 0.10
PSI_BREACH = 0.25

# A quarter's default label is only trustworthy once this many defaults have actually been
# observed — quarters near the eval window's data cutoff are right-censored (loans haven't
# had time to default yet), which is why Phase 1's report treats them as unreliable.
MIN_RELIABLE_DEFAULTS = 100
```

- [ ] **Step 2: Update `reports/generate_phase1_report.py` to use the shared constant**

Find the line `MIN_RELIABLE_DEFAULTS = 100` inside `main()` and the import line at the top of
the file. Replace:

```python
from config import MLFLOW_TRACKING_URI, PROCESSED_DIR, REPORTS_DIR
```

with:

```python
from config import MIN_RELIABLE_DEFAULTS, MLFLOW_TRACKING_URI, PROCESSED_DIR, REPORTS_DIR
```

Then delete the local `MIN_RELIABLE_DEFAULTS = 100` line inside `main()` (the one directly
above `quarterly["reliable"] = quarterly["n_defaults"] >= MIN_RELIABLE_DEFAULTS`) — the name
now resolves to the imported constant instead.

- [ ] **Step 3: Run the Phase 1 test suite to confirm the refactor didn't break anything**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_train.py tests/test_prepare.py tests/test_features.py -v
```

Expected: all previously-passing tests still pass (this refactor only touches a report
script, not tested code, so this is a sanity check).

- [ ] **Step 4: Write the failing tests**

```python
# tests/test_drift.py
import numpy as np
import pandas as pd

from drift.statistical import (
    compute_feature_drift,
    fit_statistical_reference,
    ks_statistic,
    psi_categorical,
    psi_numeric,
)


def _reference_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "income": rng.normal(50000, 10000, size=n),
        "grade": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
    })


def test_psi_numeric_near_zero_for_identical_distribution():
    reference = fit_statistical_reference(_reference_df(), ["income"], [], n_bins=10)
    same_dist = pd.Series(np.random.default_rng(1).normal(50000, 10000, size=1000))
    assert psi_numeric(reference, "income", same_dist) < 0.05


def test_psi_numeric_large_for_shifted_distribution():
    reference = fit_statistical_reference(_reference_df(), ["income"], [], n_bins=10)
    shifted = pd.Series(np.random.default_rng(1).normal(20000, 10000, size=1000))
    assert psi_numeric(reference, "income", shifted) > 0.25


def test_psi_categorical_flags_unseen_category():
    reference = fit_statistical_reference(_reference_df(), [], ["grade"], n_bins=10)
    comparison = pd.Series(["D"] * 500 + ["A"] * 500)  # "D" never seen in reference
    assert psi_categorical(reference, "grade", comparison) > 0.25


def test_ks_statistic_large_for_shifted_distribution():
    reference = fit_statistical_reference(_reference_df(), ["income"], [], n_bins=10)
    shifted = pd.Series(np.random.default_rng(1).normal(20000, 10000, size=1000))
    assert ks_statistic(reference, "income", shifted) > 0.3


def test_compute_feature_drift_returns_all_features_sorted_by_psi():
    reference = fit_statistical_reference(_reference_df(), ["income"], ["grade"], n_bins=10)
    comparison_df = pd.DataFrame({
        "income": np.random.default_rng(1).normal(50000, 10000, size=1000),
        "grade": pd.Series(np.random.default_rng(1).choice(["A", "B", "C"], size=1000, p=[0.5, 0.3, 0.2])),
    })
    result = compute_feature_drift(reference, comparison_df, ["income"], ["grade"])
    assert set(result["feature"]) == {"income", "grade"}
    assert list(result.columns) == ["feature", "type", "psi", "ks_stat"]
    assert result["psi"].is_monotonic_decreasing
```

- [ ] **Step 5: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_drift.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'drift'`.

- [ ] **Step 6: Create the `drift` package and implement `drift/statistical.py`**

```bash
cd ~/IdeaProjects/mlops
touch drift/__init__.py
```

```python
# drift/statistical.py
"""Statistical drift signals: PSI and KS-test, computed against a frozen reference distribution."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_EPSILON = 1e-4  # smoothing to avoid log(0) / division by zero in PSI
_UNSEEN_CATEGORY = "__UNSEEN__"


@dataclass
class StatisticalDriftReference:
    """Frozen reference distribution for PSI/KS comparisons.

    Bin edges and reference values are computed once from the reference window and reused
    for every comparison window — recomputing them per window (e.g. from quantiles of the
    comparison data itself) is a common bug that makes PSI read ~0 forever, since every
    window would be compared against its own distribution.
    """
    numeric_bin_edges: dict[str, np.ndarray]
    numeric_reference_values: dict[str, np.ndarray]
    categorical_reference_frequencies: dict[str, pd.Series]


def fit_statistical_reference(
    reference_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    n_bins: int = 10,
) -> StatisticalDriftReference:
    numeric_bin_edges: dict[str, np.ndarray] = {}
    numeric_reference_values: dict[str, np.ndarray] = {}
    for feature in numeric_features:
        values = reference_df[feature].dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        numeric_bin_edges[feature] = edges
        numeric_reference_values[feature] = values

    categorical_reference_frequencies = {
        feature: reference_df[feature].value_counts(normalize=True)
        for feature in categorical_features
    }

    return StatisticalDriftReference(
        numeric_bin_edges=numeric_bin_edges,
        numeric_reference_values=numeric_reference_values,
        categorical_reference_frequencies=categorical_reference_frequencies,
    )


def _psi_from_proportions(ref_props: np.ndarray, comp_props: np.ndarray) -> float:
    ref_props = np.clip(ref_props, _EPSILON, None)
    comp_props = np.clip(comp_props, _EPSILON, None)
    return float(np.sum((comp_props - ref_props) * np.log(comp_props / ref_props)))


def psi_numeric(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float:
    edges = reference.numeric_bin_edges[feature]
    ref_values = reference.numeric_reference_values[feature]
    ref_counts, _ = np.histogram(ref_values, bins=edges)
    comp_counts, _ = np.histogram(comparison.dropna().to_numpy(), bins=edges)
    ref_props = ref_counts / ref_counts.sum()
    comp_props = comp_counts / max(comp_counts.sum(), 1)
    return _psi_from_proportions(ref_props, comp_props)


def psi_categorical(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float:
    ref_freq = reference.categorical_reference_frequencies[feature]
    categories = list(ref_freq.index) + [_UNSEEN_CATEGORY]
    comp_mapped = comparison.where(comparison.isin(ref_freq.index), _UNSEEN_CATEGORY)
    comp_freq = comp_mapped.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    ref_props = ref_freq.reindex(categories, fill_value=0.0).to_numpy()
    return _psi_from_proportions(ref_props, comp_freq.to_numpy())


def ks_statistic(reference: StatisticalDriftReference, feature: str, comparison: pd.Series) -> float:
    ref_values = reference.numeric_reference_values[feature]
    comp_values = comparison.dropna().to_numpy()
    return float(ks_2samp(ref_values, comp_values).statistic)


def compute_feature_drift(
    reference: StatisticalDriftReference,
    comparison_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in numeric_features:
        rows.append({
            "feature": feature,
            "type": "numeric",
            "psi": psi_numeric(reference, feature, comparison_df[feature]),
            "ks_stat": ks_statistic(reference, feature, comparison_df[feature]),
        })
    for feature in categorical_features:
        rows.append({
            "feature": feature,
            "type": "categorical",
            "psi": psi_categorical(reference, feature, comparison_df[feature]),
            "ks_stat": float("nan"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_drift.py -v
```

Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
cd ~/IdeaProjects/mlops
git add drift/__init__.py drift/statistical.py tests/test_drift.py config.py reports/generate_phase1_report.py
git commit -m "Add PSI/KS statistical drift signals with frozen reference bins"
```

---

### Task 2: PyTorch autoencoder reconstruction-error signal

**Files:**
- Create: `drift/autoencoder.py`
- Test: `tests/test_drift.py` (append)

**Interfaces:**
- Produces:
  - `DriftAutoencoder(nn.Module)` — constructor `__init__(self, input_dim: int, hidden_dim:
    int = 16, bottleneck_dim: int = 8)`. `forward(x: torch.Tensor) -> torch.Tensor`.
  - `train_autoencoder(X: np.ndarray, epochs: int = 30, lr: float = 1e-3, batch_size: int =
    256) -> DriftAutoencoder` — `X` is a dense float array (the fitted `ColumnTransformer`
    output). Fits on `X` only; caller is responsible for passing only reference-window data.
  - `reconstruction_error(model: DriftAutoencoder, X: np.ndarray) -> np.ndarray` — one
    mean-squared-error value per row of `X`. Task 3 (`drift/engine.py`) calls this both to
    build the reference error distribution and to score comparison windows.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_drift.py
from drift.autoencoder import reconstruction_error, train_autoencoder


def test_reconstruction_error_lower_on_reference_than_random_noise():
    rng = np.random.default_rng(0)
    X_reference = rng.normal(0, 1, size=(500, 10)).astype(np.float32)
    model = train_autoencoder(X_reference, epochs=20)

    reference_errors = reconstruction_error(model, X_reference)
    noise_errors = reconstruction_error(model, rng.normal(10, 5, size=(500, 10)).astype(np.float32))

    assert reference_errors.mean() < noise_errors.mean()


def test_reconstruction_error_returns_one_value_per_row():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(50, 6)).astype(np.float32)
    model = train_autoencoder(X, epochs=5)
    errors = reconstruction_error(model, X)
    assert errors.shape == (50,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_drift.py -v -k autoencoder
```

Expected: FAIL with `ModuleNotFoundError: No module named 'drift.autoencoder'`.

- [ ] **Step 3: Implement `drift/autoencoder.py`**

```python
# drift/autoencoder.py
"""PyTorch autoencoder producing a multivariate drift signal via reconstruction error."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DriftAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X: np.ndarray, epochs: int = 30, lr: float = 1e-3, batch_size: int = 256
) -> DriftAutoencoder:
    """Train on standardized reference feature vectors — the same fitted ColumnTransformer
    output the Phase 1 champion model consumes (model.features.build_preprocessor). Fit
    only on the reference window; never on a comparison window."""
    torch.manual_seed(42)
    model = DriftAutoencoder(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

    return model


def reconstruction_error(model: DriftAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = model(X_tensor)
        errors = ((reconstructed - X_tensor) ** 2).mean(dim=1)
    return errors.numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_drift.py -v -k autoencoder
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/IdeaProjects/mlops
git add drift/autoencoder.py tests/test_drift.py
git commit -m "Add PyTorch autoencoder reconstruction-error drift signal"
```

---

### Task 3: Drift engine combining both signals into a DriftReport

**Files:**
- Create: `drift/engine.py`
- Test: `tests/test_drift.py` (append)
- Modify: `config.py` — add `AE_ERROR_PERCENTILE`, `AE_WARNING_FRACTION`, `AE_BREACH_FRACTION`

**Interfaces:**
- Consumes: `StatisticalDriftReference`, `compute_feature_drift` from `drift.statistical`;
  `DriftAutoencoder`, `reconstruction_error` from `drift.autoencoder`; `PSI_WARNING`,
  `PSI_BREACH`, `AE_ERROR_PERCENTILE`, `AE_WARNING_FRACTION`, `AE_BREACH_FRACTION` from
  `config`.
- Produces:
  - `DriftReport` (dataclass) — fields `window: str`, `status: str`, `feature_drift:
    pd.DataFrame`, `autoencoder_fraction_above_threshold: float`, `autoencoder_status: str`,
    `statistical_status: str`.
  - `evaluate_drift(window: str, statistical_reference: StatisticalDriftReference,
    autoencoder: DriftAutoencoder, reference_errors: np.ndarray, comparison_df: pd.DataFrame,
    comparison_X: np.ndarray, numeric_features: list[str], categorical_features: list[str])
    -> DriftReport`. `reference_errors` is `reconstruction_error(autoencoder, reference_X)`
    computed once by the caller and passed in — Task 4's report script computes it once per
    run, not once per window. Task 4 calls this directly, once per monthly window.

- [ ] **Step 1: Add autoencoder thresholds to `config.py`**

Append to `config.py`:

```python
# Autoencoder drift thresholds. By construction, ~1% of in-distribution rows exceed the
# reference set's own 99th-percentile reconstruction error — these fractions describe how
# much larger than that 1% baseline a window's exceedance rate must be to count as drift.
AE_ERROR_PERCENTILE = 0.99
AE_WARNING_FRACTION = 0.02
AE_BREACH_FRACTION = 0.05
```

- [ ] **Step 2: Write the failing tests**

```python
# append to tests/test_drift.py
from drift.engine import evaluate_drift
from drift.statistical import fit_statistical_reference as _fit_ref  # already imported above; reuse


def _make_reference_and_stats(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "income": rng.normal(50000, 10000, size=n),
        "grade": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),
    })
    return df


def _encode(df: pd.DataFrame, income_mean: float, income_std: float) -> np.ndarray:
    return np.column_stack([
        (df["income"] - income_mean) / income_std,
        (df["grade"] == "A").astype(float),
        (df["grade"] == "B").astype(float),
    ]).astype(np.float32)


def test_evaluate_drift_returns_ok_when_comparison_matches_reference():
    reference_df = _make_reference_and_stats(seed=0)
    income_mean, income_std = reference_df["income"].mean(), reference_df["income"].std()
    reference_X = _encode(reference_df, income_mean, income_std)

    comparison_df = _make_reference_and_stats(seed=1)
    comparison_X = _encode(comparison_df, income_mean, income_std)

    statistical_reference = fit_statistical_reference(reference_df, ["income"], ["grade"], n_bins=10)
    autoencoder = train_autoencoder(reference_X, epochs=20)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    report = evaluate_drift(
        window="test-window",
        statistical_reference=statistical_reference,
        autoencoder=autoencoder,
        reference_errors=reference_errors,
        comparison_df=comparison_df,
        comparison_X=comparison_X,
        numeric_features=["income"],
        categorical_features=["grade"],
    )
    assert report.status == "ok"


def test_evaluate_drift_returns_breach_when_features_shifted():
    reference_df = _make_reference_and_stats(seed=0)
    income_mean, income_std = reference_df["income"].mean(), reference_df["income"].std()
    reference_X = _encode(reference_df, income_mean, income_std)

    rng = np.random.default_rng(2)
    shifted_df = pd.DataFrame({
        "income": rng.normal(15000, 5000, size=1000),
        "grade": rng.choice(["D", "E"], size=1000),  # entirely unseen categories
    })
    shifted_X = _encode(shifted_df, income_mean, income_std)

    statistical_reference = fit_statistical_reference(reference_df, ["income"], ["grade"], n_bins=10)
    autoencoder = train_autoencoder(reference_X, epochs=20)
    reference_errors = reconstruction_error(autoencoder, reference_X)

    report = evaluate_drift(
        window="test-window",
        statistical_reference=statistical_reference,
        autoencoder=autoencoder,
        reference_errors=reference_errors,
        comparison_df=shifted_df,
        comparison_X=shifted_X,
        numeric_features=["income"],
        categorical_features=["grade"],
    )
    assert report.status == "breach"
    assert report.statistical_status == "breach"
```

Remove the placeholder `_fit_ref` import alias line — it isn't needed since
`fit_statistical_reference` is already imported at the top of `tests/test_drift.py` from
Task 1. Only add the line `from drift.engine import evaluate_drift` to the existing import
block at the top of the file (alongside the Task 1 and Task 2 imports), not a new import
block.

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_drift.py -v -k evaluate_drift
```

Expected: FAIL with `ModuleNotFoundError: No module named 'drift.engine'`.

- [ ] **Step 4: Implement `drift/engine.py`**

```python
# drift/engine.py
"""Combines statistical and autoencoder drift signals into a single DriftReport."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    AE_BREACH_FRACTION,
    AE_ERROR_PERCENTILE,
    AE_WARNING_FRACTION,
    PSI_BREACH,
    PSI_WARNING,
)
from drift.autoencoder import DriftAutoencoder, reconstruction_error
from drift.statistical import StatisticalDriftReference, compute_feature_drift

_STATUS_ORDER = {"ok": 0, "warning": 1, "breach": 2}


@dataclass
class DriftReport:
    window: str
    status: str
    feature_drift: pd.DataFrame
    autoencoder_fraction_above_threshold: float
    autoencoder_status: str
    statistical_status: str


def _statistical_status(feature_drift: pd.DataFrame) -> str:
    if (feature_drift["psi"] > PSI_BREACH).any():
        return "breach"
    if (feature_drift["psi"] > PSI_WARNING).any():
        return "warning"
    return "ok"


def _autoencoder_status(fraction_above_threshold: float) -> str:
    if fraction_above_threshold > AE_BREACH_FRACTION:
        return "breach"
    if fraction_above_threshold > AE_WARNING_FRACTION:
        return "warning"
    return "ok"


def evaluate_drift(
    window: str,
    statistical_reference: StatisticalDriftReference,
    autoencoder: DriftAutoencoder,
    reference_errors: np.ndarray,
    comparison_df: pd.DataFrame,
    comparison_X: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> DriftReport:
    feature_drift = compute_feature_drift(
        statistical_reference, comparison_df, numeric_features, categorical_features
    )
    statistical_status = _statistical_status(feature_drift)

    threshold = np.percentile(reference_errors, AE_ERROR_PERCENTILE * 100)
    comparison_errors = reconstruction_error(autoencoder, comparison_X)
    fraction_above_threshold = float((comparison_errors > threshold).mean())
    autoencoder_status = _autoencoder_status(fraction_above_threshold)

    overall_status = max([statistical_status, autoencoder_status], key=lambda s: _STATUS_ORDER[s])

    return DriftReport(
        window=window,
        status=overall_status,
        feature_drift=feature_drift,
        autoencoder_fraction_above_threshold=fraction_above_threshold,
        autoencoder_status=autoencoder_status,
        statistical_status=statistical_status,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/IdeaProjects/mlops
uv run pytest tests/test_drift.py -v
```

Expected: 9 passed (5 from Task 1, 2 from Task 2, 2 from this task).

- [ ] **Step 6: Commit**

```bash
cd ~/IdeaProjects/mlops
git add drift/engine.py tests/test_drift.py config.py
git commit -m "Add DriftReport combining statistical and autoencoder signals"
```

---

### Task 4: Monthly walk-forward against real data, lead-time measurement

**Files:**
- Create: `reports/generate_phase2_report.py`

**Interfaces:**
- Consumes: `fit_statistical_reference` from `drift.statistical`; `train_autoencoder`,
  `reconstruction_error` from `drift.autoencoder`; `evaluate_drift` from `drift.engine`;
  `build_preprocessor` from `model.features`; `train_baseline`, `quarterly_auc` from
  `model.train`; `config.PROCESSED_DIR`, `config.REPORTS_DIR`, `config.NUMERIC_FEATURES`,
  `config.CATEGORICAL_FEATURES`, `config.MIN_RELIABLE_DEFAULTS`.
- Produces: a standalone script producing `reports/phase2_drift_lead_time.png` and a printed
  monthly status table + lead-time number. No other module depends on this script's internals.

- [ ] **Step 1: Write the report script**

```python
# reports/generate_phase2_report.py
"""Phase 2 deliverable: monthly drift status through 2019-2020, and the lead time between
the first drift warning/breach and the point where Phase 1's champion AUC actually craters.
All numbers below are a backtest on public historical Lending Club data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CATEGORICAL_FEATURES,
    MIN_RELIABLE_DEFAULTS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from drift.autoencoder import reconstruction_error, train_autoencoder
from drift.engine import evaluate_drift
from drift.statistical import fit_statistical_reference
from model.features import build_preprocessor
from model.train import quarterly_auc, train_baseline

_STATUS_RANK = {"ok": 0, "warning": 1, "breach": 2}


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
        rows.append({"month": month, "status": report.status, "n_loans": len(group)})
        if first_alert_month is None and report.status != "ok":
            first_alert_month = month

    monthly = pd.DataFrame(rows)
    print("\n=== BACKTEST ON HISTORICAL DATA — not a live evaluation ===")
    print("Monthly drift status, 2019-2020 (reference = pre-2019 window, never retrained):")
    print(monthly.to_string(index=False))

    if first_alert_month is not None:
        alert_date = pd.Period(first_alert_month, freq="M").start_time
        lead_time_weeks = (trough_date - alert_date).days / 7
        print(f"\nFirst drift alert: {first_alert_month} (starts {alert_date.date()})")
        print(f"Lead time vs AUC trough ({trough_quarter}): {lead_time_weeks:.1f} weeks")
    else:
        print("\nNo drift alert fired at any point in the eval window.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    status_numeric = monthly["status"].map(_STATUS_RANK)
    ax.step(monthly["month"], status_numeric, where="mid", marker="o")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["ok", "warning", "breach"])
    ax.set_xlabel("Month")
    ax.set_title("Monthly Drift Status — Backtest on Historical Lending Club Data")
    if first_alert_month is not None:
        ax.axvline(first_alert_month, color="red", linestyle="--", label=f"first alert ({first_alert_month})")
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
```

- [ ] **Step 2: Run the report against the real prepared data**

```bash
cd ~/IdeaProjects/mlops
uv run python -m reports.generate_phase2_report
```

Expected: prints the AUC trough quarter/date (should match Phase 1's finding — 2020Q1 was
the lowest-AUC reliable quarter), then a monthly `ok`/`warning`/`breach` table across
2019-01 through 2020-09, the first alert month, and a lead-time-in-weeks number. Saves
`reports/phase2_drift_lead_time.png`.

**Stop here and inspect the result before proceeding to Phase 3** — the spec's Phase 2
acceptance criterion (§8) is that the drift status starts firing in early-to-mid 2020, ahead
of where AUC bottoms out. If the first alert month is at or after the AUC trough month, or if
every month reads `"ok"`, that's a real finding to diagnose (check: is `reference_X` /
`comparison_X` actually using the same fitted `preprocessor` instance in both calls — a
common bug is accidentally calling `.fit_transform()` twice instead of `.fit_transform()`
once then `.transform()`; check whether the autoencoder thresholds in `config.py` are too
loose for this feature set's scale) rather than silently accepting a report that doesn't
support the claim.

- [ ] **Step 3: Commit**

```bash
cd ~/IdeaProjects/mlops
git add reports/generate_phase2_report.py reports/phase2_drift_lead_time.png
git commit -m "Add monthly drift walk-forward report with lead-time measurement vs AUC trough"
```

---

## Self-review notes

- **Spec coverage:** `drift/statistical.py` (PSI + KS per feature, frozen reference) ✓;
  `drift/autoencoder.py` (PyTorch AE, reconstruction-error signal) ✓; `drift/engine.py`
  (combined `DriftReport` with threshold-based status) ✓; acceptance criterion — drift fires
  ahead of AUC cratering, lead time measured — covered by Task 4's report script, which
  explicitly computes and prints the lead time rather than leaving it as a manual read of a
  chart ✓. Test file name matches spec §6 (`tests/test_drift.py`) ✓.
- **Placeholder scan:** none remaining — the one draft `_fit_ref` import alias in Task 3's
  test step was explicitly flagged for removal, with the corrected single-line addition
  spelled out.
- **Type consistency:** `StatisticalDriftReference`, `DriftAutoencoder`, and `DriftReport`
  are defined once (Tasks 1, 2, 3 respectively) and referenced with matching names/types in
  every later task and in `reports/generate_phase2_report.py`. `evaluate_drift`'s parameter
  list in Task 3's implementation matches every call site in Task 3's tests and Task 4's
  report script exactly (`window`, `statistical_reference`, `autoencoder`, `reference_errors`,
  `comparison_df`, `comparison_X`, `numeric_features`, `categorical_features`).
