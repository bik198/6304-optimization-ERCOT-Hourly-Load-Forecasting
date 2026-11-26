"""Ridge strand: alpha search on validation and the Cholesky reference fit.

Source logic: `preliminary/ercot_ridge_and_svr.ipynb` (section 6) and the reference fit
in `optimization/ridge_solvers.ipynb`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .beads import bead
from .s02_hourly_features import HourlyExamples
from .s09_metrics import errors

ALPHAS = (0.1, 1, 10, 100, 1000)
SELECTED_ALPHA = 0.1


@bead("ridge_search", strand="ridge", after=("hourly_examples",),
      title="Ridge Alpha Validation Sweep",
      io=("train/val 12-feat", "best α = 0.1"),
      tokens=("Ridge", "alpha", "0.1", "2,294", "2,307.95", "regularization", "L2"),
      explains="Fit standardized Ridge for each alpha and keep the lowest validation MAE.")
def ridge_search(ex: HourlyExamples, alphas=ALPHAS) -> tuple[Pipeline, list[dict]]:
    train, val = ex.split("train"), ex.split("val")
    results, best, best_mae = [], None, float("inf")
    for alpha in alphas:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(ex.x[train], ex.y[train])
        score = errors(ex.y[val], model.predict(ex.x[val]))
        results.append({"alpha": alpha, **score})
        if score["mae_mw"] < best_mae:
            best, best_mae = model, score["mae_mw"]
    return best, results


@dataclass(frozen=True)
class ScaledProblem:
    scaler: StandardScaler
    scaled: np.ndarray      # all rows, standardized with training statistics
    reference: Ridge        # Cholesky solution, the optimum the solvers are measured against


@bead("ridge_reference", strand="ridge", after=("hourly_examples",),
      title="Cholesky Reference Fit",
      io=("train scaled", "Cholesky w* (12)"),
      tokens=("Cholesky", "exact optimum", "StandardScaler", "Ridge", "reference"),
      explains="Standardize on training hours and solve Ridge exactly with Cholesky.")
def ridge_reference(ex: HourlyExamples, alpha: float = SELECTED_ALPHA) -> ScaledProblem:
    train = ex.split("train")
    scaler = StandardScaler().fit(ex.x[train])
    scaled = scaler.transform(ex.x)
    return ScaledProblem(scaler, scaled, Ridge(alpha=alpha, solver="cholesky").fit(scaled[train], ex.y[train]))
