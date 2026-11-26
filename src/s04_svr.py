"""RBF-SVR strand: evenly spaced training sample and the 12-setting validation grid.

Source logic: `preliminary/ercot_ridge_and_svr.ipynb` (section 7).
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .beads import bead
from .s02_hourly_features import HourlyExamples
from .s09_metrics import errors

MAX_SAMPLES = 6000
GRID = {"C": [1, 10, 100], "epsilon": [0.01, 0.1], "gamma": ["scale", 0.01]}
SELECTED = {"C": 10, "epsilon": 0.1, "gamma": 0.01}


@bead("svr_sample", strand="svr", after=("hourly_examples",),
      title="Evenly Spaced SVR Subsample",
      io=("51,552 train hrs", "6,000 spaced hrs"),
      tokens=("SVR", "6,000", "subsampling", "kernel"),
      explains="Take 6,000 training hours at evenly spaced positions to keep the kernel fit tractable.")
def svr_sample(ex: HourlyExamples, max_samples: int = MAX_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    train = ex.split("train")
    x, y = ex.x[train], ex.y[train]
    if len(x) <= max_samples:
        return x, y
    positions = np.linspace(0, len(x) - 1, max_samples, dtype=int)
    return x[positions], y[positions]


@bead("svr_fit", strand="svr", after=("svr_sample",),
      title="RBF-Kernel SVR Fit",
      io=("6,000 × 12", "RBF-SVR model"),
      tokens=("RBF", "SVR", "kernel", "C=10", "epsilon=0.1", "gamma=0.01"),
      explains="Fit an RBF-kernel SVR with standardized inputs and target.")
def svr_fit(x: np.ndarray, y: np.ndarray, C, epsilon, gamma) -> TransformedTargetRegressor:
    model = TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()),
                            ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma, cache_size=500))]),
        transformer=StandardScaler(),
    )
    return model.fit(x, y)


@bead("svr_search", strand="svr", after=("svr_fit",),
      title="12-Setting SVR Grid Search",
      io=("12-setting grid", "2,351.17 MW MAE"),
      tokens=("SVR", "2,351", "grid search", "validation"),
      explains="Try every C, epsilon and gamma in the grid and keep the lowest validation MAE.")
def svr_search(ex: HourlyExamples, grid=GRID) -> tuple[TransformedTargetRegressor, dict, list[dict]]:
    x, y = svr_sample(ex)
    val = ex.split("val")
    results, best, best_params, best_mae = [], None, None, float("inf")
    for params in ParameterGrid(grid):
        model = svr_fit(x, y, **params)
        score = errors(ex.y[val], model.predict(ex.x[val]))
        results.append({**params, **score})
        if score["mae_mw"] < best_mae:
            best, best_params, best_mae = model, params, score["mae_mw"]
    return best, best_params, results
