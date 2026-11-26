"""Forecast error metrics shared by every strand. Residual = forecast - actual, in MW."""

from __future__ import annotations

import numpy as np

from .beads import bead


@bead("errors", strand="evaluate",
      title="MAE & RMSE Error Metric",
      io=("actual, forecast MW", "MAE & RMSE in MW"),
      tokens=("MAE", "RMSE", "residual", "metrics", "MW"),
      explains="Score a forecast by mean absolute error and root mean squared error in MW.")
def errors(actual, forecast) -> dict[str, float]:
    residual = np.asarray(forecast, dtype=np.float64).ravel() - np.asarray(actual, dtype=np.float64).ravel()
    return {"mae_mw": float(np.abs(residual).mean()), "rmse_mw": float(np.sqrt(np.mean(residual ** 2)))}
