"""Evaluation strand: the shared 2026 test cohort and the comparison tables in the report.

Source logic: `shared_baselines()` in `optimization/ridge_solvers.ipynb`, cell 8 of
`optimization/feedforward_optimizers.ipynb` and `test_results()` in
`optimization/lstm_tuning.ipynb`, which each join the saved exports the same way.
Every function takes a frame with `timestamp`, `actual` and one column per forecast, so
fresh forecasts from the strand modules can be scored the same way as saved ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .beads import bead
from .config import ROOT_DIR
from .s09_metrics import errors

PRELIMINARY = ROOT_DIR / "preliminary"
RESULTS = ROOT_DIR / "optimization" / "results"
MODELS = ("lstm", "ffn_sgd", "ffn_armijo", "ridge", "svr", "previous_day", "previous_week")
LABELS = {
    "lstm": "LSTM ensemble", "ffn_sgd": "Feed-forward network, mini-batch SGD",
    "ffn_armijo": "Feed-forward network, Armijo GD", "ridge": "Ridge regression",
    "svr": "RBF support vector regression", "previous_day": "Previous day, same hour",
    "previous_week": "Previous week, same hour",
}
SEASONS = {"Jan–Feb": (1, 2), "Mar–May": (3, 4, 5), "Jun–Aug": (6, 7, 8), "Sep–Nov": (9, 10, 11)}


@bead("shared_cohort", strand="evaluate", after=("ridge_search", "svr_search", "ffn_forecast", "lstm_ensemble"),
      title="Match 301 Test Days",
      io=("ML + LSTM + FFN CSVs", "301 days / 7,224 hrs"),
      tokens=("301 days", "7,224 hours", "shared cohort", "2026 test", "comparison"),
      explains="Join every model's saved test forecasts on timestamp and keep the 301 days all of them cover.")
def shared_cohort() -> pd.DataFrame:
    ml = pd.read_csv(PRELIMINARY / "ercot_ml_24hour_forecasts.csv", parse_dates=["timestamp"])
    lstm = pd.read_csv(PRELIMINARY / "lstm_outputs" / "lstm_day_ahead_forecasts.csv", parse_dates=["timestamp"])
    ffn = pd.read_csv(RESULTS / "feedforward" / "test_forecasts.csv", parse_dates=["timestamp"])
    shared = ml.merge(lstm[lstm["split"] == "test"], on="timestamp", validate="one_to_one")
    shared = shared.merge(ffn.drop(columns="actual_mw"), on="timestamp", validate="one_to_one").sort_values("timestamp")
    days = shared.groupby(shared["timestamp"].dt.normalize()).size()
    if len(days) != 301 or not (days == 24).all():
        raise ValueError("The saved exports must share 301 complete forecast days.")
    np.testing.assert_array_equal(shared["actual_load_mw"].to_numpy(), shared["actual_mw"].to_numpy())
    return pd.DataFrame({
        "timestamp": shared["timestamp"].to_numpy(), "actual": shared["actual_mw"].to_numpy(),
        "lstm": shared["lstm_mw"].to_numpy(), "ffn_sgd": shared["sgd_mw"].to_numpy(),
        "ffn_armijo": shared["armijo_mw"].to_numpy(), "ridge": shared["ridge_forecast_mw"].to_numpy(),
        "svr": shared["svr_forecast_mw"].to_numpy(), "previous_day": shared["naive_prev_day_mw"].to_numpy(),
        "previous_week": shared["naive_prev_week_mw"].to_numpy(),
    })


def _models(frame, models):
    return [m for m in (models or MODELS) if m in frame.columns]


@bead("score_table", strand="evaluate", after=("shared_cohort", "errors"),
      title="Rank All 7 Forecasts",
      io=("7,224 hrs × 7 models", "ranked MAE/RMSE table"),
      tokens=("ranking", "1,691.65", "1,786.59", "2,294.08", "2,351.17", "2,393.23", "Table 8"),
      explains="Rank every forecast by MAE and RMSE on the same hours.")
def score_table(frame: pd.DataFrame, models=None) -> pd.DataFrame:
    rows = [{"model": m, "hours": len(frame), **errors(frame["actual"], frame[m])} for m in _models(frame, models)]
    return pd.DataFrame(rows).sort_values("mae_mw", ignore_index=True)


@bead("high_load", strand="evaluate", after=("shared_cohort",),
      title="Error at Top 10% and Peak Hours",
      io=("top 10% = 723 hrs", "peak 2026-07-22 17:00"),
      tokens=("peak", "91.1 GW", "2026-07-22", "top 10%", "723 hours", "high load", "Table 9"),
      explains="Score the top tenth of hours by measured load, and the error at the single highest hour.")
def high_load(frame: pd.DataFrame, models=None, share: float = 0.1) -> pd.DataFrame:
    n = int(np.ceil(share * len(frame)))
    top = frame.nlargest(n, "actual")
    peak = frame.loc[frame["actual"].idxmax()]
    return pd.DataFrame([{"model": m, "top_hours": n, "top_mae_mw": float((top[m] - top["actual"]).abs().mean()),
                          "peak_timestamp": peak["timestamp"], "peak_error_mw": float(peak[m] - peak["actual"])}
                         for m in _models(frame, models)])


@bead("mae_by_hour", strand="evaluate", after=("shared_cohort",),
      title="Error by Forecast Hour (0–23)",
      io=("7,224 hrs", "24-hour MAE curve"),
      tokens=("horizon", "hour of day", "555.73", "2,143.70", "lead hour"),
      explains="Average absolute error separately for each of the 24 forecast hours.")
def mae_by_hour(frame: pd.DataFrame, models=None) -> pd.DataFrame:
    cols = _models(frame, models)
    return frame[cols].sub(frame["actual"], axis=0).abs().groupby(frame["timestamp"].dt.hour).mean().rename_axis("hour")


@bead("seasonal_mae", strand="evaluate", after=("shared_cohort",),
      title="Seasonal Error Breakdown",
      io=("4 season blocks", "Jun-Aug 1,525.67 MW"),
      tokens=("seasonal", "summer", "Jun-Aug", "1,525.67", "Mar-May", "Table 10"),
      explains="Average absolute error within each block of months.")
def seasonal_mae(frame: pd.DataFrame, models=None) -> pd.DataFrame:
    cols, month = _models(frame, models), frame["timestamp"].dt.month
    rows = []
    for block, months in SEASONS.items():
        part = frame[month.isin(months)]
        rows.append({"months": block, "hours": len(part),
                     **{m: float((part[m] - part["actual"]).abs().mean()) for m in cols}})
    return pd.DataFrame(rows)


@bead("daily_wins", strand="evaluate", after=("shared_cohort",),
      title="Count Days Each Model Wins",
      io=("301 paired days", "LSTM beats Ridge 214/301"),
      tokens=("214", "71%", "daily wins", "head-to-head", "LSTM vs Ridge"),
      explains="Count the days on which one forecast's daily MAE beats another's.")
def daily_wins(frame: pd.DataFrame, model: str, rival: str) -> dict:
    day = frame["timestamp"].dt.normalize()
    daily = frame[[model, rival]].sub(frame["actual"], axis=0).abs().groupby(day).mean()
    wins = int((daily[model] < daily[rival]).sum())
    return {"model": model, "rival": rival, "days": len(daily), "wins": wins, "share": wins / len(daily)}
