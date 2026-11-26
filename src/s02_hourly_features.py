"""Ridge/SVR strand inputs: 12 predictors per target hour, built on the Step 0 daily profiles.

Source logic: `preliminary/ercot_ridge_and_svr.ipynb` (section 2) and the vectorized
`hourly_examples()` in `optimization/ridge_solvers.ipynb`. Only days with 24 distinct
clock hours are used, and both the previous day and the same weekday one week earlier
must be complete. That excludes daylight-saving days and the days next to 2026-10-30.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .beads import bead

FEATURES = (
    "previous_day_same_hour", "previous_week_same_hour",
    "previous_day_mean", "previous_day_max", "previous_day_min",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
)


@dataclass(frozen=True)
class HourlyExamples:
    x: np.ndarray                 # (hours, 12)
    y: np.ndarray                 # (hours,) measured load, MW
    timestamps: pd.DatetimeIndex  # target hour of each row

    def split(self, name: str) -> np.ndarray:
        """Boolean mask: train 2018-2023, val 2024-2025, test 2026."""
        years = self.timestamps.year
        return {"train": years <= 2023, "val": years.isin([2024, 2025]), "test": years == 2026}[name]


@bead("complete_days", strand="ridge", after=("daily_profiles",),
      title="Filter Complete 24h Days",
      io=("3,229 daily", "complete 24h days"),
      tokens=("DST", "October 30", "complete days", "24h"),
      explains="Keep only days with 24 distinct clock hours, excluding daylight-saving days.")
def complete_days(profiles: dict[date, dict[str, Any]]) -> pd.DataFrame:
    days = sorted(d for d, p in profiles.items() if p["raw_hour_count"] == 24)
    return pd.DataFrame(np.vstack([profiles[d]["load_24"] for d in days]),
                        index=pd.DatetimeIndex(days), columns=range(24))


@bead("hourly_examples", strand="ridge", after=("complete_days",),
      title="Build 12 Inputs per Hour",
      io=("complete days", "(76,032, 12) hourly"),
      tokens=("12 features", "51,552", "17,256", "7,224", "lags", "calendar", "sine cosine"),
      explains="Turn each target hour into 12 predictors: yesterday, last week, yesterday's summary, and calendar.")
def hourly_examples(daily: pd.DataFrame) -> HourlyExamples:
    dates = daily.index
    yesterday = dates - pd.Timedelta(days=1)
    last_week = dates - pd.Timedelta(days=7)
    valid = yesterday.isin(dates) & last_week.isin(dates)
    dates = dates[valid]
    yesterday_load = daily.loc[yesterday[valid]].to_numpy()
    week_load = daily.loc[last_week[valid]].to_numpy()
    hours = np.tile(np.arange(24), len(dates))
    target_dates = dates.repeat(24)
    hour_angle = 2 * np.pi * hours / 24
    day_of_week = target_dates.day_of_week
    dow_angle = 2 * np.pi * day_of_week / 7
    month_angle = 2 * np.pi * target_dates.month / 12
    x = np.column_stack([
        yesterday_load.ravel(), week_load.ravel(),
        yesterday_load.mean(axis=1).repeat(24), yesterday_load.max(axis=1).repeat(24),
        yesterday_load.min(axis=1).repeat(24),
        np.sin(hour_angle), np.cos(hour_angle), np.sin(dow_angle), np.cos(dow_angle),
        np.sin(month_angle), np.cos(month_angle), day_of_week >= 5,
    ])
    return HourlyExamples(x, daily.loc[dates].to_numpy().ravel(), target_dates + pd.to_timedelta(hours, unit="h"))
