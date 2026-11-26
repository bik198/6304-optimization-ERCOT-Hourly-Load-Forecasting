"""LSTM strand, data half: day grid, calendar, sequence samples and training-only scaling.

Source logic: `preliminary/ercot_lstm_forecast.ipynb` (cells 9-14). The notebook builds its
own 24-slot day grid. Here the grid comes from the Step 0 daily profiles (fall-back hours
averaged, spring-forward hours interpolated) and is identical. A whole missing day
(2026-10-30) is filled with the same hours one week earlier and is never scored.
The hindsight weather upper-bound variant is not carried over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar

from .beads import bead
from .s01_system_data import day_split

CONFIG: dict[str, Any] = {
    "mode": "day", "lookback": 14, "level_days": 7,
    "hidden": 64, "layers": 2, "dropout": 0.2, "head_units": 128,
    "optimizer": "sgd_nesterov", "lr": 0.03, "momentum": 0.9, "batch": 32, "wd": 0.0, "clip": 1.0,
    "lr_factor": 0.5, "lr_patience": 8, "max_epochs": 200, "patience": 25,
}


@dataclass(frozen=True)
class DayGrid:
    days: pd.DatetimeIndex
    load: np.ndarray        # (days, 24) MW
    target_ok: np.ndarray   # day has measured load and may be scored
    uri: np.ndarray         # day touches the Winter Storm Uri window
    split: np.ndarray       # "train" / "val" / "test"


@bead("day_grid", strand="lstm", after=("daily_profiles",),
      title="Continuous 24h Day Grid",
      io=("3,229 daily", "3,230 × 24 continuous"),
      tokens=("October 30", "day grid", "missing day", "LSTM"),
      explains="Lay every calendar day on a 24-slot grid; a missing day borrows last week's hours as input only.")
def day_grid(profiles: dict[date, dict[str, Any]]) -> DayGrid:
    first, last = min(profiles), max(profiles)
    days = pd.date_range(first, last, freq="D")
    load = np.empty((len(days), 24))
    ok = np.zeros(len(days), dtype=bool)
    uri = np.zeros(len(days), dtype=bool)
    for i, d in enumerate(days.date):
        if d in profiles:
            load[i], ok[i], uri[i] = profiles[d]["load_24"], True, profiles[d]["is_uri"]
        else:
            load[i] = load[i - 7]
    return DayGrid(days, load, ok, uri, np.array([day_split(d) for d in days.date]))


def _cyc(v, period):
    return np.sin(2 * np.pi * v / period), np.cos(2 * np.pi * v / period)


@dataclass(frozen=True)
class Calendar:
    dow: np.ndarray
    offday: np.ndarray
    doy_s: np.ndarray
    doy_c: np.ndarray
    day_cal: np.ndarray     # (days, 5): dow sin/cos, off-day, day-of-year sin/cos


@bead("lstm_calendar", strand="lstm", after=("day_grid",),
      title="Cyclical Calendar & Holidays",
      io=("3,230 dates", "5-col calendar + holidays"),
      tokens=("holidays", "calendar", "off-day", "sine cosine"),
      explains="Encode weekday and day-of-year as sine/cosine pairs and flag weekends and federal holidays.")
def lstm_calendar(grid: DayGrid) -> Calendar:
    days = grid.days
    holidays = USFederalHolidayCalendar().holidays(days.min(), days.max())
    dow = days.day_of_week.to_numpy()
    offday = ((dow >= 5) | days.isin(holidays)).astype(float)
    dow_s, dow_c = _cyc(dow, 7)
    doy_s, doy_c = _cyc(days.day_of_year.to_numpy(), 365.25)
    return Calendar(dow, offday, doy_s, doy_c, np.column_stack([dow_s, dow_c, offday, doy_s, doy_c]))


@bead("lstm_samples", strand="lstm", after=("lstm_calendar",),
      title="14-Day Normalized Windows",
      io=("14-day window", "relative load / 7d mean"),
      tokens=("14-day lookback", "7-day mean", "level normalization", "LSTM", "sequence"),
      explains="For each target day, stack the previous 14 days divided by the trailing 7-day mean load.")
def lstm_samples(grid: DayGrid, cal: Calendar, lookback: int, level_days: int = 7) -> dict[str, np.ndarray]:
    L = grid.load
    rows = np.array([t for t in range(max(lookback, level_days), len(L)) if grid.target_ok[t]])
    level = np.array([L[t - level_days:t].mean() for t in rows])
    seq_load = np.stack([L[t - lookback:t] for t in rows]) / level[:, None, None]
    static = np.column_stack([np.eye(7)[cal.dow[rows]], cal.offday[rows], cal.doy_s[rows], cal.doy_c[rows],
                              np.log(level)])
    return dict(rows=rows, level=level, seq_load=seq_load,
                seq_cal=np.stack([cal.day_cal[t - lookback:t] for t in rows]),
                static=static, y_rel=L[rows] / level[:, None], split=grid.split[rows],
                uri=np.array([grid.uri[t - lookback:t + 1].any() for t in rows]))


class Scaler:
    """Standardization fit on training samples only; one scale shared by load inputs and target."""

    def fit(self, S, train_mask):
        self.mu_load, self.sd_load = S["seq_load"][train_mask].mean(), S["seq_load"][train_mask].std()
        self.mu_y, self.sd_y = self.mu_load, self.sd_load
        self.mu_st, self.sd_st = S["static"][train_mask].mean(0), S["static"][train_mask].std(0)
        self.sd_st[self.sd_st == 0] = 1.0
        self.mu_st[:8], self.sd_st[:8] = 0.0, 1.0   # weekday one-hot and off-day flag stay unscaled
        return self

    def transform(self, S):
        seq = np.concatenate([(S["seq_load"] - self.mu_load) / self.sd_load, S["seq_cal"]], axis=2)
        static = (S["static"] - self.mu_st) / self.sd_st
        y = (S["y_rel"] - self.mu_y) / self.sd_y
        return seq.astype(np.float32), static.astype(np.float32), y.astype(np.float32)

    def to_mw(self, y_std, level):
        return (y_std * self.sd_y + self.mu_y) * level[:, None]


@bead("lstm_tensors", strand="lstm", after=("lstm_samples",),
      title="Uri Exclusion & Tensor Split",
      io=("sequences + Uri mask", "2,156 train / 731 val / 307 test"),
      tokens=("Winter Storm Uri", "Uri exclusion", "21 windows", "2,156", "tensors", "leakage"),
      explains="Drop training samples that touch Winter Storm Uri, fit the scaler on training days, and build tensors.")
def lstm_tensors(grid: DayGrid, cal: Calendar, cfg: dict, exclude_uri_train: bool = True) -> dict[str, Any]:
    S = lstm_samples(grid, cal, cfg["lookback"], cfg["level_days"])
    train = (S["split"] == "train") & ~(S["uri"] if exclude_uri_train else np.zeros_like(S["uri"]))
    scaler = Scaler().fit(S, train)
    seq, static, y = scaler.transform(S)
    masks = {"train": train, "val": S["split"] == "val", "test": S["split"] == "test"}
    tensors = {k: tuple(torch.tensor(a[m]) for a in (seq, static, y)) for k, m in masks.items()}
    return dict(S=S, scaler=scaler, masks=masks, tensors=tensors, n_seq=seq.shape[2], n_static=static.shape[1])
