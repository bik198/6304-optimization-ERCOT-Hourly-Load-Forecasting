"""Data strand: the Step 0 pipeline registered as beads, plus a loader for its saved output.

The Step 0 functions in `data_pipeline.py` are reused unchanged. `load_system_data()`
rebuilds the Step 0 `system_data` dictionary from `data/ercot_system_hourly.csv`, so the
model strands can start from the deduplicated hourly table instead of re-parsing the
619,968-row raw file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data_pipeline as step0
from .beads import bead, register
from .config import EXPECTED_ZONES, SYSTEM_CSV_PATH, WEATHER_COLUMNS

audit_and_deduplicate = register(
    "dedupe", step0.audit_and_deduplicate, strand="data",
    title="Audit & Deduplicate Zones",
    io=("619,968 × 13 raw", "77,496 × 21 hourly"),
    tokens=("619,968", "77,496", "8 zones", "dedupe", "DST", "missing weather", "0 MW spread", "raw data"),
    explains="Collapse the eight zone rows of each hour into one system hour and audit the raw table.",
)
daily_profiles = register(
    "daily_profiles", step0._build_daily_profiles, strand="data", after=("load_system_data",),
    title="Build 24h Daily Profiles",
    io=("77,496 hourly", "3,229 days × 24h"),
    tokens=("DST", "fall-back", "spring-forward", "24 hours", "daily profile", "Uri"),
    explains="Arrange each calendar day as 24 clock-hour loads, averaging fall-back hours and filling spring-forward hours.",
)
day_ahead_features = register(
    "day_ahead_features", step0.split_and_prepare_features, strand="data", after=("load_system_data",),
    title="Build 67 Inputs per Day",
    io=("77,496 hourly", "3,221 × 67 → 24h"),
    tokens=("67 features", "3,221", "leakage", "train scaler", "chronological split", "lags", "calendar"),
    explains="Build the 67 causal inputs and 24 targets for each forecast day, scaled on training days only.",
)


@bead("load_system_data", strand="data", after=("dedupe",),
      title="Load Clean Hourly Table",
      io=("ercot_system_hourly.csv", "77,496 hr arrays"),
      tokens=("77,496", "system load", "hourly", "data"),
      explains="Read the deduplicated hourly table that Step 0 saved.")
def load_system_data(path: Path = SYSTEM_CSV_PATH) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype={"timestamp": str, "source": str, "split": str})
    return {
        "timestamp": frame["timestamp"].tolist(),
        "dst_flag": frame["DSTFlag"].to_numpy(dtype=bool),
        "source": frame["source"].tolist(),
        "uri_flag": frame["uri_flag"].to_numpy(dtype=bool),
        "split": frame["split"].tolist(),
        "system_total_mw": frame["system_total_mw"].to_numpy(dtype=np.float64),
        "zone_load_sum_mw": frame["zone_load_sum_mw"].to_numpy(dtype=np.float64),
        "zone_loads": {z: frame[f"load_{z}_mw"].to_numpy(dtype=np.float64) for z in EXPECTED_ZONES},
        "weather_mean": {w: frame[f"{w}_mean"].to_numpy(dtype=np.float64) for w in WEATHER_COLUMNS},
    }


def day_split(day: date) -> str:
    """Chronological split of a calendar day, using the Step 0 boundaries."""
    return step0.assign_split(f"{day.isoformat()} 00:00:00")
