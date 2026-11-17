"""Step 0 data audit, deduplication, and leakage-safe feature pipeline.

Why no pandas?
    On Windows machines where Application Control policies block unsigned compiled
    C-extensions (such as pandas's `nattype` DLL), `import pandas` fails. Using
    Python's built-in `csv` + `datetime` modules alongside `numpy` keeps this entire
    data pipeline fast, deterministic, and dependency-safe everywhere.

Main Workflow:
    1. `audit_and_deduplicate()` reads all 619,968 raw rows in a single streaming pass,
       verifies that `system_total_mw` repeats identically across the 8 ERCOT weather
       zones, audits DST fall-back hours / missing dates / missing weather values, and
       collapses the table into 77,496 unique hourly system records.
    2. `split_and_prepare_features()` builds day-ahead forecasting pairs (at the end
       of day `d`, predict all 24 hours of day `d+1`) using strictly causal past load
       and deterministic calendar variables, while fitting scalers strictly on `train`.
    3. `save_system_csv()` exports the clean 77,496-row hourly system table.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    EXPECTED_COLUMNS,
    EXPECTED_TIMESTAMPS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_ZONES,
    TEST_START,
    TRAIN_END,
    VAL_START,
    WEATHER_COLUMNS,
)


def _parse_float(value: str) -> float:
    """Convert a raw CSV cell string to float, returning NaN for empty or NA markers."""
    s = value.strip()
    if s == "" or s.lower() in {"na", "nan", "null", "none"}:
        return float("nan")
    return float(s)


def _parse_bool(value: str) -> bool:
    """Convert a CSV boolean flag string ('True'/'False') to a Python bool."""
    return value.strip().lower() == "true"


def assign_split(ts_str: str) -> str:
    """Assign each timestamp strictly chronologically to 'train', 'val', or 'test'.

    ISO-8601 strings ("YYYY-MM-DD HH:MM:SS") compare cleanly lexicographically:
      - train : 2018-01-01 00:00:00 through 2023-12-31 23:00:00
      - val   : 2024-01-01 00:00:00 through 2025-12-31 23:00:00
      - test  : 2026-01-01 00:00:00 onward
    """
    if ts_str <= TRAIN_END:
        return "train"
    if ts_str < TEST_START:
        return "val"
    return "test"


def _find_missing_dates(timestamps: list[str]) -> list[str]:
    """Scan every calendar date from start to end and return any completely absent dates.

    Detecting missing calendar days (e.g., 2026-10-30 in the API segment) is critical
    so downstream lag generation never silently treats `d+2` as consecutive day `d+1`.
    """
    seen = {ts[:10] for ts in timestamps}
    start_d = date.fromisoformat(min(seen))
    end_d = date.fromisoformat(max(seen))
    missing: list[str] = []
    cur = start_d
    while cur <= end_d:
        iso = cur.isoformat()
        if iso not in seen:
            missing.append(iso)
        cur += timedelta(days=1)
    return missing


def audit_and_deduplicate(raw_csv_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit raw multi-zone ERCOT CSV and deduplicate to a unique hourly system series.

    Each timestamp in `ercot_model_table.csv` appears 8 times (once per weather zone).
    This function:
      - Streams all rows once and verifies schema & expected columns.
      - Groups rows by `(timestamp, DSTFlag)` so the repeated `01:00:00` hour during
        each November Daylight Saving Time fall-back is preserved as two distinct hours
        rather than overwritten.
      - Verifies that `system_total_mw`, `source`, and `uri_flag` match identically across
        all 8 zones at each hour.
      - Averages zone weather readings per hour and imputes any remaining NaNs using
        medians computed strictly on the `train` split (2018-2023).

    Args:
        raw_csv_path: Path to `ercot_model_table.csv`.

    Returns:
        `(system_data, audit_summary)`:
          - `system_data`: dict of 77,496-element chronological arrays/lists.
          - `audit_summary`: JSON-serializable audit dictionary for reproducibility.
    """
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv_path}")

    # Running audit accumulators updated row-by-row during the single file pass
    total_rows = 0
    zone_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing_raw_by_col: Counter[str] = Counter()
    missing_raw_by_split: Counter[str] = Counter()
    uri_zone_rows = 0

    # Keyed by `(timestamp, DSTFlag)` because November fall-back repeats 01:00:00
    # once with DSTFlag=False and once with DSTFlag=True.
    hourly_records: dict[tuple[str, bool], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, bool]] = []
    max_sys_spread = 0.0
    source_mismatch_count = 0
    uri_mismatch_count = 0

    with raw_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = tuple(reader.fieldnames or ())
        if columns != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected CSV columns: {columns} != {EXPECTED_COLUMNS}")

        for row in reader:
            total_rows += 1
            ts = row["timestamp"].strip()
            dst_flag = _parse_bool(row["DSTFlag"])
            source = row["source"].strip()
            uri_flag = _parse_bool(row["uri_flag"])
            zone = row["zone"].strip()
            sys_mw = float(row["system_total_mw"])
            load_mw = float(row["load_mw"])

            # Tally per-zone, per-source, and Winter Storm Uri row counts
            zone_counts[zone] += 1
            source_counts[source] += 1
            if uri_flag:
                uri_zone_rows += 1

            # Track raw weather missingness both globally and by chronological split
            split_lbl = assign_split(ts)
            weather_vals: dict[str, float] = {}
            for wcol in WEATHER_COLUMNS:
                val = _parse_float(row[wcol])
                weather_vals[wcol] = val
                if np.isnan(val):
                    missing_raw_by_col[wcol] += 1
                    missing_raw_by_split[f"{wcol}:{split_lbl}"] += 1

            # Create or update the deduplicated hourly bucket for this (timestamp, DSTFlag)
            key = (ts, dst_flag)
            rec = hourly_records.get(key)
            if rec is None:
                rec = {
                    "timestamp": ts,
                    "dst_flag": dst_flag,
                    "source": source,
                    "uri_flag": uri_flag,
                    "system_total_mw": sys_mw,
                    "zone_loads": {},
                    "weather_lists": {wcol: [] for wcol in WEATHER_COLUMNS},
                }
                hourly_records[key] = rec
                ordered_keys.append(key)
            else:
                # Sanity check: ensure system-wide values really are identical across all 8 zones
                spread = abs(rec["system_total_mw"] - sys_mw)
                if spread > max_sys_spread:
                    max_sys_spread = spread
                if rec["source"] != source:
                    source_mismatch_count += 1
                if rec["uri_flag"] != uri_flag:
                    uri_mismatch_count += 1

            # Store each individual zone's load and accumulate non-NaN weather values
            rec["zone_loads"][zone] = load_mw
            for wcol, val in weather_vals.items():
                if not np.isnan(val):
                    rec["weather_lists"][wcol].append(val)

    # Sort chronologically by timestamp string, placing DSTFlag=False before DSTFlag=True
    # on the 9 November fall-back nights so time flows strictly forward.
    ordered_keys.sort(key=lambda k: (k[0], k[1]))
    n_hours = len(ordered_keys)

    # Pre-allocate dense NumPy arrays for fast downstream slicing and plotting
    timestamps: list[str] = []
    dst_flags = np.zeros(n_hours, dtype=bool)
    sources: list[str] = []
    uri_flags = np.zeros(n_hours, dtype=bool)
    splits: list[str] = []
    system_total_mw = np.zeros(n_hours, dtype=np.float64)
    zone_load_sum_mw = np.zeros(n_hours, dtype=np.float64)
    zone_loads_arr = {z: np.zeros(n_hours, dtype=np.float64) for z in EXPECTED_ZONES}
    weather_mean_arr = {w: np.full(n_hours, np.nan, dtype=np.float64) for w in WEATHER_COLUMNS}

    # Count occurrences of each timestamp string (without DSTFlag) to detect fall-back pairs
    ts_only_counter: Counter[str] = Counter()
    for idx, key in enumerate(ordered_keys):
        rec = hourly_records[key]
        ts = rec["timestamp"]
        ts_only_counter[ts] += 1
        timestamps.append(ts)
        dst_flags[idx] = rec["dst_flag"]
        sources.append(rec["source"])
        uri_flags[idx] = rec["uri_flag"]
        splits.append(assign_split(ts))
        system_total_mw[idx] = rec["system_total_mw"]

        # Record each zone's hourly MW and compute their exact cross-zone sum
        zsum = 0.0
        for z in EXPECTED_ZONES:
            val = rec["zone_loads"][z]
            zone_loads_arr[z][idx] = val
            zsum += val
        zone_load_sum_mw[idx] = zsum

        # Compute system-wide cross-zone average weather for each valid variable
        for wcol in WEATHER_COLUMNS:
            vals = rec["weather_lists"][wcol]
            if vals:
                weather_mean_arr[wcol][idx] = float(np.mean(vals))

    # Leakage Control #1: Fit weather median imputation strictly on `train` hours (2018-2023)
    # so validation/test weather distributions never leak into training imputation rules.
    train_mask = np.array([s == "train" for s in splits], dtype=bool)
    weather_impute_medians: dict[str, float] = {}
    for wcol in WEATHER_COLUMNS:
        arr = weather_mean_arr[wcol]
        train_valid = arr[train_mask & ~np.isnan(arr)]
        med = float(np.median(train_valid)) if train_valid.size > 0 else 0.0
        weather_impute_medians[wcol] = med
        nan_mask = np.isnan(arr)
        if np.any(nan_mask):
            arr[nan_mask] = med

    # Identify all 9 DST fall-back repeated timestamps and any missing calendar dates
    dst_repeated_timestamps = sorted(ts for ts, cnt in ts_only_counter.items() if cnt > 1)
    missing_dates = _find_missing_dates(timestamps)

    # Summarize hourly load distributions for Train, Validation, and Test splits
    split_hour_counts = dict(Counter(splits))
    split_load_stats: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        mask = np.array([s == split_name for s in splits], dtype=bool)
        vals = system_total_mw[mask]
        split_load_stats[split_name] = {
            "hours": int(np.sum(mask)),
            "mean_mw": round(float(np.mean(vals)), 2),
            "std_mw": round(float(np.std(vals)), 2),
            "min_mw": round(float(np.min(vals)), 2),
            "median_mw": round(float(np.median(vals)), 2),
            "max_mw": round(float(np.max(vals)), 2),
        }

    uri_vals = system_total_mw[uri_flags]
    uri_timestamps = [ts for ts, u in zip(timestamps, uri_flags) if u]

    # Package full machine-readable audit report
    audit_summary: dict[str, Any] = {
        "raw_file": raw_csv_path.name,
        "total_raw_rows": total_rows,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "columns": list(columns),
        "unique_timestamp_dst_pairs": n_hours,
        "expected_unique_hours": EXPECTED_TIMESTAMPS,
        "unique_timestamps_without_dst": len(ts_only_counter),
        "dst_repeated_hours_count": len(dst_repeated_timestamps),
        "dst_repeated_timestamps": dst_repeated_timestamps,
        "missing_calendar_dates": missing_dates,
        "zones": dict(sorted(zone_counts.items())),
        "sources_raw_rows": dict(sorted(source_counts.items())),
        "archive_date_range": [
            min(ts for ts, s in zip(timestamps, sources) if s == "archive"),
            max(ts for ts, s in zip(timestamps, sources) if s == "archive"),
        ],
        "api_date_range": [
            min(ts for ts, s in zip(timestamps, sources) if s == "api"),
            max(ts for ts, s in zip(timestamps, sources) if s == "api"),
        ],
        "max_cross_zone_system_mw_spread": float(max_sys_spread),
        "cross_zone_source_mismatches": source_mismatch_count,
        "cross_zone_uri_mismatches": uri_mismatch_count,
        "raw_missing_weather_counts": dict(sorted(missing_raw_by_col.items())),
        "raw_missing_weather_by_split": dict(sorted(missing_raw_by_split.items())),
        "train_weather_impute_medians": {k: round(v, 4) for k, v in weather_impute_medians.items()},
        "uri_flag_zone_rows": uri_zone_rows,
        "uri_flag_system_hours": int(np.sum(uri_flags)),
        "uri_date_range": [min(uri_timestamps), max(uri_timestamps)] if uri_timestamps else [],
        "uri_peak_mw": round(float(np.max(uri_vals)), 2) if uri_vals.size else 0.0,
        "split_boundaries": {
            "train": ["2018-01-01 00:00:00", TRAIN_END],
            "val": [VAL_START, "2025-12-31 23:00:00"],
            "test": [TEST_START, timestamps[-1]],
        },
        "split_hour_counts": split_hour_counts,
        "split_load_stats_mw": split_load_stats,
        "overall_load_stats_mw": {
            "mean_mw": round(float(np.mean(system_total_mw)), 2),
            "std_mw": round(float(np.std(system_total_mw)), 2),
            "min_mw": round(float(np.min(system_total_mw)), 2),
            "median_mw": round(float(np.median(system_total_mw)), 2),
            "max_mw": round(float(np.max(system_total_mw)), 2),
        },
    }

    system_data: dict[str, Any] = {
        "timestamp": timestamps,
        "dst_flag": dst_flags,
        "source": sources,
        "uri_flag": uri_flags,
        "split": splits,
        "system_total_mw": system_total_mw,
        "zone_load_sum_mw": zone_load_sum_mw,
        "zone_loads": zone_loads_arr,
        "weather_mean": weather_mean_arr,
    }
    return system_data, audit_summary


def _build_daily_profiles(system_data: dict[str, Any]) -> dict[date, dict[str, Any]]:
    """Aggregate the hourly system array into fixed 24-hour clock vectors [00..23] per day.

    Handling Daylight Saving Time (DST) transitions cleanly:
      - Normal days (3,211 days) have exactly 1 observation for each hour 0..23.
      - Fall-back DST days (9 days, 25 hours) have two 01:00 rows; averaging them aligns
        the fixed-width 24-hour matrix while flagging `is_dst_transition=True`.
      - Spring-forward DST days (9 days, 23 hours) skip 02:00; linear neighbor interpolation
        fills hour 02 ONLY inside the fixed-width 24-hour matrix (the exported hourly CSV
        preserves all 77,496 exact raw timestamps untouched).
    """
    # Group hourly array indices by ISO calendar date ("YYYY-MM-DD")
    grouped: dict[date, list[int]] = defaultdict(list)
    for idx, ts in enumerate(system_data["timestamp"]):
        grouped[date.fromisoformat(ts[:10])].append(idx)

    daily: dict[date, dict[str, Any]] = {}
    sys_mw = system_data["system_total_mw"]
    uri_flags = system_data["uri_flag"]
    dst_flags = system_data["dst_flag"]
    weather = system_data["weather_mean"]

    for d, idxs in grouped.items():
        # Bucket indices by clock hour 0..23 within this calendar day
        hour_buckets: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            hr = int(system_data["timestamp"][i][11:13])
            hour_buckets[hr].append(i)

        load_24 = np.full(24, np.nan, dtype=np.float64)
        for hr, b_idxs in hour_buckets.items():
            # For normal hours len(b_idxs)==1; on fall-back 01:00 len(b_idxs)==2
            load_24[hr] = float(np.mean(sys_mw[b_idxs]))

        # Fill skipped 02:00 slot on spring-forward 23h days via adjacent-hour average
        is_dst_day = len(idxs) != 24 or any(bool(dst_flags[i]) for i in idxs)
        if np.isnan(load_24[2]) and not np.isnan(load_24[1]) and not np.isnan(load_24[3]):
            load_24[2] = 0.5 * (load_24[1] + load_24[3])
        if np.any(np.isnan(load_24)):
            valid_hrs = np.where(~np.isnan(load_24))[0]
            load_24 = np.interp(np.arange(24), valid_hrs, load_24[valid_hrs])

        # Precompute target-day weather summaries separately for optional upper-bound experiments
        idx_arr = np.array(idxs, dtype=np.int64)
        temp_f_day = weather["temp_f"][idx_arr]
        dwpt_day = weather["dwpt"][idx_arr]
        rhum_day = weather["rhum"][idx_arr]
        wspd_day = weather["wspd"][idx_arr]

        daily[d] = {
            "load_24": load_24,
            "raw_hour_count": len(idxs),
            "is_dst_transition": is_dst_day,
            "is_uri": bool(np.any(uri_flags[idx_arr])),
            "split": system_data["split"][idxs[0]],
            "weather_summary": np.array(
                [
                    np.mean(temp_f_day),
                    np.min(temp_f_day),
                    np.max(temp_f_day),
                    np.mean(dwpt_day),
                    np.min(dwpt_day),
                    np.max(dwpt_day),
                    np.mean(rhum_day),
                    np.min(rhum_day),
                    np.max(rhum_day),
                    np.mean(wspd_day),
                    np.min(wspd_day),
                    np.max(wspd_day),
                ],
                dtype=np.float64,
            ),
        }
    return daily


def _calendar_features(target_day: date, is_dst_transition: bool) -> np.ndarray:
    """Build 11 deterministic calendar features known ahead of time for `target_day`.

    Features:
      - `[0..1]`: Cyclical month encoding (sin, cos) so Dec (12) wraps smoothly to Jan (1).
      - `[2..8]`: 7-dim one-hot day-of-week (Monday=0 .. Sunday=6).
      - `[9]`   : Binary weekend flag (1.0 for Saturday/Sunday, else 0.0).
      - `[10]`  : Binary DST transition day flag (known from calendar schedule).
    """
    month_angle = 2.0 * np.pi * (target_day.month - 1) / 12.0
    dow = target_day.weekday()  # 0=Mon .. 6=Sun
    dow_onehot = np.zeros(7, dtype=np.float64)
    dow_onehot[dow] = 1.0
    is_weekend = 1.0 if dow >= 5 else 0.0
    dst_num = 1.0 if is_dst_transition else 0.0
    return np.concatenate(
        [
            np.array([np.sin(month_angle), np.cos(month_angle)], dtype=np.float64),
            dow_onehot,
            np.array([is_weekend, dst_num], dtype=np.float64),
        ]
    )


def get_day_ahead_feature_names() -> list[str]:
    """Return ordered human-readable column names for all 67 causal day-ahead features."""
    names = [f"lag_1d_h{h:02d}" for h in range(24)]
    names += [f"lag_7d_h{h:02d}" for h in range(24)]
    names += ["past_2to6d_mean", "past_2to6d_std", "past_2to6d_min", "past_2to6d_max"]
    names += ["roll_7d_mean", "roll_7d_std", "roll_7d_min", "roll_7d_max"]
    names += ["target_month_sin", "target_month_cos"]
    names += [f"target_dow_{i}" for i in range(7)]
    names += ["target_is_weekend", "target_is_dst_transition"]
    return names


def split_and_prepare_features(system_data: dict[str, Any]) -> dict[str, Any]:
    """Build leakage-safe day-ahead 24h supervised matrices `(X_raw, X_scaled, Y)`.

    Forecast Issuance Protocol (Proposal Section 4.1):
      Standing at the end of origin day `d` (23:00), predict the 24 hourly loads of
      tomorrow `target_d = d + 1`. Only information observed on or before day `d` plus
      deterministic calendar features of `d + 1` enters `X_raw`:
        1. `lag_1d` (24 features): Today's (`origin_d`) 24 hourly loads.
        2. `lag_7d` (24 features): Same weekday one week ago (`target_d - 7d == origin_d - 6d`).
        3. `past_2to6d_summary` (4 features): Mean, std, min, max across available days in `[d-5..d-1]`.
        4. `roll_7d_summary` (4 features): Mean, std, min, max across available days in `[d-6..d]`.
        5. `calendar` (11 features): Cyclical month, day-of-week one-hot, weekend, DST transition.

    Leakage Controls:
      - Missing calendar date `2026-10-30` is handled by checking exact `date` dictionary keys
        instead of blind array indexing, preventing shifted lag alignment.
      - Target-day observed weather is NEVER included in `X_raw`/`X_scaled`; it is isolated in
        `upper_bound_weather_only` for optional non-causal upper-bound ablation checks only.
      - Feature standardization (`scaler_mean`, `scaler_std`) is fit strictly on `train` days.
    """
    daily = _build_daily_profiles(system_data)
    sorted_dates = sorted(daily.keys())

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    ub_weather_rows: list[np.ndarray] = []
    origin_dates: list[str] = []
    target_dates: list[str] = []
    sample_splits: list[str] = []
    uri_masks: list[bool] = []
    dst_masks: list[bool] = []

    for origin_d in sorted_dates:
        target_d = origin_d + timedelta(days=1)
        # Same weekday 7 days before target_d corresponds to (origin_d - 6 days)
        lag7_d = target_d - timedelta(days=7)

        # Skip boundaries where tomorrow (`target_d`) or the 7-day baseline (`lag7_d`) is absent
        # (e.g., first 6 days of Jan 2018, or adjacent to missing date 2026-10-30).
        if target_d not in daily or lag7_d not in daily:
            continue

        # Collect days [d-5 .. d-1] (robust to a single missing intermediate day)
        past_2to6_loads = [
            daily[origin_d - timedelta(days=k)]["load_24"]
            for k in range(1, 6)
            if (origin_d - timedelta(days=k)) in daily
        ]
        if not past_2to6_loads:
            continue

        # Collect the full trailing 7-day window [d-6 .. d] available at forecast issuance
        roll_7d_loads = [
            daily[origin_d - timedelta(days=k)]["load_24"]
            for k in range(0, 7)
            if (origin_d - timedelta(days=k)) in daily
        ]
        past_2to6_arr = np.concatenate(past_2to6_loads)
        roll_7d_arr = np.concatenate(roll_7d_loads)

        lag_1d_vec = daily[origin_d]["load_24"]
        lag_7d_vec = daily[lag7_d]["load_24"]
        summary_2to6 = np.array(
            [
                float(np.mean(past_2to6_arr)),
                float(np.std(past_2to6_arr)),
                float(np.min(past_2to6_arr)),
                float(np.max(past_2to6_arr)),
            ],
            dtype=np.float64,
        )
        summary_7d = np.array(
            [
                float(np.mean(roll_7d_arr)),
                float(np.std(roll_7d_arr)),
                float(np.min(roll_7d_arr)),
                float(np.max(roll_7d_arr)),
            ],
            dtype=np.float64,
        )
        cal_vec = _calendar_features(target_d, daily[target_d]["is_dst_transition"])

        # Concatenate all 67 causal predictors into one row vector for origin_d -> target_d
        feat_vec = np.concatenate([lag_1d_vec, lag_7d_vec, summary_2to6, summary_7d, cal_vec])
        x_rows.append(feat_vec)
        y_rows.append(daily[target_d]["load_24"])
        ub_weather_rows.append(daily[target_d]["weather_summary"])
        origin_dates.append(origin_d.isoformat())
        target_dates.append(target_d.isoformat())
        # Assign sample split according to the target day being predicted
        sample_splits.append(daily[target_d]["split"])
        uri_masks.append(daily[target_d]["is_uri"] or daily[origin_d]["is_uri"])
        dst_masks.append(daily[target_d]["is_dst_transition"])

    X_raw = np.vstack(x_rows)
    Y = np.vstack(y_rows)
    upper_bound_weather_only = np.vstack(ub_weather_rows)
    splits_arr = np.array(sample_splits)

    # Leakage Control #2: Compute mean & std strictly from `train` rows (2018-2023),
    # guarding near-zero-variance features with a 1.0 denominator before scaling all splits.
    train_idx = splits_arr == "train"
    scaler_mean = np.mean(X_raw[train_idx], axis=0)
    scaler_std = np.std(X_raw[train_idx], axis=0)
    scaler_std = np.where(scaler_std < 1e-8, 1.0, scaler_std)
    X_scaled = (X_raw - scaler_mean) / scaler_std

    return {
        "feature_names": np.array(get_day_ahead_feature_names()),
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "Y": Y,
        "upper_bound_weather_only": upper_bound_weather_only,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "origin_dates": np.array(origin_dates),
        "target_dates": np.array(target_dates),
        "splits": splits_arr,
        "uri_mask": np.array(uri_masks, dtype=bool),
        "dst_mask": np.array(dst_masks, dtype=bool),
    }


def save_system_csv(system_data: dict[str, Any], out_csv_path: Path) -> None:
    """Export the deduplicated 77,496-row hourly system series and zone loads to CSV."""
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    zone_cols = [f"load_{z}_mw" for z in EXPECTED_ZONES]
    weather_cols = [f"{w}_mean" for w in WEATHER_COLUMNS]
    header = [
        "timestamp",
        "DSTFlag",
        "source",
        "uri_flag",
        "split",
        "system_total_mw",
        "zone_load_sum_mw",
        *weather_cols,
        *zone_cols,
    ]

    n_rows = len(system_data["timestamp"])
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_rows):
            row = [
                system_data["timestamp"][i],
                bool(system_data["dst_flag"][i]),
                system_data["source"][i],
                bool(system_data["uri_flag"][i]),
                system_data["split"][i],
                f"{system_data['system_total_mw'][i]:.6f}",
                f"{system_data['zone_load_sum_mw'][i]:.6f}",
                *[f"{system_data['weather_mean'][w][i]:.4f}" for w in WEATHER_COLUMNS],
                *[f"{system_data['zone_loads'][z][i]:.6f}" for z in EXPECTED_ZONES],
            ]
            writer.writerow(row)


def build_feature_dict_from_raw(raw_csv_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run audit, deduplication, and feature generation end-to-end from the raw CSV path.

    Returns:
        `(system_data, audit_summary, dataset_bundle)` with day-ahead split counts attached
        to `audit_summary`.
    """
    system_data, audit_summary = audit_and_deduplicate(raw_csv_path)
    dataset_bundle = split_and_prepare_features(system_data)
    splits = dataset_bundle["splits"]
    audit_summary["day_ahead_sample_counts"] = {
        "total_days": int(len(splits)),
        "train_days": int(np.sum(splits == "train")),
        "val_days": int(np.sum(splits == "val")),
        "test_days": int(np.sum(splits == "test")),
        "n_features": int(dataset_bundle["X_raw"].shape[1]),
        "horizon_hours": int(dataset_bundle["Y"].shape[1]),
    }
    return system_data, audit_summary, dataset_bundle
