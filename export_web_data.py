"""Export the small JSON the web story (web/) reads.

The browser should never parse the 77k-row hourly CSV. This script reduces
data/ercot_system_hourly.csv (plus the Step 0 audit JSON) to the handful of
aggregates each EDA chapter draws, and writes them to web/public/data/eda.json.

Run after step0_base_audit.py:
    python export_web_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AUDIT_JSON_PATH, ROOT_DIR, SYSTEM_CSV_PATH

OUT_PATH: Path = ROOT_DIR / "web" / "public" / "data" / "eda.json"

ZONES = ["coast", "east", "farWest", "north", "northC", "southC", "southern", "west"]
ZONE_LABELS = {
    "coast": "Coast", "east": "East", "farWest": "Far West", "north": "North",
    "northC": "North Central", "southC": "South Central", "southern": "Southern", "west": "West",
}
SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def r(x: float, nd: int = 0) -> float | None:
    """Round for JSON; NaN becomes null."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    v = round(float(x), nd)
    return int(v) if nd == 0 else v


def load() -> pd.DataFrame:
    df = pd.read_csv(SYSTEM_CSV_PATH, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["season"] = df["month"].map(SEASON_OF_MONTH)
    return df


def daily_series(df: pd.DataFrame) -> list[dict]:
    g = df.groupby("date").agg(
        mean=("system_total_mw", "mean"), peak=("system_total_mw", "max"),
        temp=("temp_f_mean", "mean"), split=("split", "first"), uri=("uri_flag", "max"),
    )
    return [
        {"d": d.strftime("%Y-%m-%d"), "mean": r(row["mean"]), "peak": r(row["peak"]),
         "temp": r(row["temp"], 1), "split": row["split"], "uri": bool(row["uri"])}
        for d, row in g.iterrows()
    ]


def weekly_series(df: pd.DataFrame) -> list[dict]:
    w = df.set_index("timestamp")["system_total_mw"].resample("W-MON", label="left", closed="left").mean()
    return [{"d": d.strftime("%Y-%m-%d"), "mean": r(v)} for d, v in w.items() if not np.isnan(v)]


def yearly(df: pd.DataFrame) -> list[dict]:
    g = df.groupby("year").agg(mean=("system_total_mw", "mean"), peak=("system_total_mw", "max"),
                               hours=("system_total_mw", "size"))
    out = []
    for y, row in g.iterrows():
        peak_ts = df.loc[df["year"] == y, ["timestamp", "system_total_mw"]].nlargest(1, "system_total_mw")
        out.append({"year": int(y), "mean": r(row["mean"]), "peak": r(row["peak"]),
                    "peakAt": peak_ts["timestamp"].iloc[0].strftime("%Y-%m-%d %H:00"),
                    "partial": bool(row["hours"] < 8700)})
    return out


def hourly_profiles(df: pd.DataFrame) -> dict:
    overall = df.groupby("hour")["system_total_mw"].mean()
    seasons = {s: [r(v) for v in df[df["season"] == s].groupby("hour")["system_total_mw"].mean()]
               for s in ["winter", "spring", "summer", "fall"]}
    weekday = df[df["timestamp"].dt.day_of_week < 5].groupby("hour")["system_total_mw"].mean()
    weekend = df[df["timestamp"].dt.day_of_week >= 5].groupby("hour")["system_total_mw"].mean()
    return {"overall": [r(v) for v in overall], "seasons": seasons,
            "weekday": [r(v) for v in weekday], "weekend": [r(v) for v in weekend]}


def month_hour_heatmap(df: pd.DataFrame) -> list[list[int]]:
    p = df.pivot_table(index="month", columns="hour", values="system_total_mw", aggfunc="mean")
    return [[r(v) for v in p.loc[m]] for m in range(1, 13)]


def monthly(df: pd.DataFrame) -> list[dict]:
    g = df.groupby("month")["system_total_mw"].agg(["mean", "max"])
    return [{"month": int(m), "mean": r(row["mean"]), "peak": r(row["max"])} for m, row in g.iterrows()]


def temperature_curve(df: pd.DataFrame) -> list[dict]:
    """Daily mean load binned by daily mean temperature (2 °F bins) — the U-curve."""
    d = df.groupby("date").agg(load=("system_total_mw", "mean"), temp=("temp_f_mean", "mean")).dropna()
    d["bin"] = (np.floor(d["temp"] / 2) * 2).astype(int)
    g = d.groupby("bin")["load"].agg(["mean", "size"])
    return [{"t": int(b) + 1, "mean": r(row["mean"]), "days": int(row["size"])} for b, row in g.iterrows() if row["size"] >= 3]


def zones(df: pd.DataFrame) -> dict:
    full = df[df["year"] <= 2025]
    by_year = full.groupby("year")[[f"load_{z}_mw" for z in ZONES]].mean()
    years = [int(y) for y in by_year.index]
    share = {z: [r(v) for v in by_year[f"load_{z}_mw"]] for z in ZONES}
    first, last = by_year.iloc[0], by_year.iloc[-1]
    growth = [{"zone": z, "label": ZONE_LABELS[z],
               "mw2018": r(first[f"load_{z}_mw"]), "mw2025": r(last[f"load_{z}_mw"]),
               "growthPct": r((last[f"load_{z}_mw"] / first[f"load_{z}_mw"] - 1) * 100, 1)} for z in ZONES]
    return {"order": ZONES, "labels": ZONE_LABELS, "years": years, "meanByYear": share, "growth": growth}


def distribution(df: pd.DataFrame) -> dict:
    y = df["system_total_mw"].to_numpy()
    edges = np.arange(np.floor(y.min() / 2000) * 2000, y.max() + 2000, 2000)
    counts, _ = np.histogram(y, bins=edges)
    top = df.nlargest(5, "system_total_mw")
    return {
        "bins": [{"lo": r(edges[i]), "hi": r(edges[i + 1]), "hours": int(c)} for i, c in enumerate(counts)],
        "p50": r(np.percentile(y, 50)), "p99": r(np.percentile(y, 99)),
        "min": r(y.min()), "max": r(y.max()),
        "topHours": [{"at": t.strftime("%Y-%m-%d %H:00"), "mw": r(v)}
                     for t, v in zip(top["timestamp"], top["system_total_mw"])],
    }


def uri_window(df: pd.DataFrame) -> dict:
    w = df[(df["timestamp"] >= "2021-02-07") & (df["timestamp"] < "2021-02-25")]
    return {
        "start": "2021-02-14", "end": "2021-02-20",
        "hours": [{"t": t.strftime("%Y-%m-%d %H:00"), "mw": r(mw), "temp": r(tf, 1), "uri": bool(u)}
                  for t, mw, tf, u in zip(w["timestamp"], w["system_total_mw"], w["temp_f_mean"], w["uri_flag"])],
    }


def splits(df: pd.DataFrame, audit: dict) -> list[dict]:
    out = []
    for name in ["train", "val", "test"]:
        s = df[df["split"] == name]
        out.append({"split": name, "start": s["timestamp"].min().strftime("%Y-%m-%d"),
                    "end": s["timestamp"].max().strftime("%Y-%m-%d"), "hours": int(len(s)),
                    "mean": r(s["system_total_mw"].mean()), "peak": r(s["system_total_mw"].max()),
                    "min": r(s["system_total_mw"].min())})
    samples = audit.get("day_ahead_sample_counts") or {}
    for row in out:
        if f"{row['split']}_days" in samples:
            row["days"] = int(samples[f"{row['split']}_days"])
    return out


def main() -> None:
    df = load()
    audit = json.loads(Path(AUDIT_JSON_PATH).read_text()) if Path(AUDIT_JSON_PATH).exists() else {}
    payload = {
        "meta": {
            "rawRows": audit.get("total_raw_rows", 619_968),
            "hours": int(len(df)), "zones": len(ZONES),
            "start": df["timestamp"].min().strftime("%Y-%m-%d"),
            "end": df["timestamp"].max().strftime("%Y-%m-%d"),
            "dstRepeats": audit.get("dst_repeated_hours_count"),
            "missingDates": audit.get("missing_calendar_dates", []),
            "uriHours": audit.get("uri_flag_system_hours"),
            "uriPeak": audit.get("uri_peak_mw"),
            "maxZoneSpread": audit.get("max_cross_zone_system_mw_spread"),
            "samples": (audit.get("day_ahead_sample_counts") or {}).get("total_days"),
            "features": (audit.get("day_ahead_sample_counts") or {}).get("n_features"),
        },
        "daily": daily_series(df),
        "weekly": weekly_series(df),
        "yearly": yearly(df),
        "hourly": hourly_profiles(df),
        "heatmap": month_hour_heatmap(df),
        "monthly": monthly(df),
        "temperature": temperature_curve(df),
        "zones": zones(df),
        "distribution": distribution(df),
        "uri": uri_window(df),
        "splits": splits(df, audit),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT_PATH.relative_to(ROOT_DIR)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
