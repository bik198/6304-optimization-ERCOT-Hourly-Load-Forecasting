"""Step 0 entry-point script: audit raw data, export clean datasets, and generate EDA figures.

Usage:
    python step0_base_audit.py

Outputs generated:
    - `data/ercot_system_hourly.csv`      : Deduplicated 77,496-row hourly system table
    - `data/ercot_step0_features.npz`     : Cached NumPy arrays for Steps 1-5 modeling
    - `outputs/step0_audit_summary.json`  : Verified dataset statistics & split counts
    - `figures/fig1..fig4.png`            : Four Dark Carbon EDA visualizations (Proposal Sec. 3.3)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
# Use non-interactive Agg backend so plotting works headlessly without opening GUI windows
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    AUDIT_JSON_PATH,
    DARK_CARBON_PALETTE as PAL,
    FEATURES_NPZ_PATH,
    FIGURES_DIR,
    OUTPUTS_DIR,
    RAW_CSV_PATH,
    SYSTEM_CSV_PATH,
)
from src.data_pipeline import build_feature_dict_from_raw, save_system_csv


def _create_dark_figure(figsize: tuple[float, float], ncols: int = 1) -> tuple[plt.Figure, Any]:
    """Create a 1- or 2-panel Matplotlib figure pre-styled with the Dark Carbon palette."""
    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=160)
    fig.patch.set_facecolor(PAL["bg_darkest"])
    ax_list = list(axes) if ncols > 1 else [axes]
    for ax in ax_list:
        ax.set_facecolor(PAL["bg_primary"])
        ax.grid(True, color=PAL["bg_card"], linestyle="-", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(PAL["border_input"])
        ax.tick_params(colors=PAL["text_main"], labelsize=9)
        ax.xaxis.label.set_color(PAL["text_bright"])
        ax.yaxis.label.set_color(PAL["text_bright"])
        ax.title.set_color(PAL["text_blue"])
    return fig, axes


def _style_axes_and_legend(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_loc: str | None = None,
    legend_ncol: int = 1,
    legend_fontsize: float = 8.5,
    title_fontsize: float = 12,
    label_fontsize: float = 10,
) -> None:
    """Apply consistent title, axis labels, and optional Dark Carbon legend styling."""
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10 if title_fontsize >= 12 else 8)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")
    if legend_loc is not None:
        ax.legend(
            loc=legend_loc,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            facecolor=PAL["bg_card"],
            edgecolor=PAL["border_input"],
            labelcolor=PAL["text_main"],
        )


def _save_and_close(fig: plt.Figure, out_path: Path) -> None:
    """Tightly fit layout, write PNG preserving Dark Carbon background, and free memory."""
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_eda_figures(system_data: dict[str, Any], figures_dir: Path) -> None:
    """Parse timestamps once and render all four Proposal Section 3.3 EDA figures.

    Figures produced in `figures_dir`:
      1. `fig1_weekly_avg_load.png`       : Weekly demand trend (2018-2026) + split shading + Uri
      2. `fig2_hourly_load_profile.png`   : Diurnal mean load & IQR across hours 00-23
      3. `fig3_monthly_load_profile.png`  : Monthly demand seasonality (Jan-Dec)
      4. `fig4_load_distribution_and_uri.png`: Demand histogram + Feb 2021 Winter Storm Uri zoom
    """
    # Parse all 77,496 timestamp strings ONCE and reuse across all figures
    parsed_dts = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in system_data["timestamp"]]
    hours = np.array([dt.hour for dt in parsed_dts], dtype=np.int32)
    months = np.array([dt.month for dt in parsed_dts], dtype=np.int32)
    loads = system_data["system_total_mw"]
    uri_flags = system_data["uri_flag"]

    # -----------------------------------------------------------------------
    # Figure 1: Weekly Average ERCOT System Load Over Time (2018-2026)
    # -----------------------------------------------------------------------
    week_buckets: dict[tuple[int, int], list[tuple[datetime, float]]] = {}
    for dt, val in zip(parsed_dts, loads):
        iso = dt.isocalendar()
        week_buckets.setdefault((iso.year, iso.week), []).append((dt, float(val)))

    week_dates = [pts[0][0] for pts in week_buckets.values()]
    week_means = np.array([np.mean([p[1] for p in pts]) for pts in week_buckets.values()])

    fig, ax = _create_dark_figure((11, 4.4))
    val_dt, test_dt = datetime(2024, 1, 1), datetime(2026, 1, 1)
    uri_start, uri_end = datetime(2021, 2, 14), datetime(2021, 2, 20, 23)

    ax.axvspan(week_dates[0], val_dt, color="#252526", alpha=0.45, label="Train (2018-2023)")
    ax.axvspan(val_dt, test_dt, color="#2B2B2B", alpha=0.55, label="Validation (2024-2025)")
    ax.axvspan(test_dt, week_dates[-1], color="#323234", alpha=0.65, label="Test (2026)")
    ax.axvspan(uri_start, uri_end, color=PAL["text_red"], alpha=0.35, label="Winter Storm Uri (Feb 2021)")
    ax.plot(week_dates, week_means, color=PAL["text_link"], linewidth=1.6, label="Weekly Avg System Load")
    ax.axvline(val_dt, color=PAL["text_gold"], linestyle="--", linewidth=1.1)
    ax.axvline(test_dt, color=PAL["text_yellow"], linestyle="--", linewidth=1.1)

    uri_vals = loads[uri_flags]
    if uri_vals.size > 0:
        feb21_mask = (np.array(week_dates) >= datetime(2021, 2, 1)) & (np.array(week_dates) <= datetime(2021, 3, 1))
        ax.annotate(
            f"Uri Peak: {np.max(uri_vals):,.0f} MW",
            xy=(datetime(2021, 2, 15), float(np.max(week_means[feb21_mask]))),
            xytext=(datetime(2019, 9, 1), 69500),
            color=PAL["text_red"],
            fontsize=8.5,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": PAL["text_red"], "lw": 1.1},
        )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axes_and_legend(
        ax,
        title="ERCOT System Load: Weekly Average (2018-2026)",
        xlabel="Date",
        ylabel="System Load (MW)",
        legend_loc="upper left",
        legend_ncol=3,
        legend_fontsize=8,
    )
    _save_and_close(fig, figures_dir / "fig1_weekly_avg_load.png")

    # -----------------------------------------------------------------------
    # Figure 2: Average Diurnal Load Profile by Hour of Day (00-23)
    # -----------------------------------------------------------------------
    hr_axis = np.arange(24)
    hr_means = np.array([np.mean(loads[hours == h]) for h in hr_axis])
    hr_q25 = np.array([np.percentile(loads[hours == h], 25) for h in hr_axis])
    hr_q75 = np.array([np.percentile(loads[hours == h], 75) for h in hr_axis])

    fig, ax = _create_dark_figure((9.5, 4.2))
    ax.fill_between(hr_axis, hr_q25, hr_q75, color=PAL["text_blue"], alpha=0.22, label="Interquartile Range (25th-75th pct)")
    ax.plot(hr_axis, hr_means, color=PAL["text_yellow"], marker="o", markersize=5, linewidth=2.0, label="Mean System Load")

    peak_hr, trough_hr = int(np.argmax(hr_means)), int(np.argmin(hr_means))
    ax.scatter([peak_hr], [hr_means[peak_hr]], color=PAL["text_red"], s=65, zorder=5)
    ax.annotate(
        f"Peak Avg: {hr_means[peak_hr]:,.0f} MW (Hour {peak_hr:02d}:00)",
        xy=(peak_hr, hr_means[peak_hr]),
        xytext=(peak_hr - 6.5, hr_means[peak_hr] + 3200),
        color=PAL["text_gold"],
        fontsize=8.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": PAL["text_gold"], "lw": 1.0},
    )
    ax.annotate(
        f"Trough Avg: {hr_means[trough_hr]:,.0f} MW (Hour {trough_hr:02d}:00)",
        xy=(trough_hr, hr_means[trough_hr]),
        xytext=(trough_hr + 2.0, hr_means[trough_hr] - 4200),
        color=PAL["text_sky"],
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": PAL["text_sky"], "lw": 1.0},
    )

    ax.set_xticks(hr_axis)
    _style_axes_and_legend(
        ax,
        title="Average ERCOT System Load by Hour of Day",
        xlabel="Hour of Day (00-23)",
        ylabel="System Load (MW)",
        legend_loc="upper left",
    )
    _save_and_close(fig, figures_dir / "fig2_hourly_load_profile.png")

    # -----------------------------------------------------------------------
    # Figure 3: Monthly Demand Seasonality Bar Chart (Jan-Dec)
    # -----------------------------------------------------------------------
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m_means = np.array([np.mean(loads[months == m]) for m in range(1, 13)])
    bar_colors = [
        PAL["text_gold"] if m in (7, 8) else (PAL["text_blue"] if m in (6, 9) else PAL["text_link"])
        for m in range(1, 13)
    ]

    fig, ax = _create_dark_figure((9.5, 4.2))
    bars = ax.bar(month_names, m_means, color=bar_colors, edgecolor=PAL["border_input"], linewidth=0.9, width=0.68)
    for bar, val in zip(bars, m_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 700,
            f"{val / 1000.0:.1f}k",
            ha="center",
            va="bottom",
            color=PAL["text_main"],
            fontsize=8,
        )

    ax.set_ylim(0, float(np.max(m_means)) * 1.14)
    _style_axes_and_legend(
        ax,
        title="Average ERCOT System Load by Month (2018-2026)",
        xlabel="Month",
        ylabel="Average System Load (MW)",
    )
    _save_and_close(fig, figures_dir / "fig3_monthly_load_profile.png")

    # -----------------------------------------------------------------------
    # Figure 4: Hourly Load Distribution & Winter Storm Uri Zoom (Feb 2021)
    # -----------------------------------------------------------------------
    fig, (ax_left, ax_right) = _create_dark_figure((11.5, 4.3), ncols=2)

    bins = np.linspace(float(np.min(loads)), float(np.max(loads)), 55)
    p99, max_load = float(np.percentile(loads, 99)), float(np.max(loads))
    ax_left.hist(loads, bins=bins, color=PAL["text_blue"], alpha=0.72, edgecolor=PAL["bg_darkest"], label="All Hours (77,496)")
    ax_left.hist(loads[uri_flags], bins=bins, color=PAL["text_red"], alpha=0.85, edgecolor=PAL["bg_darkest"], label="Winter Storm Uri (168h)")
    ax_left.axvline(p99, color=PAL["text_yellow"], linestyle="--", linewidth=1.2, label=f"99th Pct: {p99:,.0f} MW")
    ax_left.axvline(max_load, color=PAL["text_red"], linestyle=":", linewidth=1.3, label=f"Max Peak: {max_load:,.0f} MW")
    _style_axes_and_legend(
        ax_left,
        title="Hourly System Load Distribution",
        xlabel="System Load (MW)",
        ylabel="Hour Count",
        legend_loc="upper right",
        legend_fontsize=7.8,
        title_fontsize=11,
        label_fontsize=9.5,
    )

    win_mask = np.array([(datetime(2021, 2, 11) <= dt <= datetime(2021, 2, 24, 23)) for dt in parsed_dts], dtype=bool)
    win_dt = [dt for dt, m in zip(parsed_dts, win_mask) if m]
    win_load, win_uri = loads[win_mask], uri_flags[win_mask]
    ax_right.plot(win_dt, win_load, color=PAL["text_sky"], linewidth=1.3, label="System Load (Feb 11-24)")
    ax_right.plot([dt for dt, u in zip(win_dt, win_uri) if u], win_load[win_uri], color=PAL["text_red"], linewidth=1.9, label="Flagged Uri Window (Feb 14-20)")
    ax_right.axvspan(uri_start, uri_end, color=PAL["text_red"], alpha=0.14)
    ax_right.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax_right.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _style_axes_and_legend(
        ax_right,
        title="Winter Storm Uri Event Window (Feb 2021)",
        xlabel="Date",
        ylabel="System Load (MW)",
        legend_loc="lower right",
        legend_fontsize=7.8,
        title_fontsize=11,
        label_fontsize=9.5,
    )
    _save_and_close(fig, figures_dir / "fig4_load_distribution_and_uri.png")


def run_step0() -> dict[str, Any]:
    """Execute the full Step 0 pipeline: audit, export datasets, plot EDA, and print summary."""
    # Ensure output directories exist before writing artifacts
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Read raw 619,968-row CSV, deduplicate to 77,496 hours, and build causal tensors
    system_data, audit_summary, dataset_bundle = build_feature_dict_from_raw(RAW_CSV_PATH)

    # Step 2: Export clean hourly CSV (tracked in git) and compressed NPZ cache (gitignored)
    save_system_csv(system_data, SYSTEM_CSV_PATH)
    np.savez_compressed(FEATURES_NPZ_PATH, **dataset_bundle)

    # Step 3: Persist human- & machine-readable audit report JSON
    with AUDIT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Step 4: Generate all four Dark Carbon Exploratory Data Analysis charts in one pass
    generate_eda_figures(system_data, FIGURES_DIR)

    # Step 5: Print concise verification report to terminal
    print("=" * 72)
    print("STEP 0 AUDIT & LEAKAGE-SAFE PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Raw Rows Audited        : {audit_summary['total_raw_rows']:,} (expected {audit_summary['expected_total_rows']:,})")
    print(f"Deduplicated Hours      : {audit_summary['unique_timestamp_dst_pairs']:,} (8 zones -> 1 system row/hr)")
    print(f"Max Cross-Zone Spread   : {audit_summary['max_cross_zone_system_mw_spread']:.6f} MW")
    print(f"DST Repeated 01:00 Hrs  : {audit_summary['dst_repeated_hours_count']} timestamps")
    print(f"Missing Calendar Dates  : {audit_summary['missing_calendar_dates']}")
    print(f"Raw Missing Weather     : {audit_summary['raw_missing_weather_counts']}")
    print(f"Winter Storm Uri Hours  : {audit_summary['uri_flag_system_hours']} hrs ({audit_summary['uri_date_range'][0]} to {audit_summary['uri_date_range'][1]})")
    print("-" * 72)
    print("Chronological Splits (Hours / Day-Ahead 24h Samples):")
    da = audit_summary["day_ahead_sample_counts"]
    sh = audit_summary["split_hour_counts"]
    print(f"  Train (2018-2023)     : {sh['train']:,} hrs | {da['train_days']:,} days")
    print(f"  Val   (2024-2025)     : {sh['val']:,} hrs | {da['val_days']:,} days")
    print(f"  Test  (2026-11-04)    : {sh['test']:,} hrs | {da['test_days']:,} days")
    print(f"  Feature Matrix Shape  : ({da['total_days']}, {da['n_features']}) -> Target ({da['total_days']}, {da['horizon_hours']})")
    print("-" * 72)
    print(f"Saved System CSV        : {SYSTEM_CSV_PATH}")
    print(f"Saved Feature Cache     : {FEATURES_NPZ_PATH}")
    print(f"Saved Audit JSON        : {AUDIT_JSON_PATH}")
    print(f"Saved 4 EDA Figures in  : {FIGURES_DIR}")
    print("=" * 72)
    return audit_summary


if __name__ == "__main__":
    run_step0()
