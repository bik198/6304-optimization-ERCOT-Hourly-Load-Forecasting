"""Central project configuration: file paths, data schema, chronological splits, battery QP defaults, and UI colors.

Keeping all paths, date boundaries, and physical assumptions in one place prevents
mismatch across scripts (Step 0 audit, forecasting models, CVXPY dispatch, and video rendering).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem Paths (resolved relative to repository root)
# ---------------------------------------------------------------------------
# Root folder of the project (two levels up from src/config.py)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Raw 61.9 MB multi-zone CSV (excluded from git via .gitignore)
RAW_CSV_PATH: Path = ROOT_DIR / "ercot_model_table.csv"

# Directory for derived datasets (clean system CSV + NumPy feature cache)
DATA_DIR: Path = ROOT_DIR / "data"

# Deduplicated 1-row-per-hour system CSV (77,496 rows, safe for git tracking)
SYSTEM_CSV_PATH: Path = DATA_DIR / "ercot_system_hourly.csv"

# Compressed NumPy archive holding leakage-safe train/val/test feature & target arrays
FEATURES_NPZ_PATH: Path = DATA_DIR / "ercot_step0_features.npz"

# Output directory for Exploratory Data Analysis (EDA) PNG charts
FIGURES_DIR: Path = ROOT_DIR / "figures"

# Output directory for machine-readable JSON summaries and model results
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

# Step 0 audit report JSON produced by step0_base_audit.py
AUDIT_JSON_PATH: Path = OUTPUTS_DIR / "step0_audit_summary.json"

# ---------------------------------------------------------------------------
# Expected Dataset Profile Constants (Proposal Section 3.1)
# Used as strict schema assertions when reading ercot_model_table.csv
# ---------------------------------------------------------------------------
# 77,496 unique hourly timestamps x 8 ERCOT weather zones = 619,968 total rows
EXPECTED_TOTAL_ROWS: int = 619_968

# Exact expected column names in order inside the raw CSV
EXPECTED_COLUMNS: tuple[str, ...] = (
    "timestamp",        # Local wall-clock hour ("YYYY-MM-DD HH:MM:SS")
    "DSTFlag",          # True on the repeated 01:00 hour during November DST fall-back
    "source",           # Data provenance: "archive" (through Sep 2026) or "api" (Oct-Nov 2026)
    "uri_flag",         # True during Winter Storm Uri event window (Feb 14-20, 2021)
    "system_total_mw",  # Total ERCOT system-wide electricity demand in Megawatts (repeats across 8 zones)
    "zone",             # One of the 8 ERCOT weather/load zones
    "load_mw",          # Demand specifically within that single zone in MW
    "temp",             # Air temperature in Celsius
    "dwpt",             # Dew point temperature in Celsius
    "rhum",             # Relative humidity (%)
    "wspd",             # Wind speed (km/h)
    "prcp",             # Precipitation (mm)
    "temp_f",           # Air temperature in Fahrenheit
)

# All 8 ERCOT zones present at each hourly timestamp
EXPECTED_ZONES: tuple[str, ...] = (
    "coast",
    "east",
    "farWest",
    "north",
    "northC",
    "southC",
    "southern",
    "west",
)

# Expected number of unique hourly system records after deduplicating the 8 zones
EXPECTED_TIMESTAMPS: int = 77_496

# Numerical weather columns tracked for missingness auditing and training-only imputation
WEATHER_COLUMNS: tuple[str, ...] = ("temp", "dwpt", "rhum", "wspd", "prcp", "temp_f")

# ---------------------------------------------------------------------------
# Chronological Split Boundaries (Proposal Section 3.2)
# Time-series data is NEVER randomly shuffled; past trains future evaluation.
# ---------------------------------------------------------------------------
TRAIN_START: str = "2018-01-01 00:00:00"  # 6 years of historical training data
TRAIN_END: str = "2023-12-31 23:00:00"
VAL_START: str = "2024-01-01 00:00:00"    # 2 years reserved for hyperparameter tuning & early stopping
VAL_END: str = "2025-12-31 23:00:00"
TEST_START: str = "2026-01-01 00:00:00"   # Held-out 2026 test period evaluated once after fixing models
TEST_END: str = "2026-11-04 23:00:00"

# ---------------------------------------------------------------------------
# Representative Battery QP Planning Assumptions (Proposal Section 4.3)
# Note: The raw CSV has no physical battery telemetry, so these define our
# baseline synthetic grid-scale battery used in Step 4/5 sensitivity tests.
# ---------------------------------------------------------------------------
BATTERY_DEFAULTS: dict[str, float] = {
    "E_mwh": 2000.0,       # Maximum battery energy storage capacity (2,000 MWh)
    "C_max_mw": 500.0,     # Maximum charging rate in any hour (500 MW)
    "D_max_mw": 500.0,     # Maximum discharging rate in any hour (500 MW)
    "eta_c": 0.95,         # Charging efficiency (95% of grid power enters battery)
    "eta_d": 0.95,         # Discharging efficiency (5% loss when pushing energy back to grid)
    "s_init_mwh": 1000.0,  # Initial & end-of-day required state of charge (50% full = 1,000 MWh)
    "lambda_reg": 1e-4,    # Quadratic penalty weight on (c_t^2 + d_t^2) to smooth large charge spikes
    "mu_wear": 0.05,       # Linear penalty weight on (c_t + d_t) to discourage unnecessary battery cycling
}

# ---------------------------------------------------------------------------
# Antigravity Dark Carbon Palette (from workspace GEMINI.md design system)
# Neutral carbon gray backgrounds (#181818 / #1F1F1F) paired with high-contrast
# White, Blue, and Yellow text/series accents for all plots & motion graphics.
# ---------------------------------------------------------------------------
DARK_CARBON_PALETTE: dict[str, str] = {
    "bg_darkest": "#181818",   # Outer figure background / app shell
    "bg_primary": "#1F1F1F",   # Plot canvas / primary workspace background
    "bg_secondary": "#252526", # Subtle shaded band (e.g., train period region)
    "bg_card": "#2B2B2B",      # Gridlines and legend container background
    "border_input": "#3C3C3C", # Axis spines and legend borders
    "text_bright": "#FFFFFF",  # Axis titles / high-contrast labels
    "text_main": "#CCCCCC",    # Tick labels, legend text, and bar callouts
    "text_muted": "#858585",   # Secondary annotations
    "text_blue": "#569CD6",    # Plot titles and histogram bars
    "text_sky": "#9CDCFE",     # Secondary curves and low-demand annotations
    "text_link": "#3794FF",    # Primary time-series line
    "text_yellow": "#DCDCAA",  # Diurnal curve and test split boundary line
    "text_gold": "#CCA700",    # Validation boundary and summer peak bars
    "text_orange": "#CE9178",  # Highlight accents
    "text_green": "#73C991",   # Positive / charging flow accents
    "text_red": "#F14C4C",     # Extreme events (Winter Storm Uri) & peak load markers
}
