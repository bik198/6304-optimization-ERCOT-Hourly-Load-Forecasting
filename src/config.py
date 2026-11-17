"""Central configuration, dataset schema constants, splits, and palette."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
RAW_CSV_PATH: Path = ROOT_DIR / "ercot_model_table.csv"
DATA_DIR: Path = ROOT_DIR / "data"
SYSTEM_CSV_PATH: Path = DATA_DIR / "ercot_system_hourly.csv"
FEATURES_NPZ_PATH: Path = DATA_DIR / "ercot_step0_features.npz"
FIGURES_DIR: Path = ROOT_DIR / "figures"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
AUDIT_JSON_PATH: Path = OUTPUTS_DIR / "step0_audit_summary.json"

# Expected Dataset Profile Constants (Proposal Section 3.1)
EXPECTED_TOTAL_ROWS: int = 619_968
EXPECTED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "DSTFlag",
    "source",
    "uri_flag",
    "system_total_mw",
    "zone",
    "load_mw",
    "temp",
    "dwpt",
    "rhum",
    "wspd",
    "prcp",
    "temp_f",
)
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
EXPECTED_TIMESTAMPS: int = 77_496
WEATHER_COLUMNS: tuple[str, ...] = ("temp", "dwpt", "rhum", "wspd", "prcp", "temp_f")

# Chronological Split Boundaries (Proposal Section 3.2)
TRAIN_START: str = "2018-01-01 00:00:00"
TRAIN_END: str = "2023-12-31 23:00:00"
VAL_START: str = "2024-01-01 00:00:00"
VAL_END: str = "2025-12-31 23:00:00"
TEST_START: str = "2026-01-01 00:00:00"
TEST_END: str = "2026-11-04 23:00:00"

# Representative Battery QP Parameters (Proposal Section 4.3)
BATTERY_DEFAULTS: dict[str, float] = {
    "E_mwh": 2000.0,
    "C_max_mw": 500.0,
    "D_max_mw": 500.0,
    "eta_c": 0.95,
    "eta_d": 0.95,
    "s_init_mwh": 1000.0,
    "lambda_reg": 1e-4,
    "mu_wear": 0.05,
}

# Antigravity Dark Carbon Palette (GEMINI.md)
DARK_CARBON_PALETTE: dict[str, str] = {
    "bg_darkest": "#181818",
    "bg_primary": "#1F1F1F",
    "bg_secondary": "#252526",
    "bg_card": "#2B2B2B",
    "border_input": "#3C3C3C",
    "text_bright": "#FFFFFF",
    "text_main": "#CCCCCC",
    "text_muted": "#858585",
    "text_blue": "#569CD6",
    "text_sky": "#9CDCFE",
    "text_link": "#3794FF",
    "text_yellow": "#DCDCAA",
    "text_gold": "#CCA700",
    "text_orange": "#CE9178",
    "text_green": "#73C991",
    "text_red": "#F14C4C",
}
