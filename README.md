# ERCOT Hourly Load Forecasting & Battery Dispatch Optimization

**Course:** ASDS 6304 Optimization and Big Data Analytics — Fall 2026  
**Group 3:** Roshan Adhikari, Ryan Bui, Alemayehu Worash, Bikesh Adhikari  
**Proposal:** [ERCOT_Hourly_Load_Forecasting_Optimization_Proposal.pdf](ERCOT_Hourly_Load_Forecasting_Optimization_Proposal.pdf)

---

## Project Overview

This repository implements an end-to-end **forecast-to-decision optimization pipeline** on historical ERCOT electricity demand (`2018–2026`):

1. **24-Hour Day-Ahead Demand Forecasting:** At the end of day $d$, predict hourly ERCOT system demand $\hat{L}_{d+1, 1..24}$ using causal past load sequences and deterministic calendar features (excluding target-day observed weather to avoid look-ahead leakage).
2. **Numerical Optimization Studies (Course Methods):**
   - Compare **Gradient Descent** vs. **Conjugate Gradient** on the Ridge regression quadratic objective.
   - Compare **Full-Batch Gradient Descent with Backtracking Line Search** vs. **Mini-Batch SGD** on a controlled PyTorch feed-forward network.
3. **Constrained Battery Dispatch Optimization (Convex QP):** Feed each model's 24-hour forecast into a `CVXPY` quadratic program choosing hourly charging $c_t$, discharging $d_t$, and state of charge $s_t$ to shave daily peak load $P$ while penalizing battery wear.
4. **Cinema-Style Motion Graphics Explainer (Step 6 / Option 3):** Render 60fps MP4 animations driven directly by backend forecasts, optimizer trajectories, and live 24-hour battery State-of-Charge ($s_t$) peak-shaving playback.

---

## Current Status: Step 0 Complete

**Base Step 0** audits the raw multi-zone dataset (`ercot_model_table.csv`, excluded from git via `.gitignore` due to size >50 MB), deduplicates it into a clean hourly system series (`data/ercot_system_hourly.csv`), constructs leakage-safe chronological splits and feature tensors, generates Dark Carbon EDA figures, and validates everything with unit tests.

### Verified Step 0 Audit Results (`outputs/step0_audit_summary.json`)

| Metric | Observed Value |
| :--- | :--- |
| **Raw Table Rows & Columns** | `619,968` rows $\times$ `13` columns |
| **Zones Covered** | `8` (`coast`, `east`, `farWest`, `north`, `northC`, `southC`, `southern`, `west` — `77,496` rows each) |
| **Deduplicated System Hours** | `77,496` unique `(timestamp, DSTFlag)` rows (`2018-01-01 00:00:00` to `2026-11-04 23:00:00`) |
| **Cross-Zone `system_total_mw` Spread** | `0.000000 MW` (exact duplicate across all 8 zones verified) |
| **Data Sources** | `613,432` archive rows (`2018-01-01` – `2026-09-30`) + `6,536` API rows (`2026-10-01` – `2026-11-04`) |
| **DST Fall-Back Repeated `01:00:00`** | `9` timestamps (paired `DSTFlag=False` / `DSTFlag=True`) |
| **Missing Calendar Date Detected** | `2026-10-30` (`1` missing day in API window; handled explicitly so lags never misalign) |
| **Raw Missing Weather (All in Train)** | `temp`: 67, `temp_f`: 67, `dwpt`: 67, `rhum`: 66, `wspd`: 7 (`prcp`: 0) |
| **Winter Storm Uri (`uri_flag=True`)** | `168` system hours (`2021-02-14` – `2021-02-20`, peak `69,692.49 MW` prior to forced load shed) |

### Chronological Splits (Zero Shuffling / Zero Future Leakage)

| Split | Period | Hourly Rows | Day-Ahead Samples | Mean Load (MW) | Peak Load (MW) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Train** | `2018-01-01` – `2023-12-31` | `52,584` | `2,184` | `45,812.30` | `85,464.12` |
| **Validation** | `2024-01-01` – `2025-12-31` | `17,544` | `731` | `54,133.91` | `85,198.85` |
| **Test** | `2026-01-01` – `2026-11-04` | `7,368` | `306` | `59,694.34` | `91,133.73` |

Day-ahead feature matrix shape: `(3,221, 67)` $\to$ 24-hour target `(3,221, 24)`. All imputers and feature scalers are fit strictly on `train` (`2018–2023`). Observed target-day weather is isolated in `upper_bound_weather_only` and excluded from primary forecasting inputs.

---

## Exploratory Data Analysis (Step 0 Figures)

* `figures/fig1_weekly_avg_load.png` — Weekly average ERCOT system load (2018–2026) with Train/Val/Test bands and Winter Storm Uri highlight
* `figures/fig2_hourly_load_profile.png` — Diurnal load profile by hour of day (`00–23`, trough `40,869 MW` at `03:00`, peak `56,147 MW` at `17:00`)
* `figures/fig3_monthly_load_profile.png` — Monthly average demand (`Aug` peak `61.2k MW`, `Jul` `59.1k MW`)
* `figures/fig4_load_distribution_and_uri.png` — Hourly demand histogram (`99th pct: 81,445 MW`, `max: 91,134 MW`) & Feb 2021 Winter Storm Uri window

---

## Repository Structure

```text
.
├── ERCOT_Hourly_Load_Forecasting_Optimization_Proposal.pdf
├── MASTER_PLAN.md                  # Full 6-step engineering & video roadmap
├── README.md
├── step0_base_audit.py             # Step 0 audit, feature builder & EDA plot generator
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, schema constants, splits, battery QP defaults & palette
│   └── data_pipeline.py            # Pure CSV/NumPy audit, deduplication & leakage-safe features
├── data/
│   └── ercot_system_hourly.csv     # Deduplicated 77,496-hour system load & weather series
├── outputs/
│   └── step0_audit_summary.json    # Complete machine-readable audit report
├── figures/
│   ├── fig1_weekly_avg_load.png
│   ├── fig2_hourly_load_profile.png
│   ├── fig3_monthly_load_profile.png
│   └── fig4_load_distribution_and_uri.png
└── tests/
    ├── __init__.py
    └── test_step0.py               # Audit, split ordering, and zero-leakage unit tests
```

---

## How to Run Step 0

Place `ercot_model_table.csv` in the repository root and run:

```powershell
python step0_base_audit.py
python -m unittest discover -s tests -p "test_*.py" -v
```

> **Note:** On Windows environments where Application Control blocks `pandas` C-extensions (`nattype`), `src/data_pipeline.py` uses Python's standard `csv` module + `numpy` directly so Step 0 runs with zero `pandas` dependency.
