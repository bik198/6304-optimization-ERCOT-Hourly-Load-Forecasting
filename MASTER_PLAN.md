# Master Implementation Plan — ERCOT Forecasting, Optimization & Motion Graphics

**Course:** ASDS 6304 Optimization and Big Data Analytics (Fall 2026)  
**Group 3:** Roshan Adhikari, Ryan Bui, Alemayehu Worash, Bikesh Adhikari  
**Target Repository:** `https://github.com/ijoygigabyte-art/6304-optimization-ERCOT-Hourly-Load-Forecasting.git`

---

## Roadmap Overview (Steps 0 – 6)

| Step | Scope & Deliverables | Status |
| :--- | :--- | :--- |
| **Step 0** | **Base Setup, Data Audit, Leakage-Safe Deduplication & EDA** | **Done** |
| **Step 1** | **Baselines & Course AR Benchmark (`Seasonal-Naive`, `AR(p)` + AIC/ACF/PACF)** | Planned |
| **Step 2** | **ML Models (`Ridge`, `RBF-SVR`) + Optimizer Study #1 (GD vs. Conjugate Gradient)** | Planned |
| **Step 3** | **Deep Learning (`PyTorch LSTM` & FFN) + Optimizer Study #2 (Batch GD + Backtracking vs. Mini-Batch SGD)** | Planned |
| **Step 4** | **Constrained Convex QP for 24-Hour Battery Dispatch & Peak Shaving (`CVXPY`)** | Planned |
| **Step 5** | **Chronological Holdout Evaluation, Regret vs. Perfect Foresight & Sensitivity Analysis** | Planned |
| **Step 6** | **Option 3: Cinema-Style 60fps Motion Graphics Video Pipeline (Manim / Remotion)** | Planned |

---

## Step-by-Step Technical Specification

### Step 0 — Base Setup, Data Audit, Leakage-Safe Deduplication & EDA *(Completed)*
- **Files:** `src/config.py`, `src/data_pipeline.py`, `step0_base_audit.py`, `tests/test_step0.py`, `data/ercot_system_hourly.csv`, `outputs/step0_audit_summary.json`, `figures/fig1..fig4.png`.
- **Key Safeguards Implemented:**
  - Verified `system_total_mw` repeats identically across all 8 ERCOT zones (`0.0 MW` max spread) and deduplicated `619,968` rows to `77,496` unique `(timestamp, DSTFlag)` hours.
  - Audited `9` fall-back DST repeated `01:00:00` hours, `9` spring-forward 23-hour days, `1` missing calendar date (`2026-10-30`), `168` Winter Storm Uri hours (`2021-02-14`–`2021-02-20`), and `train`-only missing weather values.
  - Built leakage-safe `(3221, 67) -> (3221, 24)` day-ahead splits (`train: 2184d`, `val: 731d`, `test: 306d`) with `train`-only standardization and isolated `upper_bound_weather_only`.

### Step 1 — Baselines & Autoregressive Course Benchmark
- **Models:**
  1. `Previous Day` and `Previous Week Same-Hour` seasonal-naive operational references.
  2. Course-aligned Autoregressive $\text{AR}(p)$ benchmark (`statsmodels` / Yule-Walker or OLS fallback) with **AIC order selection**, **ACF/PACF plots**, and **Ljung-Box residual autocorrelation checks**.
- **Outputs:** Validation/test MAE & RMSE in MW, residual diagnostic figures, and rolling-origin error curves.

### Step 2 — Machine Learning (`Ridge` & `RBF-SVR`) + Optimizer Study #1
- **Forecasting Models:**
  - Multi-output `Ridge` regression and `RBF-kernel SVR` (`scikit-learn`), tuned strictly on chronological validation (`2024–2025`).
  - Optional upper-bound experiment incorporating `upper_bound_weather_only` (explicitly labeled as non-causal oracle weather).
- **Course Optimizer Study #1 (Ridge Convex Quadratic Objective):**
  - Implement from scratch in NumPy:
    1. **Gradient Descent** (fixed step & backtracking line search)
    2. **Linear Conjugate Gradient**
  - Track and compare objective gap $f(w_k) - f(w^*)$, gradient norm $\|\nabla f(w_k)\|_2$, iterations, and runtime.

### Step 3 — Deep Learning (`PyTorch LSTM` & Feed-Forward Net) + Optimizer Study #2
- **Primary Deep Forecaster:** PyTorch sequence `LSTM` taking past hourly demand sequences and predicting the next 24 hours, trained with mini-batch SGD and validation early stopping.
- **Controlled Optimizer Study #2 (PyTorch Feed-Forward Network):**
  - Fix identical network architecture, initialization seed, and dataset split; train separately with:
    1. **Full-Batch Gradient Descent + Backtracking Armijo Line Search**
    2. **Mini-Batch Stochastic Gradient Descent**
  - Compare loss curves, gradient norm decay, wall-clock time, and validation/test forecast accuracy.

### Step 4 — Constrained Battery Dispatch Optimization (`CVXPY` QP)
- **Formulation (Proposal Section 4.3):**
  For each 24-hour forecast $\hat{L}_t$ ($t = 1,\dots,24$), solve the convex quadratic program:
  $$\min_{P,\,c,\,d,\,s} \; P + \lambda \sum_{t=1}^{24} (c_t^2 + d_t^2) + \mu \sum_{t=1}^{24} (c_t + d_t)$$
  subject to:
  - Peak ceiling: $P \ge \hat{L}_t + c_t - d_t \quad \forall t = 1,\dots,24$
  - Battery energy balance: $s_{t+1} = s_t + \eta_c c_t - d_t / \eta_d$, $\quad 0 \le s_t \le E$
  - Power & cyclic boundary limits: $0 \le c_t \le C_{\max}$, $\quad 0 \le d_t \le D_{\max}$, $\quad s_1 = s_{25} = s_{\text{init}}$
- **Representative Defaults (`src/config.py`):** $E = 2000\text{ MWh}$, $C_{\max} = D_{\max} = 500\text{ MW}$, $\eta_c = \eta_d = 0.95$, $s_{\text{init}} = 1000\text{ MWh}$, $\lambda = 10^{-4}$, $\mu = 0.05$.

### Step 5 — End-to-End Evaluation, Regret & Sensitivity Analysis
- Run each model's 24-hour test forecasts (`2026`) through the identical QP battery dispatcher and evaluate realized net load $L_t^{\text{real}} + c_t - d_t$.
- Report **Peak Reduction (%) vs. No Storage**, **Daily Throughput**, **Feasibility**, and **Decision Regret** vs. the offline **Perfect-Foresight QP**.
- Perform sensitivity sweeps across battery energy capacity $E$, power limits $C_{\max}/D_{\max}$, wear penalties $(\lambda, \mu)$, seasons, peak summer days, and Winter Storm Uri robustness (`uri_flag`).

### Step 6 — Option 3: Cinema-Style 60fps Motion Graphics Explainer Video
- Build an automated rendering pipeline (Python **Manim** / SVG-frame renderer + **FFmpeg**) powered directly by serialized JSON trajectories from Steps 1–5:
  1. **Scene 1 — Problem & Data Scale:** Animated zoom from `619,968` multi-zone rows to deduplicated ERCOT demand curves & Winter Storm Uri shock.
  2. **Scene 2 — Optimization Under the Hood:** Live animated comparison racing **Gradient Descent** vs. **Conjugate Gradient** (Ridge contours) and **Batch GD + Backtracking** vs. **Mini-Batch SGD** (loss landscape & step sizes).
  3. **Scene 3 — Real-Time 24h Battery Dispatch & Peak Shaving:** Hour-by-hour `00:00 -> 23:00` animation synced across:
     - Forecasted ($\hat{L}_t$) vs. Realized ($L_t$) vs. Net Load ($L_t + c_t - d_t$) curve with shaded peak-shaving zone below ceiling $P$.
     - Glowing Dark Carbon battery core animating State of Charge $s_t$, green charging flow ($c_t > 0$) during overnight troughs (`02:00–06:00`), and gold/red discharging flow ($d_t > 0$) during afternoon peaks (`15:00–19:00`).
     - Side-by-side **Regret Replay** comparing `Ridge` vs. `RBF-SVR` vs. `LSTM` vs. `Perfect Foresight` on a high-demand `2026` summer peak day.
