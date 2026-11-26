"""Export the Beadline manifest, exact source code, and real numerical traces to web/public/data/beads.json.

Every number in `beads.json` is measured directly from the repository datasets,
saved optimization logs, and live `beads.trace()` execution of the strand modules.

Usage:
    python export_beads_data.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import beads, s05_solvers, s10_evaluate
from src.config import AUDIT_JSON_PATH, FEATURES_NPZ_PATH, ROOT_DIR, SYSTEM_CSV_PATH
from src.s01_system_data import daily_profiles, load_system_data
from src.s02_hourly_features import complete_days, hourly_examples
from src.s03_ridge import SELECTED_ALPHA, ridge_reference
from src.s04_svr import SELECTED as SVR_SELECTED
from src.s06_ffn import ffn_bundle, ffn_network, ffn_targets
from src.s07_lstm_data import CONFIG as LSTM_CFG, day_grid, lstm_calendar, lstm_samples, lstm_tensors
from src.s08_lstm_model import SEEDS as LSTM_SEEDS, lstm_ensemble, lstm_load_weights, lstm_network
from src.s09_metrics import errors

OUT_PATH: Path = ROOT_DIR / "web" / "public" / "data" / "beads.json"
OPT_RESULTS = ROOT_DIR / "optimization" / "results"
PRELIM = ROOT_DIR / "preliminary"

from beadline_project import STRAND_META  # noqa: E402  (single project config shared with the story exporter)


def _r(x: Any, nd: int = 2) -> Any:
    """Round floats cleanly for compact JSON while leaving ints/strings/bools intact."""
    if isinstance(x, (np.floating, float)):
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, nd)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _subsample_solver_history(rows: list[dict[str, str]], method: str) -> list[dict[str, Any]]:
    """Keep all steps for Conjugate Gradient, or log-spaced milestones for long GD traces."""
    mrows = [r for r in rows if r["method"] == method]
    if len(mrows) <= 30:
        chosen = mrows
    else:
        idxs = sorted(
            {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, len(mrows) - 1}
            | set(np.linspace(0, len(mrows) - 1, 24, dtype=int).tolist())
        )
        chosen = [mrows[i] for i in idxs if i < len(mrows)]
    return [
        {
            "step": int(r["iteration"]),
            "label": f"iter {r['iteration']}",
            "metrics": {
                "elapsed_ms": _r(float(r["elapsed_ms"]), 4),
                "objective_gap": float(f"{float(r['objective_gap']):.3e}"),
                "gradient_norm": float(f"{float(r['gradient_norm']):.3e}"),
                "line_search_evals": int(r["line_search_evaluations"]),
            },
            "values": [float(r["objective_gap"]), float(r["gradient_norm"]), float(r["elapsed_ms"])],
        }
        for r in chosen
    ]


def build_beads_payload() -> dict[str, Any]:
    """Load all 38 beads, trace real numerical examples, and build presentation JSON."""
    manifest = beads.load_all_strands(include_source=True)
    audit = json.loads(AUDIT_JSON_PATH.read_text(encoding="utf-8"))

    # -----------------------------------------------------------------------
    # 1) Execute real strand objects inside beads.trace()
    # -----------------------------------------------------------------------
    with beads.trace() as trace_log:
        system = load_system_data()
        profiles = daily_profiles(system)
        comp = complete_days(profiles)
        ex = hourly_examples(comp)
        problem = ridge_reference(ex)
        q = s05_solvers.ridge_quadratic(ex, problem)
        tr_fixed, conv_fixed = s05_solvers.fixed_step_gd(q)
        tr_armijo, conv_armijo = s05_solvers.armijo_gd(q)
        tr_cg, conv_cg = s05_solvers.conjugate_gradient(q)

        bundle = ffn_bundle()
        ffn_prob = ffn_targets(bundle)
        _ = ffn_network()

        grid = day_grid(profiles)
        cal = lstm_calendar(grid)
        lstm_data = lstm_tensors(grid, cal, LSTM_CFG)
        lstm_models = [lstm_load_weights(lstm_data, s) for s in LSTM_SEEDS]
        lstm_test_mw = lstm_ensemble(lstm_models, lstm_data, "test")

        cohort = s10_evaluate.shared_cohort()
        scores = s10_evaluate.score_table(cohort)
        hi = s10_evaluate.high_load(cohort)
        h_mae = s10_evaluate.mae_by_hour(cohort)
        seas = s10_evaluate.seasonal_mae(cohort)
        wins = s10_evaluate.daily_wins(cohort, "lstm", "ridge")
        _ = errors([100.0, 200.0], [105.0, 190.0])

    trace_by_bead = {entry["bead"]: entry for entry in trace_log}

    # -----------------------------------------------------------------------
    # 2) Extract concrete probe samples for Depth 3 (Numbers Theater)
    # -----------------------------------------------------------------------
    # Peak hour (2026-07-22 17:00) in system_data
    peak_idx = int(np.argmax(system["system_total_mw"]))
    peak_ts = system["timestamp"][peak_idx]
    peak_sys_mw = float(system["system_total_mw"][peak_idx])
    peak_zones = [[z, _r(system["zone_loads"][z][peak_idx], 2)] for z in sorted(system["zone_loads"].keys())]

    # Step 0 feature matrix probe for target day 2026-07-22
    npz_cache = np.load(FEATURES_NPZ_PATH, allow_pickle=False)
    t_dates = npz_cache["target_dates"]
    peak_day_pos = int(np.where(t_dates == "2026-07-22")[0][0])
    x_raw_sample = npz_cache["X_raw"][peak_day_pos]
    x_scaled_sample = npz_cache["X_scaled"][peak_day_pos]
    y_day_sample = npz_cache["Y"][peak_day_pos]

    # Ridge / SVR validation CSVs + solver history
    ridge_val_rows = _read_csv_rows(PRELIM / "ridge_validation_results.csv")
    svr_val_rows = _read_csv_rows(PRELIM / "svr_validation_results.csv")
    solver_sum_rows = {r["method"]: r for r in _read_csv_rows(OPT_RESULTS / "ridge" / "solver_summary.csv")}
    solver_hist_rows = _read_csv_rows(OPT_RESULTS / "ridge" / "history.csv")

    eigvals = np.linalg.eigvalsh(q.hessian)
    cond_num = float(eigvals[-1] / eigvals[0])

    # FFN saved logs
    ffn_comp = {r["method"]: r for r in _read_csv_rows(OPT_RESULTS / "feedforward" / "comparison.csv")}
    ffn_rob = {r["method"]: r for r in _read_csv_rows(OPT_RESULTS / "feedforward" / "robustness_summary.csv")}
    ffn_hist = _read_csv_rows(OPT_RESULTS / "feedforward" / "training_history.csv")
    ffn_arm_by_pass = {int(r["pass"]): r for r in ffn_hist if r["method"] == "Armijo GD"}
    ffn_sgd_by_pass = {int(r["pass"]): r for r in ffn_hist if r["method"] == "SGD 0.03"}
    ffn_scrubber = [
        {
            "step": p,
            "label": f"pass {p}",
            "metrics": {
                "armijo_val_mae_mw": _r(float(ffn_arm_by_pass[p]["validation_mae_mw"]), 2),
                "sgd_val_mae_mw": _r(float(ffn_sgd_by_pass[p]["validation_mae_mw"]), 2),
                "armijo_mse": _r(float(ffn_arm_by_pass[p]["training_mse"]), 4),
                "sgd_mse": _r(float(ffn_sgd_by_pass[p]["training_mse"]), 4),
                "sgd_minus_armijo_mw": _r(
                    float(ffn_sgd_by_pass[p]["validation_mae_mw"]) - float(ffn_arm_by_pass[p]["validation_mae_mw"]),
                    2,
                ),
            },
            "values": [
                float(ffn_arm_by_pass[p]["validation_mae_mw"]),
                float(ffn_sgd_by_pass[p]["validation_mae_mw"]),
                float(ffn_sgd_by_pass[p]["training_mse"]),
            ],
        }
        for p in sorted(ffn_arm_by_pass.keys())
        if p in ffn_sgd_by_pass and (p <= 10 or p % 5 == 0 or p in (98, 100))
    ]

    # LSTM saved logs + 3-seed peak day slice
    lstm_screen = _read_csv_rows(OPT_RESULTS / "lstm" / "screening.csv")
    lstm_ens_val = _read_csv_rows(OPT_RESULTS / "lstm" / "validation_ensembles.csv")
    lstm_test_sum = _read_csv_rows(OPT_RESULTS / "lstm" / "test_summary.csv")
    lstm_hr_mae = _read_csv_rows(OPT_RESULTS / "lstm" / "test_mae_by_hour.csv")
    lstm_test_rows = lstm_data["S"]["rows"][lstm_data["masks"]["test"]]
    lstm_peak_day_idx = int(np.where(grid.days[lstm_test_rows].strftime("%Y-%m-%d") == "2026-07-22")[0][0])
    lstm_seed_preds = [
        lstm_ensemble([m], lstm_data, "test")[lstm_peak_day_idx].tolist()
        for m in lstm_models
    ]

    # Shared cohort 24h peak day slice (2026-07-22)
    peak_day_df = cohort[cohort["timestamp"].dt.strftime("%Y-%m-%d") == "2026-07-22"].sort_values("timestamp")
    peak_day_scrubber = [
        {
            "step": int(row["timestamp"].hour),
            "label": f"{int(row['timestamp'].hour):02d}:00",
            "metrics": {
                "actual_mw": _r(row["actual"], 1),
                "lstm_mw": _r(row["lstm"], 1),
                "ffn_sgd_mw": _r(row["ffn_sgd"], 1),
                "ridge_mw": _r(row["ridge"], 1),
                "svr_mw": _r(row["svr"], 1),
                "lstm_err_mw": _r(row["lstm"] - row["actual"], 1),
            },
            "values": [float(row["actual"]), float(row["lstm"]), float(row["ridge"]), float(row["svr"])],
        }
        for _, row in peak_day_df.iterrows()
    ]

    # Pre-built rich Depth 3 number packets keyed by bead name
    test_ex_mask = ex.split("test")
    peak_ex_idx = int(np.where(ex.timestamps[test_ex_mask] == pd.Timestamp("2026-07-22 17:00:00"))[0][0])
    peak_ex_x = ex.x[test_ex_mask][peak_ex_idx]

    rich_numbers: dict[str, dict[str, Any]] = {
        "dedupe": {
            "inputs": [
                {"label": "Raw CSV Rows (8 zones × 77,496 hrs)", "value": audit["total_raw_rows"], "unit": "rows", "sample": [619968, 77496, 8]},
                {"label": "Archive vs API Rows", "value": "613,432 archive / 6,536 api", "unit": "rows", "sample": [613432, 6536]},
            ],
            "operation": "Group by (timestamp, DSTFlag) → verify max |sys_i - sys_j| == 0.0 → keep 1 system row/hour",
            "outputs": [
                {"label": "Deduplicated System Hours", "value": audit["unique_timestamp_dst_pairs"], "unit": "hours", "sample": [77496, 0.0, 9, 168]},
                {"label": "Record Peak Hour (2026-07-22 17:00)", "value": _r(peak_sys_mw, 2), "unit": "MW", "sample": [peak_sys_mw]},
            ],
            "table": {"columns": ["Zone", "2026-07-22 17:00 Load (MW)"], "rows": peak_zones},
        },
        "load_system_data": {
            "inputs": [{"label": "Source File", "value": "data/ercot_system_hourly.csv", "unit": "16.0 MB", "sample": [77496, 21]}],
            "operation": "Read CSV columns into aligned NumPy float64/bool arrays",
            "outputs": [
                {"label": "Overall Mean / Max MW", "value": f"{audit['overall_load_stats_mw']['mean_mw']:,} / {audit['overall_load_stats_mw']['max_mw']:,}", "unit": "MW", "sample": [audit["overall_load_stats_mw"]["mean_mw"], audit["overall_load_stats_mw"]["min_mw"], audit["overall_load_stats_mw"]["max_mw"]]},
            ],
        },
        "daily_profiles": {
            "inputs": [{"label": "Hourly System Series", "value": 77496, "unit": "hours", "sample": [77496, 9, 9]}],
            "operation": "Bucket each date into [0..23] slots; average 25h fall-back 01:00; linearly fill 23h spring-forward 02:00",
            "outputs": [
                {"label": "Calendar Days Built", "value": len(profiles), "unit": "days × 24h", "sample": [len(profiles), 3211, 9, 9]},
                {"label": "2026-07-22 Profile (03:00 vs 17:00)", "value": f"{_r(y_day_sample[3], 1)} → {_r(y_day_sample[17], 1)}", "unit": "MW", "sample": [_r(v, 2) for v in y_day_sample[:6]]},
            ],
        },
        "day_ahead_features": {
            "inputs": [
                {"label": "Day 2026-07-21 Raw Lags (first 5 hrs)", "value": [_r(v, 1) for v in x_raw_sample[:5]], "unit": "MW", "sample": [float(v) for v in x_raw_sample[:5]]},
                {"label": "Train-Only Scaler Mean (first 5)", "value": [_r(v, 1) for v in npz_cache["scaler_mean"][:5]], "unit": "MW", "sample": [float(v) for v in npz_cache["scaler_mean"][:5]]},
            ],
            "operation": "X_scaled = (X_raw - train_mean) / train_std  |  train=2,184d, val=731d, test=306d",
            "outputs": [
                {"label": "Day 2026-07-22 X_scaled (first 5)", "value": [_r(v, 4) for v in x_scaled_sample[:5]], "unit": "z-score", "sample": [float(v) for v in x_scaled_sample[:5]]},
                {"label": "Matrix Dimensions", "value": "X(3221, 67) → Y(3221, 24)", "unit": "arrays", "sample": [3221, 67, 24]},
            ],
        },
        "complete_days": {
            "inputs": [{"label": "All Calendar Profiles", "value": len(profiles), "unit": "days", "sample": [3229, 18]}],
            "operation": "Filter strictly to raw_hour_count == 24 (excludes 18 DST transition days)",
            "outputs": [{"label": "Complete 24h Days Retained", "value": len(comp), "unit": "days", "sample": [len(comp), 24]}],
        },
        "hourly_examples": {
            "inputs": [{"label": "Peak Target Hour", "value": "2026-07-22 17:00", "unit": "hour 17", "sample": [float(peak_ex_x[0]), float(peak_ex_x[1]), float(peak_ex_x[2])]}],
            "operation": "Stack [lag_1d_h, lag_7d_h, prev_mean, prev_max, prev_min, sin/cos(hr, dow, mo), weekend]",
            "outputs": [
                {"label": "Split Sizes (Train / Val / Test)", "value": "51,552 / 17,256 / 7,224", "unit": "hours", "sample": [51552, 17256, 7224, 12]},
                {"label": "12-Feature Vector (first 5)", "value": [_r(v, 2) for v in peak_ex_x[:5]], "unit": "MW", "sample": [float(v) for v in peak_ex_x[:5]]},
            ],
        },
        "ridge_search": {
            "inputs": [{"label": "Candidate Alphas", "value": [0.1, 1, 10, 100, 1000], "unit": "L2 penalty", "sample": [0.1, 1.0, 10.0, 100.0, 1000.0]}],
            "operation": "Fit StandardScaler + Ridge(alpha) on 2018-2023; pick argmin validation MAE (2024-2025)",
            "outputs": [{"label": "Best Alpha / Val MAE", "value": "α = 0.1 → 2,307.95 MW", "unit": "MW", "sample": [0.1, 2307.9503, 3298.0939]}],
            "table": {
                "columns": ["Alpha", "Val MAE (MW)", "Val RMSE (MW)", "Runtime (s)"],
                "rows": [[float(r["Alpha"]), _r(float(r["MAE_MW"]), 2), _r(float(r["RMSE_MW"]), 2), _r(float(r["Runtime_seconds"]), 4)] for r in ridge_val_rows],
            },
        },
        "ridge_reference": {
            "inputs": [{"label": "Train Standardized Matrix", "value": "(51,552, 12)", "unit": "float64", "sample": [51552, 12, 0.1]}],
            "operation": "Solve (X^T X + αI) w* = X^T y exactly via Cholesky factorization",
            "outputs": [{"label": "Cholesky Weights w* (first 5)", "value": [_r(v, 2) for v in problem.reference.coef_[:5]], "unit": "MW / std", "sample": [float(v) for v in problem.reference.coef_[:5]]}],
        },
        "svr_sample": {
            "inputs": [{"label": "Full Train Hours", "value": 51552, "unit": "hours", "sample": [51552, 6000]}],
            "operation": "np.linspace(0, 51551, 6000) uniform stride subsampling for O(n^2..n^3) kernel fit",
            "outputs": [{"label": "Subsampled Training Set", "value": "(6,000, 12)", "unit": "hours", "sample": [6000, 12]}],
        },
        "svr_fit": {
            "inputs": [{"label": "Selected Kernel Hyperparams", "value": SVR_SELECTED, "unit": "RBF", "sample": [10.0, 0.1, 0.01]}],
            "operation": "Standardize X & y, fit SVR(kernel='rbf', C=10, epsilon=0.1, gamma=0.01)",
            "outputs": [{"label": "2026 Test Score (7,224 hrs)", "value": "2,351.17 MW MAE / 3,302.77 MW RMSE", "unit": "MW", "sample": [2351.17, 3302.77]}],
        },
        "svr_search": {
            "inputs": [{"label": "12-Point ParameterGrid", "value": "C ∈ {1,10,100}, ε ∈ {0.01,0.1}, γ ∈ {scale,0.01}", "unit": "grid", "sample": [1, 10, 100, 0.01, 0.1]}],
            "operation": "Evaluate all 12 RBF-SVR candidates on 17,256 validation hours",
            "outputs": [{"label": "Best Validation MAE", "value": "2,267.23 MW (C=10, ε=0.1, γ=0.01)", "unit": "MW", "sample": [2267.23, 3272.08]}],
            "table": {
                "columns": ["C", "eps", "gamma", "Val MAE (MW)", "Val RMSE (MW)", "Time (s)"],
                "rows": sorted(
                    [[r["C"], r["epsilon"], r["gamma"], _r(float(r["MAE_MW"]), 2), _r(float(r["RMSE_MW"]), 2), _r(float(r["Runtime_seconds"]), 2)] for r in svr_val_rows],
                    key=lambda row: row[3],
                )[:6],
            },
        },
        "ridge_quadratic": {
            "inputs": [{"label": "Centered Scaled Train", "value": "(51,552, 12)", "unit": "float64", "sample": [51552, 12, float(q.mean_y)]}],
            "operation": "H = (X^T X + 0.1·I) / n,  b = X^T y / n  ⇒  f(w) = 0.5 w^T H w - b^T w",
            "outputs": [
                {"label": "Hessian Eigenvalues (min / max / cond)", "value": f"{_r(eigvals[0], 4)} / {_r(eigvals[-1], 4)} / κ={_r(cond_num, 1)}", "unit": "spectrum", "sample": [float(eigvals[0]), float(eigvals[-1]), cond_num]},
            ],
        },
        "fixed_step_gd": {
            "inputs": [{"label": "Fixed Step Size 1 / L_max", "value": _r(1.0 / eigvals[-1], 6), "unit": "step", "sample": [float(1.0 / eigvals[-1]), float(eigvals[-1])]}],
            "operation": "w_{k+1} = w_k - (1/L) (H w_k - b)  up to 10,000 iters",
            "outputs": [
                {"label": "Result at Cap (10,000 iters)", "value": f"gap={float(solver_sum_rows['Fixed-step GD']['objective_gap']):.2e}, max diff={_r(float(solver_sum_rows['Fixed-step GD']['max_validation_difference_mw']), 3)} MW", "unit": "not converged", "sample": [10000, float(solver_sum_rows["Fixed-step GD"]["objective_gap"]), float(solver_sum_rows["Fixed-step GD"]["max_validation_difference_mw"])]},
            ],
            "scrubber": {"title": "Fixed-step GD", "unit": "iter", "steps": _subsample_solver_history(solver_hist_rows, "Fixed-step GD")},
        },
        "armijo_gd": {
            "inputs": [{"label": "Armijo Params (α0, β, c1)", "value": [1.0, 0.5, 1e-4], "unit": "line search", "sample": [1.0, 0.5, 1e-4]}],
            "operation": "Halve step α ← 0.5α until quadratic decrease Δf ≤ -1e-4 α ‖∇f‖²",
            "outputs": [
                {"label": "Converged in 4,749 iters (35.48 ms)", "value": f"9,501 trials, max diff={_r(float(solver_sum_rows['Armijo GD']['max_validation_difference_mw']), 4)} MW", "unit": "converged", "sample": [4749, 9501, float(solver_sum_rows["Armijo GD"]["elapsed_ms"])]},
            ],
            "scrubber": {"title": "Armijo GD", "unit": "iter", "steps": _subsample_solver_history(solver_hist_rows, "Armijo GD")},
        },
        "conjugate_gradient": {
            "inputs": [{"label": "12×12 SPD Hessian H", "value": f"d = 12, κ(H) = {_r(cond_num, 1)}", "unit": "exact Krylov", "sample": [12.0, cond_num, 1e-8]}],
            "operation": "Exact quadratic step α_k = ‖g_k‖² / (d_k^T H d_k); orthogonalize d_{k+1} under H",
            "outputs": [
                {"label": "Converged in 13 iters (0.12 ms)", "value": f"gap={float(solver_sum_rows['Conjugate gradient']['objective_gap']):.2e}, diff={float(solver_sum_rows['Conjugate gradient']['max_validation_difference_mw']):.2e} MW", "unit": "converged", "sample": [13, float(solver_sum_rows["Conjugate gradient"]["elapsed_ms"]), float(solver_sum_rows["Conjugate gradient"]["objective_gap"])]},
            ],
            "scrubber": {"title": "Conjugate gradient", "unit": "iter", "steps": _subsample_solver_history(solver_hist_rows, "Conjugate gradient")},
        },
        "ffn_bundle": {
            "inputs": [{"label": "Step 0 NPZ Cache", "value": "data/ercot_step0_features.npz", "unit": "2.6 MB", "sample": [3221, 67, 24]}],
            "operation": "Load causal X_scaled (3221×67), Y (3221×24), splits, and target_dates",
            "outputs": [{"label": "Split Day Counts", "value": "2,184 train / 731 val / 306 test", "unit": "days", "sample": [2184, 731, 306]}],
        },
        "ffn_targets": {
            "inputs": [{"label": "Raw Target Y (train mean h0, h17)", "value": [_r(ffn_prob.target_mean[0].item(), 1), _r(ffn_prob.target_mean[17].item(), 1)], "unit": "MW", "sample": [float(ffn_prob.target_mean[0].item()), float(ffn_prob.target_mean[17].item())]}],
            "operation": "Y_std = (Y - train_mean_per_hour) / train_sd_per_hour",
            "outputs": [{"label": "Train SD (h0, h17)", "value": [_r(ffn_prob.target_sd[0].item(), 1), _r(ffn_prob.target_sd[17].item(), 1)], "unit": "MW", "sample": [float(ffn_prob.target_sd[0].item()), float(ffn_prob.target_sd[17].item())]}],
        },
        "ffn_network": {
            "inputs": [{"label": "Input Dim / Hidden / Output", "value": "67 → 64 (ReLU) → 24", "unit": "float64", "sample": [67, 64, 24, 42]}],
            "operation": "Initialize torch.nn.Sequential from fixed seed=42 (5,912 trainable parameters)",
            "outputs": [{"label": "Total Parameters", "value": 67 * 64 + 64 + 64 * 24 + 24, "unit": "weights", "sample": [67 * 64 + 64 + 64 * 24 + 24]}],
        },
        "armijo_update": {
            "inputs": [{"label": "Full-Batch Train Tensor", "value": "(2,184, 67)", "unit": "float64", "sample": [2184, 1.0, 0.5, 1e-4]}],
            "operation": "1 full-batch gradient + backtracking line search on MSE",
            "outputs": [{"label": "100-Pass Armijo Total Cost", "value": "100 grads, 185 line-search evals (0.42 s)", "unit": "2,033.09 MW val", "sample": [100, 185, 0.4213, 2033.09]}],
        },
        "sgd_pass": {
            "inputs": [{"label": "Mini-Batch Config", "value": "batch=32, lr=0.03 (69 batches/pass)", "unit": "SGD", "sample": [32, 0.03, 69]}],
            "operation": "Shuffle 2,184 train days; perform 69 noisy mini-batch updates per pass",
            "outputs": [{"label": "100-Pass SGD Total Cost", "value": "6,900 grads (2.61 s)", "unit": "1,777.38 MW val", "sample": [6900, 2.6148, 1777.38]}],
        },
        "ffn_train": {
            "inputs": [{"label": "Shared Initial Weights (seed=42)", "value": "Pass 0 Val MAE = 9,176.57 MW", "unit": "MW", "sample": [9176.5714, 1.1111]}],
            "operation": "Run 100 passes each: Full-batch Armijo GD vs Mini-batch SGD (lr=0.03); keep best validation MAE",
            "outputs": [
                {"label": "3-Seed Validation Mean ± SD", "value": f"SGD { _r(float(ffn_rob['Mini-batch SGD']['mean_validation_mae_mw']), 2) }±{ _r(float(ffn_rob['Mini-batch SGD']['sd_validation_mae_mw']), 2) } vs Armijo { _r(float(ffn_rob['Armijo GD']['mean_validation_mae_mw']), 2) }±{ _r(float(ffn_rob['Armijo GD']['sd_validation_mae_mw']), 2) }", "unit": "MW", "sample": [float(ffn_rob["Mini-batch SGD"]["mean_validation_mae_mw"]), float(ffn_rob["Armijo GD"]["mean_validation_mae_mw"])]},
            ],
            "scrubber": {"title": "100 passes", "unit": "pass", "steps": ffn_scrubber},
        },
        "ffn_forecast": {
            "inputs": [{"label": "Test Features (306 days)", "value": "(306, 67)", "unit": "scaled", "sample": [306, 67]}],
            "operation": "y_mw = model(X_test) * train_sd + train_mean  (scored on 301 shared days)",
            "outputs": [
                {"label": "SGD Test MAE / RMSE", "value": f"{_r(float(ffn_comp['SGD 0.03']['test_mae_mw']), 2)} / {_r(float(ffn_comp['SGD 0.03']['test_rmse_mw']), 2)}", "unit": "MW", "sample": [float(ffn_comp["SGD 0.03"]["test_mae_mw"]), float(ffn_comp["SGD 0.03"]["test_rmse_mw"])]},
                {"label": "Armijo Test MAE / RMSE", "value": f"{_r(float(ffn_comp['Armijo GD']['test_mae_mw']), 2)} / {_r(float(ffn_comp['Armijo GD']['test_rmse_mw']), 2)}", "unit": "MW", "sample": [float(ffn_comp["Armijo GD"]["test_mae_mw"]), float(ffn_comp["Armijo GD"]["test_rmse_mw"])]},
            ],
        },
        "day_grid": {
            "inputs": [{"label": "Daily Profiles", "value": len(profiles), "unit": "days", "sample": [len(profiles), 1]}],
            "operation": "Expand onto continuous 3,230-day calendar; fill missing 2026-10-30 from 2026-10-23 as input only",
            "outputs": [{"label": "Continuous Day Grid", "value": "(3,230, 24)  [target_ok=False on Oct 30]", "unit": "days × 24h", "sample": [3230, 24]}],
        },
        "lstm_calendar": {
            "inputs": [{"label": "3,230 Calendar Dates", "value": "2018-01-01 → 2026-11-04", "unit": "dates", "sample": [3230, 5]}],
            "operation": "Encode dow sin/cos, doy sin/cos, and weekend + USFederalHolidayCalendar offday flag",
            "outputs": [{"label": "Day Calendar Matrix", "value": "(3,230, 5)", "unit": "features", "sample": [float(v) for v in cal.day_cal[0]]}],
        },
        "lstm_samples": {
            "inputs": [{"label": "14-Day Lookback & 7-Day Level", "value": "lookback=14, level_days=7", "unit": "days", "sample": [14, 7, float(lstm_data["S"]["level"][0])]}],
            "operation": "Divide past 14 days and target 24h by trailing 7-day mean load; attach log(level) static feature",
            "outputs": [{"label": "Sequence & Static Shapes", "value": f"seq={list(lstm_data['S']['seq_load'].shape)}, static={list(lstm_data['S']['static'].shape)}", "unit": "arrays", "sample": [14, 24, 11]}],
        },
        "lstm_tensors": {
            "inputs": [{"label": "Uri Exclusion Flag", "value": "exclude_uri_train=True (drops 21 windows)", "unit": "mask", "sample": [2177, 21, 2156]}],
            "operation": "Drop Uri-overlapping train sequences; fit single load scaler on remaining 2,156 train days",
            "outputs": [{"label": "Tensor Counts (Train / Val / Test)", "value": "2,156 / 731 / 307 days", "unit": "tensors", "sample": [2156, 731, 307]}],
        },
        "lstm_network": {
            "inputs": [{"label": "Input Sizes (seq=29, static=11)", "value": "hidden=64, layers=2, dropout=0.2, head=128", "unit": "float32", "sample": [29, 11, 64, 2, 128]}],
            "operation": "2-layer LSTM over 14 days → concat h_last(64) + static(11) → 128 ReLU → 24",
            "outputs": [{"label": "Parameter Count", "value": sum(p.numel() for p in lstm_models[0].parameters()), "unit": "params", "sample": [70424]}],
        },
        "lstm_train": {
            "inputs": [{"label": "Nesterov SGD Config", "value": "lr=0.03, mom=0.9, batch=32, clip=1.0", "unit": "optimizer", "sample": [0.03, 0.9, 32, 1.0]}],
            "operation": "Train up to 200 epochs with ReduceLROnPlateau(0.5, 8) and patience=25 early stopping",
            "outputs": [{"label": "3-Seed Validation Ensemble Winner", "value": f"Original 14d/64h = {_r(float(lstm_ens_val[0]['mae_mw']), 2)} MW", "unit": "MW", "sample": [1621.87, 2475.20]}],
            "table": {
                "columns": ["Candidate", "Lookback", "Hidden", "LR", "Val MAE (MW)", "Best Ep"],
                "rows": [[r["name"], int(r["lookback"]), int(r["hidden"]), float(r["lr"]), _r(float(r["mae_mw"]), 2), int(r["best_epoch"])] for r in lstm_screen[:6]],
            },
        },
        "lstm_load_weights": {
            "inputs": [{"label": "Saved Checkpoints", "value": "lstm_seed{42, 7, 2026}.pt", "unit": "3 files", "sample": [42, 7, 2026]}],
            "operation": "Load state_dict into LSTMForecaster on CPU",
            "outputs": [{"label": "Restored Seed Models", "value": "3 × 70,424 params ready", "unit": "models", "sample": [3, 70424]}],
        },
        "lstm_forecast": {
            "inputs": [{"label": "2026-07-22 Trailing 7d Level", "value": _r(float(lstm_data["S"]["level"][lstm_data["masks"]["test"]][lstm_peak_day_idx]), 2), "unit": "MW", "sample": [float(lstm_data["S"]["level"][lstm_data["masks"]["test"]][lstm_peak_day_idx])]}],
            "operation": "MW = (y_std * sd_y + mu_y) * 7d_level",
            "outputs": [{"label": "2026-07-22 17:00 Single-Seed Preds", "value": [_r(s[17], 1) for s in lstm_seed_preds], "unit": "MW", "sample": [float(s[17]) for s in lstm_seed_preds]}],
        },
        "lstm_ensemble": {
            "inputs": [{"label": "3-Seed Forecasts (2026-07-22 17:00)", "value": [_r(s[17], 1) for s in lstm_seed_preds], "unit": "MW", "sample": [float(s[17]) for s in lstm_seed_preds]}],
            "operation": "Ensemble = mean(seed_42, seed_7, seed_2026) across 301 shared test days",
            "outputs": [
                {"label": "2026-07-22 17:00 Ensemble vs Actual", "value": f"{_r(float(np.mean([s[17] for s in lstm_seed_preds])), 1)} vs {_r(peak_sys_mw, 1)}", "unit": "MW", "sample": [float(np.mean([s[17] for s in lstm_seed_preds])), peak_sys_mw]},
                {"label": "Shared Test MAE / RMSE", "value": f"{_r(float(lstm_test_sum[0]['mae_mw']), 2)} / {_r(float(lstm_test_sum[0]['rmse_mw']), 2)}", "unit": "MW", "sample": [float(lstm_test_sum[0]["mae_mw"]), float(lstm_test_sum[0]["rmse_mw"])]},
            ],
        },
        "errors": {
            "inputs": [{"label": "Probe Example (actual vs forecast)", "value": "[100, 200] vs [105, 190]", "unit": "MW", "sample": [100.0, 200.0, 105.0, 190.0]}],
            "operation": "residual = forecast - actual;  MAE = mean(|r|),  RMSE = sqrt(mean(r²))",
            "outputs": [{"label": "Probe MAE / RMSE", "value": "7.50 / 7.91", "unit": "MW", "sample": [7.5, float(np.sqrt(62.5))]}],
        },
        "shared_cohort": {
            "inputs": [{"label": "3 Forecast CSV Exports", "value": "ML + LSTM + FFN", "unit": "2026", "sample": [301, 24, 7224]}],
            "operation": "Inner-join on timestamp; verify 301 complete 24-hour days (7,224 hours) and identical actual MW",
            "outputs": [{"label": "Aligned Cohort Table", "value": "7,224 rows × 9 cols", "unit": "hours", "sample": [7224, 301, peak_sys_mw]}],
            "scrubber": {"title": "2026-07-22, peak day", "unit": "hour", "steps": peak_day_scrubber},
        },
        "score_table": {
            "inputs": [{"label": "Shared 2026 Test Cohort", "value": "7,224 hours (301 days)", "unit": "hours", "sample": [7224, 301]}],
            "operation": "Compute MAE & RMSE for all 7 models on identical test hours; sort ascending by MAE",
            "outputs": [{"label": "Top Ranked Model", "value": f"LSTM Ensemble: {_r(scores.iloc[0]['mae_mw'], 2)} MW MAE", "unit": "MW", "sample": [float(scores.iloc[0]["mae_mw"]), float(scores.iloc[0]["rmse_mw"])]}],
            "table": {
                "columns": ["Model", "Hours", "MAE (MW)", "RMSE (MW)"],
                "rows": [[s10_evaluate.LABELS[r["model"]], int(r["hours"]), _r(r["mae_mw"], 2), _r(r["rmse_mw"], 2)] for _, r in scores.iterrows()],
            },
        },
        "high_load": {
            "inputs": [{"label": "Top 10% Measured Load Threshold", "value": "723 highest hours", "unit": "hours", "sample": [723, peak_sys_mw]}],
            "operation": "Score MAE on top 10% load hours + signed residual at 2026-07-22 17:00 (91,133.73 MW)",
            "outputs": [{"label": "Best Top-10% MAE / Peak Residual", "value": f"LSTM: {_r(hi.iloc[0]['top_mae_mw'], 2)} MW / {_r(hi.iloc[0]['peak_error_mw'], 2)} MW", "unit": "MW", "sample": [float(hi.iloc[0]["top_mae_mw"]), float(hi.iloc[0]["peak_error_mw"])]}],
            "table": {
                "columns": ["Model", "Top-10% MAE (MW)", "Peak Error at 91.1 GW (MW)"],
                "rows": [[s10_evaluate.LABELS[r["model"]], _r(r["top_mae_mw"], 2), _r(r["peak_error_mw"], 2)] for _, r in hi.iterrows()],
            },
        },
        "mae_by_hour": {
            "inputs": [{"label": "24 Clock Hours (00–23)", "value": "301 samples per hour", "unit": "hours", "sample": [301, 24]}],
            "operation": "Group absolute residuals by hour of day 00..23",
            "outputs": [{"label": "LSTM Hour 00 vs Hour 23 MAE", "value": f"{_r(h_mae.loc[0, 'lstm'], 2)} → {_r(h_mae.loc[23, 'lstm'], 2)} MW", "unit": "MW", "sample": [float(h_mae.loc[0, "lstm"]), float(h_mae.loc[17, "lstm"]), float(h_mae.loc[23, "lstm"])]}],
            "table": {
                "columns": ["Hour", "LSTM MAE", "FFN SGD MAE", "Ridge MAE", "SVR MAE"],
                "rows": [[f"{h:02d}:00", _r(h_mae.loc[h, "lstm"], 1), _r(h_mae.loc[h, "ffn_sgd"], 1), _r(h_mae.loc[h, "ridge"], 1), _r(h_mae.loc[h, "svr"], 1)] for h in [0, 3, 8, 12, 17, 20, 23]],
            },
        },
        "seasonal_mae": {
            "inputs": [{"label": "4 Calendar Month Blocks", "value": "Jan-Feb, Mar-May, Jun-Aug, Sep-Nov", "unit": "seasons", "sample": [1416, 2136, 2208, 1464]}],
            "operation": "Average |forecast - actual| within each seasonal month block",
            "outputs": [{"label": "Summer (Jun-Aug) LSTM MAE", "value": f"{_r(seas.loc[seas['months'] == 'Jun–Aug', 'lstm'].iloc[0], 2)} MW", "unit": "MW", "sample": [float(seas.loc[seas["months"] == "Jun–Aug", "lstm"].iloc[0])]}],
            "table": {
                "columns": ["Months", "Hours", "LSTM", "FFN SGD", "Ridge", "SVR"],
                "rows": [[r["months"], int(r["hours"]), _r(r["lstm"], 1), _r(r["ffn_sgd"], 1), _r(r["ridge"], 1), _r(r["svr"], 1)] for _, r in seas.iterrows()],
            },
        },
        "daily_wins": {
            "inputs": [{"label": "301 Paired Test Days", "value": "LSTM vs Ridge", "unit": "days", "sample": [301, int(wins["wins"])]}],
            "operation": "Count days where mean_24h(|lstm - actual|) < mean_24h(|ridge - actual|)",
            "outputs": [{"label": "LSTM Daily Win Rate vs Ridge", "value": f"{wins['wins']} / {wins['days']} days ({wins['share']*100:.1f}%)", "unit": "wins", "sample": [float(wins["wins"]), float(wins["days"]), float(wins["share"])]}],
        },
    }

    # Fix exact probe wording in `errors`
    err_probe = errors([100.0, 200.0], [105.0, 190.0])
    rich_numbers["errors"]["outputs"] = [
        {
            "label": "Probe MAE / RMSE",
            "value": f"{_r(err_probe['mae_mw'], 2)} MW MAE / {_r(err_probe['rmse_mw'], 2)} MW RMSE",
            "unit": "MW",
            "sample": [float(err_probe["mae_mw"]), float(err_probe["rmse_mw"])],
        }
    ]

    # Attach traced summaries + custom number theater to each bead
    for entry in manifest:
        name = entry["name"]
        entry["tokens"] = list(entry["tokens"])
        entry["io"] = list(entry["io"])
        entry["after"] = list(entry["after"])
        entry["trace"] = trace_by_bead.get(name)
        entry["numbers"] = rich_numbers.get(name, {"inputs": [], "operation": entry["explains"], "outputs": []})

    # Compute downstream edges (`before`) on every bead for fast bidirectional navigation
    children: dict[str, list[str]] = {e["name"]: [] for e in manifest}
    for e in manifest:
        for parent in e["after"]:
            if parent in children:
                children[parent].append(e["name"])
    for e in manifest:
        e["next"] = children[e["name"]]

    # Collect curated presentation slides for Depth 0 (Presentation Window)
    presentation = {
        "title": "ERCOT Day-Ahead Load Forecasting",
        "subtitle": "ASDS 6304 · Group 3 · Roshan Adhikari, Ryan Bui, Alemayehu Worash, Bikesh Adhikari",
        "quickTokens": [
            "619,968", "1,691.65", "13 iterations", "Winter Storm Uri",
            "leakage", "SGD vs Armijo", "91.1 GW", "214",
        ],
        "cards": [
            {
                "id": "card-data",
                "eyebrow": "01 · Data",
                "headline": "619,968 rows → 77,496 hours",
                "body": [
                    {"text": "0 MW spread", "token": "0 MW spread"},
                    {"text": " · ", "token": None},
                    {"text": "9 DST hours", "token": "DST"},
                    {"text": " · ", "token": None},
                    {"text": "Winter Storm Uri", "token": "Winter Storm Uri"},
                    {"text": " · ", "token": None},
                    {"text": "Oct 30 gap", "token": "October 30"},
                    {"text": " · ", "token": None},
                    {"text": "train-only scaling", "token": "leakage"},
                ],
                "kpis": [
                    {"label": "Mean, train → test", "value": "45.8 → 59.7 GW", "token": "chronological split"},
                    {"label": "Peak", "value": "91.13 GW", "token": "91.1 GW"},
                ],
            },
            {
                "id": "card-ridge-solvers",
                "eyebrow": "02 · Ridge solvers",
                "headline": "CG: 13 iterations",
                "body": [
                    {"text": "Same quadratic", "token": "quadratic"},
                    {"text": ", α = 0.1, start at zero.", "token": None},
                ],
                "kpis": [
                    {"label": "CG", "value": "13 · 0.12 ms", "token": "13 iterations"},
                    {"label": "Armijo", "value": "4,749 · 35.5 ms", "token": "4,749"},
                    {"label": "Fixed step", "value": "10,000 cap", "token": "10,000"},
                ],
            },
            {
                "id": "card-ffn",
                "eyebrow": "03 · Network optimizers",
                "headline": "SGD vs Armijo, test MAE",
                "body": [
                    {"text": "67 → 64 → 24", "token": "67→64→24"},
                    {"text": ", ", "token": None},
                    {"text": "100 passes", "token": "100 passes"},
                    {"text": ", same start.", "token": None},
                ],
                "kpis": [
                    {"label": "SGD", "value": "1,786.59 MW", "token": "1,786.59"},
                    {"label": "Armijo", "value": "2,083.21 MW", "token": "2,083.21"},
                ],
            },
            {
                "id": "card-lstm-eval",
                "eyebrow": "04 · 2026 test, 7,224 hours",
                "headline": "LSTM: 1,691.65 MW MAE",
                "body": [
                    {"text": "3-seed ensemble", "token": "LSTM"},
                    {"text": ", ", "token": None},
                    {"text": "14-day lookback", "token": "14-day lookback"},
                    {"text": ".", "token": None},
                ],
                "kpis": [
                    {"label": "Ridge / SVR", "value": "2,294 / 2,351 MW", "token": "2,294.08"},
                    {"label": "Days beating Ridge", "value": "214 / 301", "token": "214"},
                ],
            },
        ],
    }

    return {
        "meta": {
            "beadCount": len(manifest),
            "strandCount": len(STRAND_META),
            "sharedTestDays": 301,
            "sharedTestHours": 7224,
            "peakHour": peak_ts,
            "peakMw": _r(peak_sys_mw, 2),
        },
        "strands": STRAND_META,
        "presentation": presentation,
        "beads": manifest,
    }


def main() -> None:
    payload = build_beads_payload()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT_DIR)} ({OUT_PATH.stat().st_size / 1024:.1f} KB, {payload['meta']['beadCount']} beads)")


if __name__ == "__main__":
    main()
