# ERCOT forecasting optimization workflow

**Objective** — Predict ERCOT system demand (`system_total_mw`) for each of the next 24 hours. Compare numerical optimizers under controlled conditions, then test whether hyperparameter tuning improves the existing LSTM forecast. Battery dispatch belongs to a separate part of the group proposal.

**Current handoff** — All three experiments are complete. The original LSTM remains the main forecast. The tables and plots linked below are ready for the project report. Keep the notebooks, saved models, and forecasts in `preliminary/` as the reference runs.

## 0. Lock the evaluation protocol

- [x] Use 2018–2023 for fitting models and preprocessing, 2024–2025 for choosing settings and checkpoints, and January 1–November 4, 2026 for final evaluation after settings are frozen. The preliminary 2026 results had already been viewed before these experiments.
- [x] Use historical load available by the end of the previous day and calendar features known in advance. Fit scaling on training data. Exclude observed target-day weather from the main experiments.
- [x] Join forecast exports by target date and hour. Check for duplicate timestamps, require 24 hourly forecasts per scored day, and verify that actual loads agree. The [ML export](../preliminary/ercot_ml_24hour_forecasts.csv) and [LSTM export](../preliminary/lstm_outputs/lstm_day_ahead_forecasts.csv) share 301 test days and 7,224 hours. The baseline table, the selected LSTM, and both feed-forward runs use those hours. The model notebooks handle daylight-saving days and the October 30 gap differently.
- [x] Reproduce the reference MAEs on the shared hours. They are 2,393 MW for previous-day naive, 2,294 MW for Ridge, 2,351 MW for RBF-SVR, and 1,692 MW for the original LSTM ensemble. The LSTM's RMSE is 2,502 MW on the same hours. See [shared baseline metrics](results/ridge/shared_baselines.csv).

## 1. Compare three solvers on Ridge

Source the 12 hourly features and preprocessing from the [Ridge and SVR notebook](../preliminary/ercot_ridge_and_svr.ipynb). Reuse its selected `alpha = 0.1` and training-only `StandardScaler`.

- [x] Center the scaled training features and target using their training means to handle the unpenalized intercept. In float64, minimize `0.5 * (||Xw - y||² + alpha * ||w||²) / n`. Recover the intercept from the training means and fit scikit-learn Ridge on the same scaled features as the reference.
- [x] Start all three solvers from zero coefficients. Implement fixed-step gradient descent with step `1/L`, where `L` is the largest eigenvalue of the Hessian. Implement Armijo backtracking with initial step `1.0`, shrink factor `0.5`, and sufficient-decrease constant `1e-4`. Implement linear conjugate gradient on the Ridge normal equations using the same scaled objective.
- [x] Stop each solver when the gradient norm is at most `1e-8` times its starting value or after 10,000 iterations. Report when a solver reaches the cap. Record objective gap relative to the reference, gradient norm, iteration count, line-search objective evaluations, and wall time. Take the median of five solver runs, including solver-specific setup such as computing `L`. Exclude data loading and plotting.
- [x] Compare reference and solver predictions on validation. Check that converged predictions agree within `0.1 MW` maximum absolute difference, and report validation MAE and RMSE. Plot objective gap against iterations and elapsed time. Record any solver that misses the convergence or prediction agreement check.

Open [ridge_solvers.ipynb](ridge_solvers.ipynb) and run the cells in order. Conjugate gradient reached the tolerance in 13 iterations. Armijo gradient descent reached it in 4,749 iterations. Fixed-step gradient descent reached the 10,000-iteration cap and differed from the reference by up to 0.168 MW on validation. The scikit-learn fit reproduced the preliminary test forecasts within `3e-10 MW`. See the [solver summary](results/ridge/solver_summary.csv) and [convergence plot](results/ridge/convergence.png).

## 2. Tune the existing LSTM

Use the [LSTM notebook](../preliminary/ercot_lstm_forecast.ipynb) and its section 12 training hook. The original setting is a 14-day lookback, 64 hidden units, learning rate `0.03`, and mini-batch Nesterov SGD. Preserve the original day-step features, load normalization, Uri training exclusion, and checkpoint selection by validation loss. Start from the notebook's training definitions and direct trial logs and outputs into `optimization/`.

- [x] Screen the original setting with seed `42` and run the ten candidate triples below. Each triple gives `(lookback_days, hidden_units, learning_rate)`. Keep all other settings fixed.

  ```text
  ( 7,  64, 0.03)   (28,  64, 0.03)
  (14,  32, 0.03)   (14, 128, 0.03)
  (14,  64, 0.01)   (14,  64, 0.05)
  ( 7,  32, 0.05)   ( 7, 128, 0.01)
  (28,  32, 0.01)   (28, 128, 0.05)
  ```

- [x] Rank the candidates by 2024–2025 validation MAE in MW. Evaluate the top two candidates and the original with seeds `42`, `7`, and `2026`. Average the hourly predictions across the three seeds before computing each configuration's ensemble MAE and RMSE. Keep per-seed metrics to show variability and record training time and best epoch.
- [x] Freeze the selected settings and seeds before evaluating 2026. Retain the original if it ranks first on validation, and score only the frozen selection on the 301 shared test days. Report MAE, RMSE, and MAE by forecast hour.

Open [lstm_tuning.ipynb](lstm_tuning.ipynb) and run the cells in order. The notebook reuses completed trial checkpoints under `results/lstm/` when the experiment settings match. The 7-day, 64-unit candidate improved seed 42's validation MAE by 20.85 MW. Across three seeds, the original ensemble ranked first at **1,621.87 MW** validation MAE, ahead of both the 7-day, 64-unit setting at **1,622.95 MW** and the 7-day, 32-unit setting at **1,636.35 MW**. The original 14-day, 64-unit configuration with learning rate `0.03` was frozen for 2026. It reproduced the saved test results on 301 shared days at **1,691.65 MW MAE** and **2,501.63 MW RMSE**.

See the [screening results](results/lstm/screening.csv), [per-seed results](results/lstm/validation_trials.csv), [validation ensemble table](results/lstm/validation_ensembles.csv), [frozen selection](results/lstm/selection.json), [test summary](results/lstm/test_summary.csv), and [test MAE by forecast hour](results/lstm/test_mae_by_hour.csv).

The validation-only batch-size sweep kept the rest of the original LSTM configuration fixed and used the same three seeds. Ensemble validation MAE was **1,621.87 MW** for batch size 32, **1,632.57 MW** for batch size 64, and **1,647.83 MW** for batch size 16. The original batch size remained the best. See the [batch-size summary](results/lstm/batch_size/validation_summary.csv) and [per-seed runs](results/lstm/batch_size/validation_trials.csv).

## 3. Compare training methods on a feed-forward network

Use the [Step 0 feature builder](../src/data_pipeline.py) for 67 causal inputs and 24 hourly targets per day. Reuse `data/ercot_step0_features.npz` if available. If needed, call `build_feature_dict_from_raw` on `../proposal/ercot_model_table.csv` from the repository root to build the arrays for this experiment. Use `X_scaled`, `Y`, `splits`, and `target_dates`. Scale targets per hour using training statistics and convert predictions back to MW for evaluation.

- [x] Start with a `67 → 64 → 24` network with a ReLU hidden layer, mean squared error, and seed `42`. Copy the same initial weights into both runs. Use the same dataset, objective, device, and validation checkpoint rule.
- [x] Train one copy with full-batch gradient descent and Armijo backtracking, starting each iteration with step `1.0`, shrink factor `0.5`, and sufficient-decrease constant `1e-4`. Train the other with mini-batch SGD, batch size `32`, and learning rate selected from `0.001`, `0.01`, or `0.03` using validation. Run 100 passes over the training data and retain the checkpoint with the lowest validation MAE. Record the SGD selection run costs.
- [x] Plot full-training loss and validation MAE versus optimizer time. Record gradient evaluations, line-search objective evaluations, and run time. Report test MAE and RMSE on the 301 shared days.

Open [feedforward_optimizers.ipynb](feedforward_optimizers.ipynb) and run the cells in order. SGD at learning rate `0.03` reached **1,777.38 MW** validation MAE, compared with **2,033.09 MW** for Armijo GD. On 2026, SGD reached **1,786.59 MW MAE** and **2,625.83 MW RMSE**. Armijo reached **2,083.21 MW MAE** and **2,934.96 MW RMSE**. Armijo completed 100 full-batch gradient evaluations and 185 line-search trials in about **0.42 s** of optimizer time. The selected SGD run completed 6,900 mini-batch gradient evaluations in about **2.61 s**. These are single-seed, single-CPU timings under the same 100-pass limit. The three SGD rate trials together took about **8.39 s**, including monitoring.

See the [comparison table](results/feedforward/comparison.csv), [validation trials](results/feedforward/validation_trials.csv), and [training comparison plot](results/feedforward/training_comparison.png).

The validation robustness check kept the selected SGD learning rate `0.03` fixed and used matched starting weights for each of seeds `42`, `7`, and `2026`. Mini-batch SGD had lower validation MAE on every seed. Mean ± sample standard deviation were **1,781.12 ± 4.66 MW** for SGD and **2,015.17 ± 16.08 MW** for Armijo GD. The 2026 metrics above are from the original seed-42 comparison. See the [per-seed results](results/feedforward/robustness_trials.csv) and [summary](results/feedforward/robustness_summary.csv).

## 4. Report and handoff

- [x] Save experiment settings, seeds, metric logs, and plots under `optimization/`. Record the Python and library versions, hardware, and notebooks needed to reproduce each run.
- [x] Produce a Ridge convergence plot and solver summary, an LSTM tuning log and validation ensemble table, and a feed-forward training comparison plot and summary.
- [ ] In the final report, explain which solver reached the Ridge reference most efficiently, whether LSTM tuning improved 24-hour forecasts, and what the feed-forward experiment showed about the cost of backtracking versus mini-batch updates.
