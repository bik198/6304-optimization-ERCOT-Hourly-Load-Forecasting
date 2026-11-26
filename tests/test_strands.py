"""Regression tests: each strand module must reproduce the results the notebooks saved.

Reference values come from `optimization/results/`, `preliminary/` exports and the report.
Run with:
    python -m unittest tests.test_strands -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src import beads
from src import s10_evaluate as evaluate, s06_ffn as ffn, s05_solvers as solvers
from src.config import ROOT_DIR
from src.s01_system_data import daily_profiles, load_system_data
from src.s02_hourly_features import complete_days, hourly_examples
from src.s03_ridge import ridge_reference, ridge_search
from src.s04_svr import SELECTED, svr_fit, svr_sample
from src.s07_lstm_data import CONFIG, day_grid, lstm_calendar, lstm_tensors
from src.s08_lstm_model import SEEDS, lstm_ensemble, lstm_load_weights
from src.s09_metrics import errors

PRELIMINARY = ROOT_DIR / "preliminary"
RESULTS = ROOT_DIR / "optimization" / "results"


def _shared():
    if not hasattr(_shared, "cache"):
        system = load_system_data()
        profiles = daily_profiles(system)
        _shared.cache = {"profiles": profiles, "examples": hourly_examples(complete_days(profiles))}
    return _shared.cache


class TestHourlyStrands(unittest.TestCase):
    """Ridge, SVR and the three Ridge solvers."""

    @classmethod
    def setUpClass(cls):
        cls.ex = _shared()["examples"]
        cls.saved = pd.read_csv(PRELIMINARY / "ercot_ml_24hour_forecasts.csv", parse_dates=["timestamp"])
        cls.test = cls.ex.split("test")

    def test_split_sizes(self):
        self.assertEqual([int(self.ex.split(s).sum()) for s in ("train", "val", "test")], [51_552, 17_256, 7_224])
        np.testing.assert_array_equal(self.ex.timestamps[self.test].to_numpy(), self.saved["timestamp"].to_numpy())
        np.testing.assert_array_equal(self.ex.y[self.test], self.saved["actual_load_mw"].to_numpy())

    def test_ridge_search_matches_saved(self):
        model, results = ridge_search(self.ex)
        saved = pd.read_csv(PRELIMINARY / "ridge_validation_results.csv")
        np.testing.assert_allclose([r["mae_mw"] for r in results], saved["MAE_MW"], rtol=0, atol=1e-6)
        self.assertLess(np.abs(model.predict(self.ex.x[self.test]) - self.saved["ridge_forecast_mw"]).max(), 1e-6)

    def test_svr_selected_setting_matches_saved(self):
        model = svr_fit(*svr_sample(self.ex), **SELECTED)
        self.assertLess(np.abs(model.predict(self.ex.x[self.test]) - self.saved["svr_forecast_mw"]).max(), 1e-6)

    def test_solvers_match_saved_summary(self):
        problem = ridge_reference(self.ex)
        q = solvers.ridge_quadratic(self.ex, problem)
        val = self.ex.split("val")
        reference_val = problem.reference.predict(problem.scaled[val])
        saved = pd.read_csv(RESULTS / "ridge" / "solver_summary.csv").set_index("method")
        for label, solve in solvers.SOLVERS.items():
            with self.subTest(solver=label):
                trace, converged = solve(q)
                weights = trace[-1][1]
                row = saved.loc[label]
                self.assertEqual(len(trace) - 1, row["iteration"])
                self.assertEqual(trace[-1][3], row["line_search_evaluations"])
                self.assertEqual(converged, bool(row["converged"]))
                diff = np.abs(q.predict(problem.scaled[val], weights) - reference_val).max()
                self.assertAlmostEqual(diff, row["max_validation_difference_mw"], delta=1e-6)
                self.assertAlmostEqual(q.gap(weights, problem.reference.coef_), row["objective_gap"],
                                       delta=1e-9 + 1e-6 * abs(row["objective_gap"]))


class TestFeedForward(unittest.TestCase):
    """Armijo GD vs SGD at seed 42 must repeat the saved validation trials."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.p = ffn.ffn_targets(ffn.ffn_bundle())
        cls.start = ffn.ffn_network(ffn.SEED)
        cls.saved = pd.read_csv(RESULTS / "feedforward" / "validation_trials.csv").set_index("method")

    def check(self, method, rate):
        run = ffn.ffn_train(self.p, method, rate, self.start)
        row = self.saved.loc[method]
        self.assertEqual(run["summary"]["best_pass"], row["best_pass"])
        self.assertEqual(run["summary"]["gradient_evaluations"], row["gradient_evaluations"])
        self.assertEqual(run["summary"]["line_search_evaluations"], row["line_search_evaluations"])
        self.assertAlmostEqual(run["summary"]["validation_mae_mw"], row["validation_mae_mw"], delta=1e-6)
        return run

    def test_armijo(self):
        run = self.check("Armijo GD", None)
        forecast = ffn.ffn_forecast(self.p, run["weights"])
        saved = pd.read_csv(RESULTS / "feedforward" / "test_forecasts.csv", parse_dates=["timestamp"])
        days = pd.DatetimeIndex(pd.to_datetime(self.p.dates[self.p.masks["test"]]))
        index = days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h")
        mine = pd.Series(forecast.ravel(), index=index).reindex(saved["timestamp"]).to_numpy()
        self.assertLess(np.abs(mine - saved["armijo_mw"]).max(), 1e-6)

    def test_sgd(self):
        self.check("SGD 0.03", 0.03)


class TestLSTM(unittest.TestCase):
    """Saved seed weights, loaded into the module network, must reproduce the saved forecasts."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)  # the notebook's thread count; float32 sums depend on it
        profiles = _shared()["profiles"]
        cls.grid = day_grid(profiles)
        cls.data = lstm_tensors(cls.grid, lstm_calendar(cls.grid), CONFIG)
        cls.models = [lstm_load_weights(cls.data, s) for s in SEEDS]
        cls.saved = pd.read_csv(PRELIMINARY / "lstm_outputs" / "lstm_day_ahead_forecasts.csv")

    def test_sample_counts(self):
        self.assertEqual({k: int(m.sum()) for k, m in self.data["masks"].items()},
                         {"train": 2_156, "val": 731, "test": 307})
        self.assertEqual(sum(p.numel() for p in self.models[0].parameters()), 70_424)

    def test_ensemble_matches_saved(self):
        for split, mae in (("val", 1_621.87), ("test", None)):
            with self.subTest(split=split):
                pred = lstm_ensemble(self.models, self.data, split)
                saved = self.saved[self.saved["split"] == split]
                self.assertLess(np.abs(pred.ravel() - saved["lstm_mw"].to_numpy()).max(), 1e-6)
                rows = self.data["S"]["rows"][self.data["masks"][split]]
                np.testing.assert_array_equal(self.grid.load[rows].ravel(), saved["actual_mw"].to_numpy())
                if mae is not None:
                    self.assertAlmostEqual(errors(self.grid.load[rows], pred)["mae_mw"], mae, delta=0.005)


class TestEvaluation(unittest.TestCase):
    """The report's comparison tables, recomputed from the shared cohort."""

    @classmethod
    def setUpClass(cls):
        cls.frame = evaluate.shared_cohort()

    def test_table8_scores(self):
        table = evaluate.score_table(self.frame).set_index("model")
        report = {"lstm": (1691.65, 2501.63), "ffn_sgd": (1786.59, 2625.83), "ffn_armijo": (2083.21, 2934.96),
                  "ridge": (2294.08, 3199.91), "svr": (2351.17, 3302.77), "previous_day": (2393.23, 3370.57),
                  "previous_week": (4606.45, 6339.62)}
        self.assertEqual(list(table.index), list(report))
        for model, (mae, rmse) in report.items():
            self.assertAlmostEqual(table.loc[model, "mae_mw"], mae, delta=0.005)
            self.assertAlmostEqual(table.loc[model, "rmse_mw"], rmse, delta=0.005)

    def test_table9_high_load(self):
        table = evaluate.high_load(self.frame).set_index("model")
        self.assertEqual(table["top_hours"].iloc[0], 723)
        self.assertEqual(str(table["peak_timestamp"].iloc[0]), "2026-07-22 17:00:00")
        report = {"lstm": (1948.36, -3299.60), "ffn_sgd": (2120.74, -3886.60), "ffn_armijo": (2061.52, -6149.79),
                  "ridge": (2511.88, -7965.37), "svr": (1971.95, -7051.50)}
        for model, (top, peak) in report.items():
            self.assertAlmostEqual(table.loc[model, "top_mae_mw"], top, delta=0.005)
            self.assertAlmostEqual(table.loc[model, "peak_error_mw"], peak, delta=0.005)

    def test_table10_and_hours(self):
        seasons = evaluate.seasonal_mae(self.frame).set_index("months")
        self.assertEqual(list(seasons["hours"]), [1416, 2136, 2208, 1464])
        self.assertAlmostEqual(seasons.loc["Jun–Aug", "lstm"], 1525.67, delta=0.005)
        self.assertAlmostEqual(seasons.loc["Mar–May", "svr"], 2851.23, delta=0.005)
        hours = evaluate.mae_by_hour(self.frame)
        self.assertAlmostEqual(hours.loc[0, "lstm"], 555.73, delta=0.005)
        self.assertAlmostEqual(hours.loc[23, "lstm"], 2143.70, delta=0.005)
        self.assertEqual(evaluate.daily_wins(self.frame, "lstm", "ridge")["wins"], 214)


class TestBeadRegistry(unittest.TestCase):

    def test_edges_point_to_registered_beads(self):
        for b in beads.REGISTRY.values():
            for upstream in b.after:
                self.assertIn(upstream, beads.REGISTRY, f"{b.name} -> {upstream}")

    def test_trace_records_calls(self):
        with beads.trace() as calls:
            errors([1.0, 2.0], [2.0, 4.0])
        self.assertEqual(calls[0]["bead"], "errors")
        self.assertEqual(calls[0]["output"], {"mae_mw": 1.5, "rmse_mw": np.sqrt(2.5)})


if __name__ == "__main__":
    unittest.main()
