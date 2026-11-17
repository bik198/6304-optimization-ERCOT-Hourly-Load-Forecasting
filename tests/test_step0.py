"""Unit tests verifying Step 0 audit integrity, zero data leakage, and generated artifacts."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from src.config import (
    AUDIT_JSON_PATH,
    EXPECTED_TIMESTAMPS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_ZONES,
    FEATURES_NPZ_PATH,
    FIGURES_DIR,
    RAW_CSV_PATH,
    SYSTEM_CSV_PATH,
)
from src.data_pipeline import audit_and_deduplicate, split_and_prepare_features


class TestStep0Pipeline(unittest.TestCase):
    """Validate raw dataset audit, chronological split safety, and EDA artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system_data, cls.audit = audit_and_deduplicate(RAW_CSV_PATH)
        cls.bundle = split_and_prepare_features(cls.system_data)

    def test_audit_matches_proposal_profile(self) -> None:
        """Audit counts must exactly match Proposal Section 3.1."""
        self.assertEqual(self.audit["total_raw_rows"], EXPECTED_TOTAL_ROWS)
        self.assertEqual(self.audit["unique_timestamp_dst_pairs"], EXPECTED_TIMESTAMPS)
        self.assertEqual(tuple(sorted(self.audit["zones"].keys())), tuple(sorted(EXPECTED_ZONES)))
        for z in EXPECTED_ZONES:
            self.assertEqual(self.audit["zones"][z], EXPECTED_TIMESTAMPS)
        self.assertEqual(self.audit["sources_raw_rows"]["archive"], 613_432)
        self.assertEqual(self.audit["sources_raw_rows"]["api"], 6_536)
        self.assertEqual(self.audit["max_cross_zone_system_mw_spread"], 0.0)
        self.assertEqual(self.audit["dst_repeated_hours_count"], 9)
        self.assertEqual(self.audit["raw_missing_weather_counts"]["temp"], 67)
        self.assertEqual(self.audit["raw_missing_weather_counts"]["rhum"], 66)
        self.assertEqual(self.audit["raw_missing_weather_counts"]["wspd"], 7)

    def test_chronological_splits_strictly_ordered(self) -> None:
        """Train timestamps must strictly precede validation, which strictly precedes test."""
        ts = self.system_data["timestamp"]
        splits = self.system_data["split"]

        train_ts = [t for t, s in zip(ts, splits) if s == "train"]
        val_ts = [t for t, s in zip(ts, splits) if s == "val"]
        test_ts = [t for t, s in zip(ts, splits) if s == "test"]

        self.assertGreater(len(train_ts), 0)
        self.assertGreater(len(val_ts), 0)
        self.assertGreater(len(test_ts), 0)
        self.assertLess(max(train_ts), min(val_ts))
        self.assertLess(max(val_ts), min(test_ts))

        # Verify day-ahead sample splits are also strictly ordered
        d_splits = self.bundle["splits"]
        t_dates = self.bundle["target_dates"]
        self.assertLess(
            max(t_dates[d_splits == "train"]),
            min(t_dates[d_splits == "val"]),
        )
        self.assertLess(
            max(t_dates[d_splits == "val"]),
            min(t_dates[d_splits == "test"]),
        )

    def test_zero_future_leakage_in_features_and_scaler(self) -> None:
        """Mutating a future day's load must not alter earlier origin-day features or train scalers."""
        # Verify scaler_mean and scaler_std match ONLY training rows
        train_mask = self.bundle["splits"] == "train"
        expected_mean = np.mean(self.bundle["X_raw"][train_mask], axis=0)
        expected_std = np.std(self.bundle["X_raw"][train_mask], axis=0)
        expected_std = np.where(expected_std < 1e-8, 1.0, expected_std)

        np.testing.assert_allclose(self.bundle["scaler_mean"], expected_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(self.bundle["scaler_std"], expected_std, rtol=1e-12, atol=1e-12)

        # Mutate the entire year 2026 test period in a copy and verify train/val X_raw are unchanged
        mutated_sys = copy.deepcopy(self.system_data)
        for i, s in enumerate(mutated_sys["split"]):
            if s == "test":
                mutated_sys["system_total_mw"][i] += 25_000.0
        mutated_bundle = split_and_prepare_features(mutated_sys)

        non_test_mask = self.bundle["splits"] != "test"
        np.testing.assert_allclose(
            mutated_bundle["X_raw"][non_test_mask],
            self.bundle["X_raw"][non_test_mask],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            mutated_bundle["X_scaled"][non_test_mask],
            self.bundle["X_scaled"][non_test_mask],
            rtol=0.0,
            atol=0.0,
        )

        # Also check mutating a single target test day (index k) leaves its own X_raw[k] identical
        test_indices = np.where(self.bundle["splits"] == "test")[0]
        target_date_to_perturb = str(self.bundle["target_dates"][test_indices[10]])
        mutated_one = copy.deepcopy(self.system_data)
        for i, ts in enumerate(mutated_one["timestamp"]):
            if ts[:10] == target_date_to_perturb:
                mutated_one["system_total_mw"][i] += 10_000.0
        one_bundle = split_and_prepare_features(mutated_one)
        idx_10 = test_indices[10]
        np.testing.assert_allclose(one_bundle["X_raw"][idx_10], self.bundle["X_raw"][idx_10], atol=0.0)
        self.assertGreater(np.mean(one_bundle["Y"][idx_10] - self.bundle["Y"][idx_10]), 9_999.0)

    def test_generated_step0_artifacts_exist_and_nonempty(self) -> None:
        """All exported CSV, NPZ, JSON, and PNG figure files must exist and be valid."""
        expected_files: list[Path] = [
            SYSTEM_CSV_PATH,
            FEATURES_NPZ_PATH,
            AUDIT_JSON_PATH,
            FIGURES_DIR / "fig1_weekly_avg_load.png",
            FIGURES_DIR / "fig2_hourly_load_profile.png",
            FIGURES_DIR / "fig3_monthly_load_profile.png",
            FIGURES_DIR / "fig4_load_distribution_and_uri.png",
        ]
        for path in expected_files:
            self.assertTrue(path.exists(), f"Missing artifact: {path}")
            self.assertGreater(path.stat().st_size, 1_000, f"Artifact unexpectedly small: {path}")

        with AUDIT_JSON_PATH.open("r", encoding="utf-8") as f:
            saved_audit = json.load(f)
        self.assertEqual(saved_audit["unique_timestamp_dst_pairs"], EXPECTED_TIMESTAMPS)


if __name__ == "__main__":
    unittest.main()
