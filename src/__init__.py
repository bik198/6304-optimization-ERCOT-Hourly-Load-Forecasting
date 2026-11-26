"""ASDS 6304 ERCOT Hourly Load Forecasting and Battery Dispatch Optimization.

This package houses the shared configuration constants, the leakage-safe Step 0 data
engineering pipeline, the bead registry (`beads.py`), and the modular strand steps
(`s01_system_data` through `s10_evaluate`) used across forecasting, optimization,
and the 5-layer Beadline presentation frontend.
"""

# Public submodules exposed when importing `src`
__all__ = ["beads", "config", "data_pipeline"]
