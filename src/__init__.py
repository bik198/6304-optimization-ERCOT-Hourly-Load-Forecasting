"""ASDS 6304 ERCOT Hourly Load Forecasting and Battery Dispatch Optimization.

This package houses the shared configuration constants and the leakage-safe data
engineering pipeline used across all forecasting models, optimizer benchmarks,
and the CVXPY battery dispatch solver.
"""

# Public submodules exposed when importing `src`
__all__ = ["config", "data_pipeline"]
