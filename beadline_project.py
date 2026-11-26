"""One place for everything project-specific that the Beadline viewer displays.

The React viewer in web/ is generic: titles, part names and colours, model labels and
colours, strand labels and colours, and the default Machine bead all come from here via
the exporters (export_story_data.py -> story.json `project` + `parts`,
export_beads_data.py -> beads.json `strands`). A second project (e.g. the time series
final) copies web/ unchanged and writes its own version of this file plus exporters.
"""

from __future__ import annotations

PROJECT = {
    "brand": "Beadline",
    "docTitle": "Beadline · ERCOT Day-Ahead Load Forecasting",
    "title": "Day-Ahead ERCOT Load Forecasting",
    "subtitle": "Comparing forecast models and numerical optimizers",
    "byline": "ASDS 6304 · Group 3 · Fall 2026 · Roshan Adhikari, Ryan Bui, Alemayehu Worash, Bikesh Adhikari",
    "defaultBead": "conjugate_gradient",
    "source": "Group 3 report and saved results · optimization/results, preliminary",
}

# Story parts, in order. `color` drives every part-tinted surface in the viewer.
PARTS = [
    {"id": 1, "title": "The Data", "color": "#3987e5",
     "question": "What is forecast, from which records, and how are the years split?"},
    {"id": 2, "title": "Forecast Models", "color": "#199e70",
     "question": "How do Ridge, RBF-SVR, and the LSTM ensemble compare?"},
    {"id": 3, "title": "Optimizers", "color": "#c98500",
     "question": "On a fixed model, how do the solvers compare in iterations, time, and error?"},
    {"id": 4, "title": "2026 Results & Limits", "color": "#d55181",
     "question": "How do all forecasts score on the matched 2026 hours, and where are errors largest?"},
]

# Model key -> (display label, categorical colour index 0..7 into --s1..--s8, None = ink).
# Colour indices are chosen so models shown together never share a hue family.
MODEL = {
    "lstm": ("LSTM ensemble", 0),             # blue
    "ffn_sgd": ("Feed-forward · SGD", 1),     # orange
    "ridge": ("Ridge", 2),                    # teal
    "svr": ("RBF-SVR", 6),                    # violet
    "ffn_armijo": ("Feed-forward · Armijo", 4),  # pink
    "previous_day": ("Previous day, same hour", 5),
    "previous_week": ("Previous week, same hour", None),
}

# Ridge solver -> colour index (blue / amber / pink: clearly separable lines).
SOLVER = {"Fixed-step GD": 3, "Armijo GD": 4, "Conjugate gradient": 0}

# `pos` = [column, row] of each group on the Code Steps map (left to right = data flow).
STRAND_META = [
    {"id": "data", "pos": [0, 1.5], "label": "Data & Step 0", "color": "#569CD6", "summary": "Raw 8-zone audit, deduplication, DST rules, and chronological splits."},
    {"id": "ridge", "pos": [1, 1], "label": "Ridge & 12 Features", "color": "#3794FF", "summary": "Complete 24-hour days, 12 hourly features, alpha search, and Cholesky reference."},
    {"id": "svr", "pos": [2, 0], "label": "RBF-SVR", "color": "#C586C0", "summary": "6,000-row even subsample and 12-setting RBF kernel grid search."},
    {"id": "solvers", "pos": [2, 1.5], "label": "Ridge Solvers", "color": "#DCDCAA", "summary": "Fixed-step GD, Armijo backtracking, and conjugate gradient on the Ridge quadratic."},
    {"id": "ffn", "pos": [1, 2], "label": "Feed-forward Optimizers", "color": "#CE9178", "summary": "67 → 64 → 24 ReLU network: full-batch Armijo GD vs. mini-batch SGD."},
    {"id": "lstm", "pos": [1, 3], "label": "LSTM Ensemble", "color": "#73C991", "summary": "14-day windows, Uri exclusion, and 3-seed Nesterov SGD ensemble."},
    {"id": "evaluate", "pos": [3, 1.5], "label": "Shared Evaluation", "color": "#CCA700", "summary": "301 common 2026 days (7,224 hours) scoring all models."},
]
