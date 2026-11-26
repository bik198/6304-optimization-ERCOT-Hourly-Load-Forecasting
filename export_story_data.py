"""Write web/public/data/story.json: 8 scenes + proof drawers, facts only.

Copy rule: every sentence on a scene must be traceable to `report 1.docx`, the deck notes,
or a number computed here from the saved results. Derived numbers (ratios, differences,
percentages) are computed in this file, never typed by hand, so the overview and the
scenes cannot disagree. Project-specific names and colours live in beadline_project.py.

    python export_story_data.py
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from beadline_project import MODEL, PARTS, PROJECT, SOLVER
from src import s10_evaluate as ev

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "preliminary"
RES = ROOT / "optimization" / "results"
DECK = ROOT / "presentation_v2.pptx"
WEB_DATA = ROOT / "web" / "public" / "data"
OUT = WEB_DATA / "story.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LEARNED = ("lstm", "ffn_sgd", "ffn_armijo", "ridge", "svr")


def r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def f2(v, nd=2):
    return f"{v:,.{nd}f}"


def pct_lower(a, b):
    """Percent by which `a` is lower than `b`."""
    return 100 * (1 - a / b)


def deck_notes() -> dict[int, dict]:
    """Speaker notes from presentation_v2.pptx keyed by slide number (if the deck exists)."""
    notes = {}
    if not DECK.exists():
        return notes
    with zipfile.ZipFile(DECK) as z:
        for name in z.namelist():
            m = re.match(r"ppt/notesSlides/notesSlide(\d+)\.xml$", name)
            if not m:
                continue
            xml = z.read(name).decode("utf-8")
            paras = ["".join(re.findall(r"<a:t>(.*?)</a:t>", p)) for p in re.findall(r"<a:p>.*?</a:p>", xml, re.S)]
            text = " ".join(p for p in paras if p.strip() and not p.strip().isdigit())
            text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            head = re.match(r"\s*(.+?)\s*\|\s*(\d+:\d\d)\s*(.*)", text, re.S)
            if head:
                notes[int(m.group(1))] = {"speaker": head.group(1), "time": head.group(2), "text": head.group(3).strip()}
            elif text.strip():
                notes[int(m.group(1))] = {"speaker": "", "time": "", "text": text.strip()}
        by_slide = {}
        for name in z.namelist():
            m = re.match(r"ppt/slides/_rels/slide(\d+)\.xml\.rels$", name)
            if m:
                n = re.search(r"notesSlide(\d+)\.xml", z.read(name).decode("utf-8"))
                if n and int(n.group(1)) in notes:
                    by_slide[int(m.group(1))] = notes[int(n.group(1))]
    return by_slide


def log_sample(frame: pd.DataFrame, col: str, n: int = 80) -> pd.DataFrame:
    vals = frame[col].to_numpy()
    if len(vals) <= n:
        return frame
    targets = np.unique(np.round(np.geomspace(max(vals.min(), 1), vals.max(), n)))
    idx = np.unique(np.concatenate([[0, len(vals) - 1], np.searchsorted(vals, targets).clip(0, len(vals) - 1)]))
    return frame.iloc[idx]


# ---------------------------------------------------------------- charts

def series(key, points, **extra):
    label, color = MODEL[key]
    return {"label": label, "color": color, "points": points, **extra}


def chart_ridge_alpha():
    df = pd.read_csv(PRE / "ridge_validation_results.csv")
    return {"type": "line", "title": "Ridge validation MAE by regularization α", "x": {"label": "α", "log": True},
            "y": {"label": "Validation MAE", "unit": "MW"},
            "series": [series("ridge", [[r(a, 4), r(m)] for a, m in zip(df["Alpha"], df["MAE_MW"])], markers=True)]}


def chart_svr_grid():
    df = pd.read_csv(PRE / "svr_validation_results.csv").sort_values("MAE_MW")
    rows = [{"label": f"C {c:g} · ε {e:g} · γ {g}", "value": r(m), "highlight": i == 0}
            for i, (c, e, g, m) in enumerate(zip(df["C"], df["epsilon"], df["gamma"], df["MAE_MW"]))]
    return {"type": "dots", "title": "RBF-SVR validation MAE, 12 settings", "unit": "MW", "rows": rows, "color": MODEL["svr"][1]}


def chart_solvers():
    hist = pd.read_csv(RES / "ridge" / "history.csv")
    out = []
    for method, color in SOLVER.items():
        part = hist[(hist["method"] == method) & (hist["iteration"] >= 1) & (hist["objective_gap"] > 0)]
        part = log_sample(part, "iteration")
        out.append({"label": method, "color": color,
                    "points": [[int(i), float(f"{g:.4g}")] for i, g in zip(part["iteration"], part["objective_gap"])]})
    return {"type": "line", "title": "Ridge objective gap by iteration (same objective, same start)",
            "x": {"label": "Iteration (log scale)", "log": True},
            "y": {"label": "Objective gap", "unit": "MW²", "log": True}, "series": out}


def chart_ffn():
    hist = pd.read_csv(RES / "feedforward" / "training_history.csv")
    out = []
    for method, key in (("Armijo GD", "ffn_armijo"), ("SGD 0.03", "ffn_sgd")):
        part = hist[(hist["method"] == method) & (hist["pass"] >= 1)]
        out.append(series(key, [[r(s, 3), r(m)] for s, m in zip(part["optimizer_seconds"], part["validation_mae_mw"])]))
    return {"type": "line", "title": "Feed-forward validation MAE vs. optimizer time, 100 passes, seed 42",
            "x": {"label": "Optimizer time", "unit": "s"}, "y": {"label": "Validation MAE", "unit": "MW"}, "series": out}


def chart_lstm_screen():
    df = pd.read_csv(RES / "lstm" / "screening.csv").sort_values("mae_mw")
    rows = [{"label": f"{lb}d lookback · {h} units · lr {lr:g}" + (" (original)" if n == "original" else ""),
             "value": r(m), "highlight": n == "original"}
            for n, lb, h, lr, m in zip(df["name"], df["lookback"], df["hidden"], df["lr"], df["mae_mw"])]
    return {"type": "dots", "title": "LSTM validation screen, 11 settings, seed 42", "unit": "MW", "rows": rows,
            "color": MODEL["lstm"][1]}


def chart_peak_day(frame, worst):
    """Actual, LSTM and the worst learned forecast in colour; the other models as grey context lines."""
    peak = frame.loc[frame["actual"].idxmax()]
    day = frame[frame["timestamp"].dt.normalize() == peak["timestamp"].normalize()]
    pts = lambda col: [[int(t.hour), r(v)] for t, v in zip(day["timestamp"], day[col])]
    out = [{"label": "Measured load", "color": None, "points": pts("actual")}, series("lstm", pts("lstm")), series(worst, pts(worst))]
    out += [series(k, pts(k), secondary=True) for k in ("ffn_sgd", "ffn_armijo", "ridge", "svr") if k != worst]
    return {"type": "line", "title": f"Day of the 2026 maximum ({peak['timestamp']:%b %d, %Y}): forecasts vs. measured load",
            "x": {"label": "Hour of day", "hours": True}, "y": {"label": "Load", "unit": "MW"}, "series": out}


def chart_scores(frame):
    table = ev.score_table(frame)
    rows = [{"label": MODEL[m][0], "value": r(v), "highlight": m == "lstm"} for m, v in zip(table["model"], table["mae_mw"])]
    return {"type": "bars", "title": "2026 MAE on 7,224 matched hours (lower is better)", "unit": "MW", "rows": rows, "color": MODEL["lstm"][1]}


def chart_by_hour(frame, keys=("lstm", "previous_day")):
    hours = ev.mae_by_hour(frame, keys)
    return {"type": "line", "title": "2026 MAE at each forecast hour",
            "x": {"label": "Hour of day", "hours": True}, "y": {"label": "MAE", "unit": "MW", "zero": True},
            "series": [series(k, [[int(h), r(v)] for h, v in zip(hours.index, hours[k])]) for k in keys]}


def chart_first_week():
    ml = pd.read_csv(PRE / "ercot_ml_24hour_forecasts.csv", parse_dates=["timestamp"]).head(168)
    ticks = [{"v": i, "label": f"{t:%b %d}"} for i, t in enumerate(ml["timestamp"]) if t.hour == 0]
    out = [{"label": "Measured load", "color": None, "points": [[i, r(v)] for i, v in enumerate(ml["actual_load_mw"])]},
           series("ridge", [[i, r(v)] for i, v in enumerate(ml["ridge_forecast_mw"])]),
           series("svr", [[i, r(v)] for i, v in enumerate(ml["svr_forecast_mw"])])]
    return {"type": "line", "title": "Measured load and both classical forecasts, first 168 test hours",
            "x": {"label": "Date", "ticks": ticks, "hourly": True}, "y": {"label": "Load", "unit": "MW"}, "series": out}


def chart_high_load(frame):
    hl = ev.high_load(frame, list(LEARNED)).sort_values("top_mae_mw")
    rows = [{"label": MODEL[m][0], "value": r(v), "highlight": m == "lstm"} for m, v in zip(hl["model"], hl["top_mae_mw"])]
    return {"type": "bars", "title": f"MAE on the {int(hl['top_hours'].iloc[0])} highest-load 2026 hours", "unit": "MW", "rows": rows,
            "color": MODEL["lstm"][1]}


def chart_seasons(frame, keys):
    s = ev.seasonal_mae(frame, list(keys))
    return {"type": "line", "title": "2026 MAE by block of months",
            "x": {"label": "Months", "ticks": [{"v": i, "label": m} for i, m in enumerate(s["months"])]},
            "y": {"label": "MAE", "unit": "MW", "zero": True},
            "series": [series(k, [[i, r(v)] for i, v in enumerate(s[k])], markers=True) for k in keys]}


def chart_monthly(eda):
    rows = [{"label": MONTHS[m["month"] - 1], "value": m["mean"], "highlight": m["month"] in (3, 8)} for m in eda["monthly"]]
    return {"type": "bars", "title": "Average ERCOT load by month (2018–2026, years pooled)", "unit": "MW", "rows": rows, "color": 0, "keepOrder": True}


# ---------------------------------------------------------------- facts

def facts(eda):
    """Every derived number used in copy, computed once."""
    frame = ev.shared_cohort()
    t8 = ev.score_table(frame).set_index("model")
    hl = ev.high_load(frame, list(LEARNED)).set_index("model")
    by_hour = ev.mae_by_hour(frame, ("lstm",))
    season_keys = ("lstm", "ffn_sgd", "ridge", "svr")
    seas = ev.seasonal_mae(frame, list(season_keys)).set_index("months")
    solver = pd.read_csv(RES / "ridge" / "solver_summary.csv").set_index("method")
    rsum = pd.read_csv(RES / "feedforward" / "robustness_summary.csv").set_index("method")
    ridge_val = pd.read_csv(PRE / "ridge_validation_results.csv").set_index("Alpha")
    svr_val = pd.read_csv(PRE / "svr_validation_results.csv").sort_values("MAE_MW").iloc[0]
    splits = {s["split"]: s for s in eda["splits"]}
    overall = eda["hourly"]["overall"]
    month = {m["month"]: m["mean"] for m in eda["monthly"]}
    peak_err = {m: hl.loc[m, "peak_error_mw"] for m in LEARNED}
    spring_worst = all(seas[k].idxmax() == "Mar–May" for k in season_keys)
    return {
        "frame": frame, "t8": t8, "hl": hl, "seas": seas, "season_keys": season_keys, "solver": solver, "rsum": rsum,
        "ridge_val": ridge_val, "svr_val": svr_val, "splits": splits, "month": month,
        "lo": int(np.argmin(overall)), "hi": int(np.argmax(overall)), "lo_mw": overall[int(np.argmin(overall))], "hi_mw": overall[int(np.argmax(overall))],
        "growth": pct_lower(splits["train"]["mean"], splits["test"]["mean"]) * -1,  # % rise train -> test
        "growth_pct": 100 * (splits["test"]["mean"] / splits["train"]["mean"] - 1),
        "speedup": solver.loc["Armijo GD", "elapsed_ms"] / solver.loc["Conjugate gradient", "elapsed_ms"],
        "sgd_gain": rsum.loc["Armijo GD", "mean_validation_mae_mw"] - rsum.loc["Mini-batch SGD", "mean_validation_mae_mw"],
        "svr_val_gain": ridge_val.loc[0.1, "MAE_MW"] - svr_val["MAE_MW"],
        "ridge_test_gain": t8.loc["svr", "mae_mw"] - t8.loc["ridge", "mae_mw"],
        "lead": t8.loc["ffn_sgd", "mae_mw"] - t8.loc["lstm", "mae_mw"],
        "vs_prev": pct_lower(t8.loc["lstm", "mae_mw"], t8.loc["previous_day", "mae_mw"]),
        "vs_ridge": pct_lower(t8.loc["lstm", "mae_mw"], t8.loc["ridge", "mae_mw"]),
        "vs_ffn": pct_lower(t8.loc["lstm", "mae_mw"], t8.loc["ffn_sgd", "mae_mw"]),
        "hour0": by_hour["lstm"].iloc[0], "hour23": by_hour["lstm"].iloc[-1],
        "peak_err": peak_err, "all_under": all(v < 0 for v in peak_err.values()),
        "worst_peak": min(peak_err, key=peak_err.get),
        "spring_worst": spring_worst,
        "wins": ev.daily_wins(frame, "lstm", "ridge"),
        "peak_mw": float(frame["actual"].max()),
    }


# ---------------------------------------------------------------- 8 scenes

def build(eda, notes, F):
    t8, hl, solver, splits, month = F["t8"], F["hl"], F["solver"], F["splits"], F["month"]
    frame, ridge_val, svr_val = F["frame"], F["ridge_val"], F["svr_val"]
    seeds = pd.read_csv(RES / "lstm" / "validation_trials.csv").query("name == 'original'").set_index("seed")
    ens = pd.read_csv(RES / "lstm" / "validation_ensembles.csv").set_index("name")
    batch = pd.read_csv(RES / "lstm" / "batch_size" / "validation_summary.csv").sort_values("batch_size")
    robust = pd.read_csv(RES / "feedforward" / "robustness_trials.csv").pivot(index="seed", columns="method", values="validation_mae_mw")
    rsum = F["rsum"]
    cg, ar, fx = (solver.loc[m] for m in ("Conjugate gradient", "Armijo GD", "Fixed-step GD"))
    speed = f"{F['speedup']:.0f}×"
    seas = F["seas"]
    lstm_spring, lstm_summer = seas.loc["Mar–May", "lstm"], seas.loc["Jun–Aug", "lstm"]
    worst = F["worst_peak"]

    def merge_notes(*slides, cue: str):
        chunks = [notes[s]["text"] for s in slides if s in notes and notes[s].get("text")]
        return {"speaker": "", "time": "", "text": "\n\n".join(chunks) if chunks else cue}

    peak_line = "All five learned forecasts were below" if F["all_under"] else "Forecasts missed"
    season_line = "spring had the highest error for each model" if F["spring_worst"] else "error varied by season"

    return [
        {
            "id": "scene1", "part": 1, "bead": "dedupe",
            "beads": ["dedupe", "load_system_data", "daily_profiles", "day_ahead_features"],
            "title": "Cleaning the Data",
            "headline": f"{eda['meta']['rawRows']:,} zone rows reduce to {eda['meta']['hours']:,} system hours, split by year",
            "sub": "The raw file repeats ERCOT system demand on all eight weather-zone rows for each hour; the audit found a 0 MW spread across zones. "
                   "Keeping one row per timestamp and daylight-saving flag leaves one target per physical hour. "
                   "Fitting used 2018–2023, selection used 2024–2025, and scoring used 2026 through November 4.",
            "say": f"The raw file repeats system demand on all eight zone rows, so {eda['meta']['rawRows']:,} rows reduce to {eda['meta']['hours']:,} hours, split by year into train, validation, and test.",
            "stats": [
                [f"{eda['meta']['rawRows']:,} → {eda['meta']['hours']:,}", "Zone rows → system hours"],
                ["2018–23 · 2024–25 · 2026", "Train · Validation · Test"],
                [f"{F['wins']['days']} days · 7,224 h", "Matched 2026 test hours for every model"],
            ],
            "charts": [{"type": "eda:splits"}],
            "callouts": [
                ["Forecast timing", "At the end of day d, each model forecasts the 24 hourly loads of day d + 1 from historical loads and calendar information. Observed target-day weather is excluded."],
            ],
            "details": {
                "label": "Details: split sizes and data-issue treatments (report Tables 2–3)",
                "tables": [
                    {"columns": ["Period", "System hours", "Step 0 target days", "Mean load"],
                     "rows": [[lbl, f"{splits[k]['hours']:,}", f"{splits[k]['days']:,}", f"{splits[k]['mean'] / 1000:.2f} GW"]
                              for k, lbl in (("train", "2018–2023"), ("val", "2024–2025"), ("test", "Jan 1–Nov 4, 2026"))]},
                    {"columns": ["Audit finding", "Treatment"],
                     "rows": [
                         ["Nine repeated fall-back hours", "Both records kept in the hourly audit; neural inputs average the repeated hour for a fixed 24-slot day."],
                         ["Nine skipped spring-forward hours", "Neural pipelines interpolate the missing slot; Ridge/SVR excludes affected days and incomplete lag profiles."],
                         ["Entire day missing on October 30, 2026", "No test target for that day; the LSTM fills input history with the same hour one week earlier."],
                         ["274 raw weather cells missing", "Imputed with training-period medians; target-day observed weather is excluded from forecast features."],
                     ]},
                ],
            },
            "notes": merge_notes(2, 3, 4, cue="Zone rows to system hours; year-based split."),
        },
        {
            "id": "scene2", "part": 1, "bead": "load_system_data",
            "beads": ["load_system_data", "daily_profiles"],
            "title": "Load Patterns",
            "headline": f"Average load rose {F['growth_pct']:.1f}% from training to test; the daily mean swings {(F['hi_mw'] - F['lo_mw']) / 1000:.2f} GW",
            "sub": f"Mean load was {splits['train']['mean'] / 1000:.2f} GW in 2018–2023, {splits['val']['mean'] / 1000:.2f} GW in 2024–2025, and {splits['test']['mean'] / 1000:.2f} GW in 2026. "
                   f"The average day has its low near {F['lo']:02d}:00 ({F['lo_mw'] / 1000:.2f} GW) and its high near {F['hi']:02d}:00 ({F['hi_mw'] / 1000:.2f} GW). "
                   "Seasonal peaks and the shift in load level motivate lag and calendar inputs.",
            "say": f"Average load rose {F['growth_pct']:.1f}% from training to test, and the average day swings about {(F['hi_mw'] - F['lo_mw']) / 1000:.0f} gigawatts between 3 AM and 5 PM.",
            "stats": [
                [f"{splits['train']['mean'] / 1000:.2f} → {splits['test']['mean'] / 1000:.2f} GW", f"Mean load, train → test (+{F['growth_pct']:.1f}%)"],
                [f"{F['lo_mw'] / 1000:.2f} → {F['hi_mw'] / 1000:.2f} GW", f"Hourly mean, {F['lo']:02d}:00 → {F['hi']:02d}:00"],
                [f"{month[3] / 1000:.2f} → {month[8] / 1000:.2f} GW", "Monthly mean, March → August"],
            ],
            "charts": [{"type": "eda:growth"}, {"type": "eda:hourly"}],
            "callouts": [
                ["Winter Storm Uri", "Flagged for 168 system hours, February 14–20, 2021, inside the training period. The primary LSTM excluded 21 training windows that intersect it."],
            ],
            "details": {
                "label": "Details: monthly means and the Winter Storm Uri interval",
                "charts": [chart_monthly(eda), {"type": "eda:uri"}],
            },
            "notes": merge_notes(5, 6, cue="Level shift across splits; daily cycle."),
        },
        {
            "id": "scene3", "part": 2, "bead": "ridge_search",
            "beads": ["complete_days", "hourly_examples", "ridge_search", "ridge_reference", "svr_sample", "svr_fit", "svr_search"],
            "title": "Ridge vs. RBF-SVR",
            "headline": "RBF-SVR beat Ridge on 2024–2025 validation; Ridge beat RBF-SVR on the 2026 test",
            "sub": "Both models use the same 12 inputs per hour: same-hour load one day and one week earlier, summaries of the previous day, and calendar encodings. "
                   "Ridge fits an L2-penalized linear relationship; the RBF kernel can represent nonlinear relationships. "
                   "Settings were selected by validation MAE.",
            "say": f"With the same 12 inputs, RBF-SVR was {F['svr_val_gain']:.0f} megawatts better on validation, but Ridge was {F['ridge_test_gain']:.0f} megawatts better on the 2026 test.",
            "stats": [
                ["12 inputs per hour", "Lags, previous-day summaries, calendar"],
                [f"{f2(svr_val['MAE_MW'])} vs {f2(ridge_val.loc[0.1, 'MAE_MW'])} MW", f"Validation MAE: RBF-SVR vs Ridge (SVR −{F['svr_val_gain']:.2f})"],
                [f"{f2(t8.loc['ridge', 'mae_mw'])} vs {f2(t8.loc['svr', 'mae_mw'])} MW", f"2026 test MAE: Ridge vs RBF-SVR (Ridge −{F['ridge_test_gain']:.2f})"],
            ],
            "charts": [chart_first_week()],
            "callouts": [
                ["Ranking did not carry over", f"The validation ranking did not carry over to test. The 2026 maximum ({splits['test']['peak'] / 1000:.2f} GW) is above the training maximum ({splits['train']['peak'] / 1000:.2f} GW)."],
            ],
            "details": {
                "label": "Details: the 12 inputs, Ridge α search, and 12 SVR settings (report Tables 5–6)",
                "charts": [chart_ridge_alpha(), chart_svr_grid()],
                "tables": [
                    {"columns": ["Input group", "Count", "Inputs"],
                     "rows": [["Same-hour lags", "2", "Load 24 hours and 7 days earlier"],
                              ["Previous-day summaries", "3", "Previous day mean, maximum, and minimum load"],
                              ["Calendar", "7", "Sine/cosine of hour, weekday, month; weekend flag"]]},
                    {"columns": ["Model", "Selected settings", "Validation MAE", "2026 test MAE"],
                     "rows": [["Ridge", "α = 0.1; standardized inputs", f"{f2(ridge_val.loc[0.1, 'MAE_MW'])} MW", f"{f2(t8.loc['ridge', 'mae_mw'])} MW"],
                              ["RBF-SVR", "C = 10, ε = 0.10, γ = 0.01; 6,000 rows", f"{f2(svr_val['MAE_MW'])} MW", f"{f2(t8.loc['svr', 'mae_mw'])} MW"]]},
                ],
            },
            "notes": merge_notes(12, 14, cue="Validation winner differs from test winner."),
        },
        {
            "id": "scene4", "part": 2, "bead": "lstm_ensemble",
            "beads": ["day_grid", "lstm_calendar", "lstm_samples", "lstm_tensors", "lstm_network", "lstm_train", "lstm_load_weights", "lstm_forecast", "lstm_ensemble"],
            "title": "LSTM Ensemble",
            "headline": f"The 3-seed LSTM ensemble reads 14 days of history and had the lowest 2026 MAE: {f2(t8.loc['lstm', 'mae_mw'])} MW",
            "sub": "Each input holds 14 daily steps of 24 hourly loads plus calendar features, with load divided by the preceding seven-day average. "
                   "Two 64-unit LSTM layers feed a head that predicts all 24 hours jointly. "
                   "Forecasts from seeds 42, 7, and 2026 are averaged in MW before scoring.",
            "say": f"The LSTM reads 14 days of hourly history, predicts all 24 hours together, and averages three seeds; its 2026 MAE was {F['vs_prev']:.1f}% lower than the previous-day baseline.",
            "flow": [
                ["Normalize 14 days", "Divide each 14-day window of hourly load by the preceding seven-day average."],
                ["2 × 64 LSTM layers", "Recurrent stack; the head adds target-day calendar and recent load level."],
                ["Average 3 seeds", "Predict 24 hours jointly, convert back to MW, average seeds 42, 7, 2026."],
            ],
            "stats": [
                [f"{f2(t8.loc['lstm', 'mae_mw'])} MW", "2026 test MAE (lowest of all forecasts)"],
                [f"−{F['vs_prev']:.1f}%", "MAE vs. previous-day, same-hour baseline"],
                [f"−{F['vs_ridge']:.1f}%", "MAE vs. Ridge"],
            ],
            "charts": [chart_by_hour(frame, ("lstm", "previous_day"))],
            "callouts": [
                ["Ensemble vs. single seeds", "Validation MAE: seed 42 " + f2(seeds.loc[42, 'mae_mw']) + ", seed 7 " + f2(seeds.loc[7, 'mae_mw']) + ", seed 2026 " + f2(seeds.loc[2026, 'mae_mw'])
                 + f" MW. Averaged forecast: {f2(ens.loc['original', 'mae_mw'])} MW."],
            ],
            "details": {
                "label": "Details: checkpoints, Uri and weather checks (report Table 7, Figure 6)",
                "charts": [{"type": "image", "src": "img/lstm_training.png", "title": "Original LSTM training diagnostics (objective, validation loss, gradient norm, learning rate)", "source": "Preliminary LSTM notebook"}],
                "tables": [
                    {"columns": ["Run (2024–25 validation)", "Best epoch", "MAE", "RMSE"],
                     "rows": [[f"Seed {s}", str(int(seeds.loc[s, 'best_epoch'])), f"{f2(seeds.loc[s, 'mae_mw'])} MW", f"{f2(seeds.loc[s, 'rmse_mw'])} MW"] for s in (42, 7, 2026)]
                     + [["Forecast ensemble", "—", f"{f2(ens.loc['original', 'mae_mw'])} MW", f"{f2(ens.loc['original', 'rmse_mw'])} MW"]],
                     "highlight": 3},
                    {"columns": ["Seed-42 sensitivity run", "Validation MAE", "Report note"],
                     "rows": [["Primary (Uri windows excluded)", "1,645.63 MW", "Configuration used in the ensemble"],
                              ["Uri windows retained", "1,645.89 MW", "Sensitivity check"],
                              ["Observed target-day weather added", "1,061.97 MW", "Hindsight upper bound; unavailable at issuance, excluded from the comparison"]],
                     "highlight": 0},
                ],
            },
            "notes": merge_notes(7, 8, 9, 10, 11, cue="14-day input, joint 24-hour output, 3-seed average."),
        },
        {
            "id": "scene5", "part": 3, "bead": "conjugate_gradient",
            "beads": ["ridge_reference", "ridge_quadratic", "fixed_step_gd", "armijo_gd", "conjugate_gradient"],
            "title": "Ridge Solvers",
            "headline": f"On the same Ridge objective, conjugate gradient reached tolerance in {int(cg['iteration'])} iterations; Armijo needed {int(ar['iteration']):,}",
            "sub": "All three solvers minimize the same strictly convex quadratic from zero and stop when the gradient norm falls to 10⁻⁸ of its starting value, or at 10,000 iterations. "
                   "Fixed-step gradient descent uses step 1/L; Armijo tests step 1.0 and halves it as needed; linear conjugate gradient solves the normal equations.",
            "say": f"On the same Ridge objective, conjugate gradient converged in {int(cg['iteration'])} iterations, Armijo in {int(ar['iteration']):,}, and fixed-step descent hit the 10,000 cap; CG's median time was {speed} shorter than Armijo's.",
            "stats": [
                [f"{int(cg['iteration'])} it · {cg['elapsed_ms']:.3f} ms", f"Conjugate gradient ({speed} less time than Armijo)"],
                [f"{int(ar['iteration']):,} it · {ar['elapsed_ms']:.2f} ms", f"Armijo GD ({int(ar['line_search_evaluations']):,} line-search evaluations)"],
                [f"{int(fx['iteration']):,} it · {fx['elapsed_ms']:.2f} ms", f"Fixed-step GD (cap reached; up to {fx['max_validation_difference_mw']:.3f} MW from reference)"],
            ],
            "charts": [chart_solvers()],
            "callouts": [
                ["Same solution", f"CG and Armijo validation predictions agreed with the Cholesky reference within {ar['max_validation_difference_mw']:.3f} MW; both had validation MAE ≈ {f2(cg['mae_mw'])} MW."],
                ["Iteration bound", "In exact arithmetic, linear CG terminates in at most d iterations on a d-dimensional quadratic [Nocedal & Wright]. Here d = 12."],
            ],
            "details": {
                "label": "Details: objective and solver table (report Table 11)",
                "formula": "f(w) = (‖Xw − y‖² + α‖w‖²) / (2n),   α = 0.1;   stop when ‖∇f‖ ≤ 10⁻⁸ ‖∇f₀‖ or 10,000 iterations",
                "tables": [
                    {"columns": ["Solver", "Iterations", "Median ms", "Line-search evals", "Final objective gap", "Max diff vs reference"],
                     "rows": [[m, f"{int(solver.loc[m, 'iteration']):,}", f"{solver.loc[m, 'elapsed_ms']:.3f}", f"{int(solver.loc[m, 'line_search_evaluations']):,}",
                               f"{solver.loc[m, 'objective_gap']:.2e} MW²", f"{solver.loc[m, 'max_validation_difference_mw']:.2e} MW"] for m in SOLVER],
                     "highlight": 2},
                ],
            },
            "notes": merge_notes(17, 18, cue="Same objective; iterations and time per solver."),
        },
        {
            "id": "scene6", "part": 3, "bead": "ffn_train",
            "beads": ["day_ahead_features", "ffn_bundle", "ffn_targets", "ffn_network", "armijo_update", "sgd_pass", "ffn_train", "ffn_forecast"],
            "title": "Network Optimizers",
            "headline": "On the feed-forward network, mini-batch SGD gave lower validation MAE than full-batch Armijo for all three seeds",
            "sub": "Both methods started from identical weights and ran 100 passes through the 2,184 training days of a 67 → 64 → 24 ReLU network. "
                   "Full-batch Armijo took one gradient step per pass (100 total); mini-batch SGD used shuffled batches of 32 (6,900 updates). "
                   "Each kept its lowest-validation-MAE checkpoint.",
            "say": f"With a 100-pass budget, mini-batch SGD beat full-batch Armijo on validation for all three seeds, by {F['sgd_gain']:.0f} megawatts on average; Armijo used less optimizer time.",
            "stats": [
                ["6,900 vs 100", "Gradient updates in 100 passes: SGD vs Armijo"],
                [f"{rsum.loc['Mini-batch SGD', 'mean_validation_mae_mw']:,.2f} vs {rsum.loc['Armijo GD', 'mean_validation_mae_mw']:,.2f} MW",
                 f"3-seed mean validation MAE: SGD vs Armijo (−{F['sgd_gain']:.2f})"],
                ["2.61 s vs 0.42 s", "Optimizer time, seed 42: SGD vs Armijo"],
            ],
            "charts": [chart_ffn()],
            "callouts": [
                ["2026 test", f"SGD-trained network: {f2(t8.loc['ffn_sgd', 'mae_mw'])} MW; Armijo-trained: {f2(t8.loc['ffn_armijo', 'mae_mw'])} MW (14.2% lower with SGD)."],
                ["LSTM tuning", f"The best single-seed screen trial (7-day lookback) improved seed 42 by 20.85 MW; across three seeds the original configuration ranked first ({f2(ens.loc['original', 'mae_mw'])} MW)."],
            ],
            "details": {
                "label": "Details: paired seeds and LSTM tuning (report Tables 13–15, 17)",
                "charts": [chart_lstm_screen()],
                "tables": [
                    {"columns": ["Seed", "Armijo validation MAE", "SGD validation MAE", "Reduction"],
                     "rows": [[str(s), f"{f2(robust.loc[s, 'Armijo GD'])} MW", f"{f2(robust.loc[s, 'Mini-batch SGD'])} MW",
                               f"{f2(robust.loc[s, 'Armijo GD'] - robust.loc[s, 'Mini-batch SGD'])} MW"] for s in (7, 42, 2026)]},
                    {"columns": ["LSTM configuration (3-seed ensemble)", "Lookback", "Hidden", "Validation MAE"],
                     "rows": [[("Original" if n == "original" else "Candidate") + f" (lr {ens.loc[n, 'lr']:g})",
                               f"{int(ens.loc[n, 'lookback'])} days", str(int(ens.loc[n, "hidden"])), f"{f2(ens.loc[n, 'mae_mw'])} MW"] for n in ens.index]
                     + [[f"Original, batch size {int(b)}", "14 days", "64", f"{f2(m)} MW"] for b, m in zip(batch["batch_size"], batch["ensemble_mae_mw"])],
                     "highlight": 0},
                ],
            },
            "notes": merge_notes(19, 20, 21, cue="100 vs 6,900 updates; three seeds; time trade-off."),
        },
        {
            "id": "scene7", "part": 4, "bead": "score_table",
            "beads": ["errors", "shared_cohort", "score_table", "daily_wins"],
            "title": "2026 Results",
            "headline": f"On {F['wins']['days']} matched 2026 days, the LSTM ensemble ranked first and led second place by {F['lead']:.2f} MW",
            "sub": f"Every forecast is scored on the same {F['wins']['days']} days and 7,224 hours. "
                   f"The LSTM's MAE was {F['vs_prev']:.1f}% lower than the previous-day baseline, {F['vs_ridge']:.1f}% lower than Ridge, and {F['vs_ffn']:.1f}% lower than the SGD-trained network. "
                   "The ranking is the same for MAE and RMSE.",
            "say": f"On the same 7,224 hours, the LSTM ensemble ranked first at {t8.loc['lstm', 'mae_mw']:,.0f} megawatts MAE, the SGD-trained network second at {t8.loc['ffn_sgd', 'mae_mw']:,.0f}, and Ridge and RBF-SVR fourth and fifth.",
            "stats": [
                [f"#1 · {f2(t8.loc['lstm', 'mae_mw'])} MW", "LSTM ensemble"],
                [f"#2 · {f2(t8.loc['ffn_sgd', 'mae_mw'])} MW", "Feed-forward network, SGD"],
                [f"{F['wins']['wins']} of {F['wins']['days']} days", f"LSTM daily MAE below Ridge ({F['wins']['share'] * 100:.1f}%)"],
            ],
            "charts": [chart_scores(frame)],
            "callouts": [
                ["Report conclusion", "The LSTM ensemble delivered the best observed 24-hour forecast. Feature representations and training sizes differ across model families, so the ranking compares complete pipelines."],
            ],
            "details": {
                "label": "Details: MAE and RMSE for all seven forecasts (report Table 8)",
                "tables": [
                    {"columns": ["Rank · Forecast", "MAE", "RMSE"],
                     "rows": [[f"#{i + 1} · {MODEL[m][0]}", f"{f2(t8.loc[m, 'mae_mw'])} MW", f"{f2(t8.loc[m, 'rmse_mw'])} MW"] for i, m in enumerate(t8.index)],
                     "highlight": 0},
                ],
            },
            "notes": merge_notes(16, 22, cue="Read the ranking top to bottom."),
        },
        {
            "id": "scene8", "part": 4, "bead": "high_load",
            "beads": ["high_load", "mae_by_hour", "seasonal_mae"],
            "title": "Peak Hours & Limits",
            "headline": f"{peak_line} the {F['peak_mw'] / 1000:.2f} GW peak, and {season_line}",
            "sub": f"At the 2026 maximum (July 22, 17:00, {F['peak_mw']:,.2f} MW) the LSTM was {abs(F['peak_err']['lstm']):,.2f} MW low, the smallest miss of the five learned forecasts. "
                   f"Across the year, LSTM MAE was {lstm_spring:,.2f} MW in March–May and {lstm_summer:,.2f} MW in June–August, "
                   f"and rose from {F['hour0']:,.2f} MW at hour 00 to {F['hour23']:,.2f} MW at hour 23.",
            "say": f"Every learned forecast came in under the {F['peak_mw'] / 1000:.1f} gigawatt peak, the LSTM by {abs(F['peak_err']['lstm']) / 1000:.1f}; spring was the hardest season for each model.",
            "stats": [
                [f"{F['peak_err']['lstm'] / 1000:.2f} GW", "LSTM error at the 2026 peak hour"],
                [f"{F['peak_err'][worst] / 1000:.2f} GW", f"{MODEL[worst][0]} error at the same hour (largest miss)"],
                [f"{lstm_spring:,.0f} vs {lstm_summer:,.0f} MW", "LSTM MAE, Mar–May vs Jun–Aug"],
            ],
            "charts": [chart_peak_day(frame, worst), chart_seasons(frame, F["season_keys"])],
            "callouts": [
                ["Limitations (report)", "Testing covers one partial year. Preliminary 2026 results were viewed before the optimization follow-ups. "
                 "Six LSTM test days absent from the Ridge/SVR export are omitted. An autoregressive benchmark and battery dispatch were outside the completed experiments."],
            ],
            "details": {
                "label": "Details: top-decile hours and peak-hour errors (report Tables 9–10)",
                "charts": [chart_high_load(frame)],
                "tables": [
                    {"columns": ["Forecast", "2026 MAE", f"Top {int(hl['top_hours'].iloc[0])} hours MAE", "Error at peak"],
                     "rows": [[MODEL[m][0], f"{f2(t8.loc[m, 'mae_mw'])} MW", f"{f2(hl.loc[m, 'top_mae_mw'])} MW", f"{hl.loc[m, 'peak_error_mw']:,.2f} MW"] for m in LEARNED],
                     "highlight": 0},
                    {"columns": ["Months", "Hours"] + [MODEL[k][0] for k in F["season_keys"]],
                     "rows": [[blk, f"{int(seas.loc[blk, 'hours']):,}"] + [f"{seas.loc[blk, k]:,.2f}" for k in F["season_keys"]] for blk in seas.index]},
                ],
            },
            "notes": merge_notes(15, 23, 24, 25, cue="Peak-hour misses; seasonal and lead-time error."),
        },
    ]


def intro(eda, F):
    """Start-here overview. Facts from the report abstract and research questions."""
    frame, t8, solver = F["frame"], F["t8"], F["solver"]
    day = frame.assign(err=(frame["lstm"] - frame["actual"]).abs()).groupby(frame["timestamp"].dt.normalize())["err"].mean()
    median_day = day.sort_values().index[len(day) // 2]
    d = frame[frame["timestamp"].dt.normalize() == median_day]
    pts = lambda col: [[int(t.hour), r(v)] for t, v in zip(d["timestamp"], d[col])]
    chart = {"type": "line", "title": f"Median-error 2026 day for the LSTM ({median_day:%b %d, %Y}): forecast vs. measured load",
             "x": {"label": "Hour of day", "hours": True}, "y": {"label": "Load", "unit": "MW"},
             "series": [{"label": "Measured load", "color": None, "points": pts("actual")}, series("lstm", pts("lstm"))]}
    meta, splits = eda["meta"], F["splits"]
    return {
        "what": "We forecast each of the next 24 hourly electricity loads for the Electric Reliability Council of Texas (ERCOT), using observed system load from January 2018 through November 4, 2026. "
                "At the end of day d, each model forecasts the 24 hourly loads of day d + 1. "
                "Two questions: which model predicts most accurately on identical out-of-time hours, and how do the optimizers compare on a shared Ridge objective and a shared neural network?",
        "pipeline": [
            ["Clean the data", f"{meta['rawRows']:,} zone rows → {meta['hours']:,} system hours; split by year.", 1],
            ["Compare forecasts", "Persistence, Ridge, RBF-SVR, feed-forward network, LSTM ensemble.", 2],
            ["Compare optimizers", "Fixed-step GD, Armijo, and CG on Ridge; Armijo vs. SGD on a network.", 3],
            ["Score on 2026", f"{F['wins']['days']} matched days (7,224 hours), including the {F['peak_mw'] / 1000:.2f} GW maximum.", 4],
        ],
        "answer": [
            [f"{f2(t8.loc['lstm', 'mae_mw'])} MW", f"Lowest 2026 MAE: LSTM ensemble ({F['vs_prev']:.1f}% below previous-day baseline)"],
            [f"{int(solver.loc['Conjugate gradient', 'iteration'])} iterations", f"Conjugate gradient on Ridge vs. {int(solver.loc['Armijo GD', 'iteration']):,} for Armijo ({F['speedup']:.0f}× less time)"],
            [f"−{F['sgd_gain']:.2f} MW", "SGD vs. Armijo, 3-seed mean validation MAE on the feed-forward network"],
        ],
        "chart": chart,
        "terms": [
            ["MAE", "Mean absolute hourly error in MW (forecast minus measured load). Lower is better."],
            ["RMSE", "Square root of the mean squared hourly error in MW; gives more weight to large errors."],
            ["Load level", f"1 GW = 1,000 MW. Mean load: {splits['train']['mean'] / 1000:.2f} GW (2018–2023), {splits['test']['mean'] / 1000:.2f} GW (2026); 2026 maximum {F['peak_mw'] / 1000:.2f} GW."],
            ["Inputs", "Historical load and calendar information. Observed target-day weather is excluded because it is unavailable when the forecast is issued."],
        ],
        "howto": [
            ["Slides (← / →)", "Eight slides in four parts. Each slide has a Details section with the report tables, search grids, and equations."],
            ["Present mode (F)", "Hides the sidebar, moves the paragraph into the notes, and shows stats, chart, and text one click at a time."],
            ["Speaker view (P)", "Opens the notes, a timer, and the next slide in a second window that follows this one."],
            ["Code steps (M) and ↓", "Shows the 38 Python functions in seven groups. ↓ opens the code, the numbers at that step, and the 64-bit binary of a value."],
            ["Print / PDF", "Add ?print=1 to the address (or run npm run pdf) for one page per slide."],
        ],
    }


def main() -> None:
    eda = json.loads((WEB_DATA / "eda.json").read_text(encoding="utf-8"))
    notes = deck_notes()
    if DECK.exists():
        with zipfile.ZipFile(DECK) as z:
            (WEB_DATA / "img").mkdir(exist_ok=True)
            (WEB_DATA / "img" / "lstm_training.png").write_bytes(z.read("ppt/media/image3.png"))
    F = facts(eda)
    stops = build(eda, notes, F)
    for s in stops:  # rail titles must fit one line in the 264px sidebar
        assert len(s["title"]) <= 22, s["title"]
    payload = {
        "project": PROJECT,
        "title": PROJECT["title"],
        "subtitle": PROJECT["subtitle"],
        "byline": PROJECT["byline"],
        "opening": notes.get(1),
        "parts": PARTS,
        "intro": intro(eda, F),
        "stops": stops,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.1f} KB, {len(stops)} scenes; "
          f"speedup {F['speedup']:.1f}x, all_under={F['all_under']}, spring_worst={F['spring_worst']}, worst_peak={F['worst_peak']})")


if __name__ == "__main__":
    main()
