"""Optimizer strand #2: full-batch Armijo GD vs mini-batch SGD on a 67 → 64 → 24 network.

Source logic: `optimization/feedforward_optimizers.ipynb`. Inputs are the Step 0 day-ahead
bundle: the cached `.npz` when present (as in the notebook), otherwise rebuilt from the saved
hourly table with the borrowed Step 0 feature bead. Targets are standardized per hour on
training days. Everything runs in float64 on one CPU thread.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .beads import bead
from .config import FEATURES_NPZ_PATH

PASSES, BATCH_SIZE, SEED = 100, 32, 42
SGD_RATES = (0.001, 0.01, 0.03)
ARMIJO = {"initial_step": 1.0, "shrink": 0.5, "sufficient_decrease": 1e-4, "max_search": 50}
_KEYS = ("X_scaled", "Y", "splits", "target_dates")


@bead("ffn_bundle", strand="ffn", after=("day_ahead_features",),
      title="Load Step 0 FFN Bundle",
      io=("ercot_step0_features.npz", "X(3221×67), Y(3221×24)"),
      tokens=("FFN", "67 features", "npz", "Step 0"),
      explains="Load the 67-input, 24-target day-ahead arrays that Step 0 built.")
def ffn_bundle() -> dict[str, np.ndarray]:
    if FEATURES_NPZ_PATH.is_file():
        with np.load(FEATURES_NPZ_PATH, allow_pickle=False) as cache:
            return {k: cache[k] for k in _KEYS}
    from .s01_system_data import day_ahead_features, load_system_data
    bundle = day_ahead_features(load_system_data())
    return {k: bundle[k] for k in _KEYS}


@dataclass
class FFNProblem:
    x: torch.Tensor
    y_mw: torch.Tensor
    y: torch.Tensor
    target_mean: torch.Tensor
    target_sd: torch.Tensor
    masks: dict[str, np.ndarray]
    dates: np.ndarray

    def part(self, split: str):
        m = self.masks[split]
        return self.x[m], self.y[m], self.y_mw[m]


@bead("ffn_targets", strand="ffn", after=("ffn_bundle",),
      title="Standardize FFN Targets",
      io=("Y in MW", "per-hour z-scored Y"),
      tokens=("target scaling", "float64", "FFN", "train stats"),
      explains="Standardize each of the 24 target hours with its training mean and spread.")
def ffn_targets(bundle: dict[str, np.ndarray]) -> FFNProblem:
    splits = bundle["splits"]
    masks = {s: splits == s for s in ("train", "val", "test")}
    x = torch.from_numpy(bundle["X_scaled"].astype(np.float64))
    y_mw = torch.from_numpy(bundle["Y"].astype(np.float64))
    mean = y_mw[masks["train"]].mean(dim=0)
    sd = y_mw[masks["train"]].std(dim=0, correction=0)
    return FFNProblem(x, y_mw, (y_mw - mean) / sd, mean, sd, masks, bundle["target_dates"])


@bead("ffn_network", strand="ffn",
      title="67 → 64 → 24 ReLU Network",
      io=("seed=42", "67 → 64 → 24 ReLU"),
      tokens=("feed-forward", "FFN", "ReLU", "neural network", "seed 42", "67→64→24"),
      explains="Create the 67 → 64 → 24 ReLU network from a fixed seed so both optimizers start identically.")
def ffn_network(seed: int = SEED) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(67, 64), nn.ReLU(), nn.Linear(64, 24)).double()


_loss = nn.MSELoss()


@bead("armijo_update", strand="ffn", after=("ffn_network", "ffn_targets"),
      title="Full-Batch Armijo FFN Step",
      io=("full batch grad", "1 step + line search"),
      tokens=("Armijo", "full-batch", "backtracking", "FFN", "2,083.21"),
      explains="One full-batch gradient step; halve the step until the Armijo sufficient-decrease test passes.")
def armijo_update(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor) -> tuple[float, int]:
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    loss = _loss(model(train_x), train_y)
    loss.backward()
    with torch.no_grad():
        start = parameters_to_vector(params)
        grad = parameters_to_vector([p.grad for p in params])
        squared_norm = grad.square().sum().item()
        base_loss = loss.item()
        step = ARMIJO["initial_step"]
        for evaluations in range(1, ARMIJO["max_search"] + 1):
            vector_to_parameters(start - step * grad, params)
            if _loss(model(train_x), train_y).item() <= base_loss - ARMIJO["sufficient_decrease"] * step * squared_norm:
                return step, evaluations
            step *= ARMIJO["shrink"]
        vector_to_parameters(start, params)
    raise RuntimeError("Armijo line search exhausted the trial steps.")


@bead("sgd_pass", strand="ffn", after=("ffn_network", "ffn_targets"),
      title="Mini-Batch SGD Epoch",
      io=("shuffled batches of 32", "69 SGD steps/pass"),
      tokens=("mini-batch", "SGD", "batch 32", "0.03", "1,786.59"),
      explains="One shuffled pass over the training days in mini-batches of 32, one SGD step per batch.")
def sgd_pass(model, optimizer, generator, train_x, train_y, batch_size: int = BATCH_SIZE) -> int:
    order = torch.randperm(len(train_x), generator=generator)
    for batch in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        _loss(model(train_x[batch]), train_y[batch]).backward()
        optimizer.step()
    return (len(order) + batch_size - 1) // batch_size


@torch.no_grad()
def _monitor(model, p: FFNProblem) -> dict[str, float]:
    train_x, train_y, _ = p.part("train")
    val_x, _, val_mw = p.part("val")
    train_loss = _loss(model(train_x), train_y).item()
    val_mae = (model(val_x) * p.target_sd + p.target_mean - val_mw).abs().mean().item()
    if not np.isfinite([train_loss, val_mae]).all():
        raise ValueError("Training produced a non-finite loss or validation error.")
    return {"training_mse": train_loss, "validation_mae_mw": val_mae}


@bead("ffn_train", strand="ffn", after=("armijo_update", "sgd_pass"),
      title="Train FFN for 100 Passes",
      io=("100 passes", "best val checkpoint"),
      tokens=("100 passes", "SGD vs Armijo", "1,777.38", "2,033.09", "6,900 grads"),
      explains="Run 100 passes with one optimizer and keep the weights with the lowest validation MAE.")
def ffn_train(p: FFNProblem, method: str, learning_rate: float | None, starting_model: nn.Module,
              shuffle_seed: int = SEED, passes: int = PASSES) -> dict:
    model = copy.deepcopy(starting_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) if learning_rate is not None else None
    generator = torch.Generator().manual_seed(shuffle_seed)
    train_x, train_y, _ = p.part("train")
    started = perf_counter()
    steps, gradients, search_evals, optimizer_seconds = [], 0, 0, 0.0
    best_mae, best_pass, best_state = float("inf"), None, None
    for epoch in range(passes + 1):
        step = None
        if epoch:
            update_start = perf_counter()
            if optimizer is None:
                step, evaluations = armijo_update(model, train_x, train_y)
                gradients += 1
                search_evals += evaluations
            else:
                gradients += sgd_pass(model, optimizer, generator, train_x, train_y)
            optimizer_seconds += perf_counter() - update_start
        diagnostics = _monitor(model, p)
        steps.append({"method": method, "pass": epoch, "optimizer_seconds": optimizer_seconds,
                      "gradient_evaluations": gradients, "line_search_evaluations": search_evals,
                      "armijo_step": step, **diagnostics})
        if diagnostics["validation_mae_mw"] < best_mae:
            best_mae, best_pass = diagnostics["validation_mae_mw"], epoch
            best_state = copy.deepcopy(model.state_dict())
    summary = {"method": method, "learning_rate": learning_rate, "best_pass": best_pass,
               "validation_mae_mw": best_mae, "optimizer_seconds": optimizer_seconds,
               "run_seconds": perf_counter() - started, "gradient_evaluations": gradients,
               "line_search_evaluations": search_evals}
    return {"summary": summary, "history": pd.DataFrame(steps), "weights": best_state}


@bead("ffn_forecast", strand="ffn", after=("ffn_train",),
      title="FFN Test Forecast",
      io=("best weights + test X", "306 × 24 MW forecast"),
      tokens=("FFN", "1,786.59", "2,083.21", "test forecast"),
      explains="Predict a split's 24-hour profiles and convert back to MW.")
@torch.no_grad()
def ffn_forecast(p: FFNProblem, weights: dict, split: str = "test") -> np.ndarray:
    model = ffn_network()
    model.load_state_dict(weights)
    return (model(p.x[p.masks[split]]) * p.target_sd + p.target_mean).numpy()
