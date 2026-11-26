"""LSTM strand, model half: network, optimizer, training loop, forecasts and the seed ensemble.

Source logic: `preliminary/ercot_lstm_forecast.ipynb` (cells 18-21). Saved weights from
`preliminary/lstm_outputs/lstm_seed*.pt` load directly into `LSTMForecaster`.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .beads import bead
from .config import ROOT_DIR
from .s07_lstm_data import CONFIG, DayGrid

SEEDS = (42, 7, 2026)
WEIGHTS_DIR = ROOT_DIR / "preliminary" / "lstm_outputs"


class LSTMForecaster(nn.Module):
    """Two stacked LSTM layers over past days; the last hidden state joins target-day features in a dense head."""

    def __init__(self, n_seq, n_static, hidden=64, layers=2, dropout=0.2, head_units=128, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, head_units), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_units, horizon))

    def forward(self, x_seq, x_static):
        _, (h, _) = self.lstm(x_seq)
        return self.head(torch.cat([h[-1], x_static], dim=1))


@bead("lstm_network", strand="lstm", after=("lstm_tensors",),
      title="2-Layer LSTM Architecture",
      io=("n_seq=29, n_static=11", "70,424-param 2-layer LSTM"),
      tokens=("LSTM", "64 hidden", "2 layers", "70,424", "dropout 0.2"),
      explains="Build the LSTM forecaster sized to the input tensors.")
def lstm_network(data: dict, cfg: dict = CONFIG) -> LSTMForecaster:
    return LSTMForecaster(data["n_seq"], data["n_static"], cfg["hidden"], cfg["layers"],
                          cfg["dropout"], cfg["head_units"])


def make_optimizer(cfg, params):
    name = cfg["optimizer"]
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.0, weight_decay=cfg["wd"])
    if name == "sgd_nesterov":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True, weight_decay=cfg["wd"])
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    raise ValueError(name)


@torch.no_grad()
def predict_std(model, seq, static, batch=512):
    model.eval()
    return torch.cat([model(seq[i:i + batch], static[i:i + batch]) for i in range(0, len(seq), batch)]).numpy()


@bead("lstm_train", strand="lstm", after=("lstm_network",),
      title="Nesterov SGD LSTM Training",
      io=("Nesterov SGD lr=0.03", "best early-stop weights"),
      tokens=("Nesterov", "SGD", "early stopping", "gradient clip", "LR schedule"),
      explains="Nesterov SGD on shuffled batches with gradient clipping, a plateau schedule and early stopping.")
def lstm_train(cfg: dict, data: dict, grid: DayGrid, seed: int = 42, verbose: bool = False):
    cfg = {**CONFIG, **cfg}
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va = data["tensors"]["train"], data["tensors"]["val"]
    model = lstm_network(data, cfg)
    opt = make_optimizer(cfg, model.parameters())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg["lr_factor"], patience=cfg["lr_patience"])
    loader = DataLoader(TensorDataset(*tr), batch_size=cfg["batch"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss_fn = nn.MSELoss()
    S, sc, va_rows = data["S"], data["scaler"], data["masks"]["val"]
    hist = {k: [] for k in ["epoch", "train_obj", "val_loss", "val_mae_mw", "grad_norm", "lr", "time"]}
    best, best_state, wait, t0 = np.inf, None, 0, time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        gnorms = []
        for xb, sb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            gnorms.append(float(nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])))
            opt.step()
        p_tr, p_va = predict_std(model, tr[0], tr[1]), predict_std(model, va[0], va[1])
        train_obj = float(((p_tr - tr[2].numpy()) ** 2).mean())
        val_loss = float(((p_va - va[2].numpy()) ** 2).mean())
        val_mae = np.abs(sc.to_mw(p_va, S["level"][va_rows]) - grid.load[S["rows"][va_rows]]).mean()
        sched.step(val_loss)
        for k, v in zip(hist, [epoch, train_obj, val_loss, val_mae, np.mean(gnorms),
                               opt.param_groups[0]["lr"], time.perf_counter() - t0]):
            hist[k].append(v)
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"ep {epoch:3d} train {train_obj:.4f} val {val_loss:.4f} valMAE {val_mae:7.0f} MW")
        if val_loss < best - 1e-5:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    hist = pd.DataFrame(hist)
    hist.attrs["best_epoch"] = int(hist.loc[hist.val_loss.idxmin(), "epoch"])
    hist.attrs["train_seconds"] = hist["time"].iloc[-1]
    return model, hist


@bead("lstm_load_weights", strand="lstm", after=("lstm_network",),
      title="Load Saved Seed Checkpoints",
      io=("lstm_seed*.pt", "restored LSTMForecaster"),
      tokens=("weights", "checkpoints", "seed 42", "seed 7", "seed 2026"),
      explains="Load the saved weights of one trained seed instead of retraining.")
def lstm_load_weights(data: dict, seed: int, cfg: dict = CONFIG, directory: Path = WEIGHTS_DIR) -> LSTMForecaster:
    model = lstm_network(data, cfg)
    model.load_state_dict(torch.load(directory / f"lstm_seed{seed}.pt", map_location="cpu", weights_only=True))
    return model


@bead("lstm_forecast", strand="lstm", after=("lstm_train", "lstm_load_weights"),
      title="Rescale LSTM Forecast to MW",
      io=("std pred × 7d level", "24h MW forecast"),
      tokens=("LSTM", "forecast", "MW", "unscale"),
      explains="Predict standardized relative load, then multiply back by each day's 7-day level to get MW.")
def lstm_forecast(model, data: dict, split: str) -> np.ndarray:
    seq, static, _ = data["tensors"][split]
    return data["scaler"].to_mw(predict_std(model, seq, static), data["S"]["level"][data["masks"][split]])


@bead("lstm_ensemble", strand="lstm", after=("lstm_forecast",),
      title="3-Seed LSTM Ensemble",
      io=("3 seeds (42, 7, 2026)", "1,691.65 MW test MAE"),
      tokens=("ensemble", "1,691.65", "2,501.63", "1,621.87", "best model", "LSTM", "3 seeds"),
      explains="Average the MW forecasts of seeds 42, 7 and 2026.")
def lstm_ensemble(models, data: dict, split: str) -> np.ndarray:
    return np.mean([lstm_forecast(m, data, split) for m in models], axis=0)
