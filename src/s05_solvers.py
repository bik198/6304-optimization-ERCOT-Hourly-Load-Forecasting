"""Optimizer strand #1: fixed-step GD, Armijo GD and conjugate gradient on the Ridge quadratic.

Source logic: `optimization/ridge_solvers.ipynb` (cells 6 and 8). The notebook's single
`solve()` is split into one bead per method with the same arithmetic, so iteration counts
and traces match. Each trace entry is (elapsed seconds, weights, gradient norm, line-search trials).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .beads import bead
from .s02_hourly_features import HourlyExamples
from .s03_ridge import ScaledProblem

RELATIVE_TOLERANCE = 1e-8
MAX_ITERATIONS = 10_000
ARMIJO_INITIAL_STEP, ARMIJO_SHRINK, ARMIJO_SUFFICIENT_DECREASE = 1.0, 0.5, 1e-4


@dataclass(frozen=True)
class Quadratic:
    hessian: np.ndarray
    rhs: np.ndarray
    mean_x: np.ndarray
    mean_y: float

    def gap(self, weights: np.ndarray, optimum: np.ndarray) -> float:
        """f(w) - f(w*), computed from the difference to avoid cancelling large objective values."""
        delta = weights - optimum
        return float(delta @ (self.hessian @ optimum - self.rhs) + 0.5 * delta @ self.hessian @ delta)

    def predict(self, scaled_x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return (scaled_x - self.mean_x) @ weights + self.mean_y


@bead("ridge_quadratic", strand="solvers", after=("ridge_reference",),
      title="Form Ridge Quadratic (H, b)",
      io=("scaled train (12)", "H(12×12), b(12)"),
      tokens=("quadratic", "Hessian", "convex", "Ridge", "f(w)"),
      explains="Center the scaled training data and write Ridge as f(w) = ½wᵀHw − bᵀw.")
def ridge_quadratic(ex: HourlyExamples, problem: ScaledProblem) -> Quadratic:
    train = ex.split("train")
    mean_x, mean_y = problem.scaled[train].mean(axis=0), ex.y[train].mean()
    centered_x, centered_y = problem.scaled[train] - mean_x, ex.y[train] - mean_y
    n, d = centered_x.shape
    alpha = problem.reference.alpha
    hessian = (centered_x.T @ centered_x + alpha * np.eye(d)) / n
    return Quadratic(hessian, (centered_x.T @ centered_y) / n, mean_x, float(mean_y))


def _start(q: Quadratic):
    weights = np.zeros_like(q.rhs)
    gradient = -q.rhs
    squared_norm = float(gradient @ gradient)
    return weights, gradient, squared_norm, RELATIVE_TOLERANCE ** 2 * squared_norm


@bead("fixed_step_gd", strand="solvers", after=("ridge_quadratic",),
      title="Fixed-Step Gradient Descent",
      io=("H, b, step=1/L", "10,000 iters (cap)"),
      tokens=("fixed step", "gradient descent", "1/L", "eigenvalue", "10,000", "0.168 MW"),
      explains="Step against the gradient by 1/L, where L is the largest Hessian eigenvalue.")
def fixed_step_gd(q: Quadratic):
    started = perf_counter()
    weights, gradient, squared_norm, tolerance_squared = _start(q)
    step = 1 / np.linalg.eigvalsh(q.hessian)[-1]
    trace = [(perf_counter() - started, weights, np.sqrt(squared_norm), 0)]
    while squared_norm > tolerance_squared and len(trace) <= MAX_ITERATIONS:
        weights = weights - step * gradient
        gradient = q.hessian @ weights - q.rhs
        squared_norm = float(gradient @ gradient)
        trace.append((perf_counter() - started, weights, np.sqrt(squared_norm), 0))
    return trace, squared_norm <= tolerance_squared


@bead("armijo_gd", strand="solvers", after=("ridge_quadratic",),
      title="Armijo Backtracking GD",
      io=("H, b, backtrack", "4,749 iters / 35.5 ms"),
      tokens=("Armijo", "backtracking", "line search", "4,749", "35.5 ms", "9,501"),
      explains="Try step 1, halve until the objective falls by at least 1e-4 · step · ‖∇f‖², then step.")
def armijo_gd(q: Quadratic):
    started = perf_counter()
    weights, gradient, squared_norm, tolerance_squared = _start(q)
    trials = 0
    trace = [(perf_counter() - started, weights, np.sqrt(squared_norm), trials)]
    while squared_norm > tolerance_squared and len(trace) <= MAX_ITERATIONS:
        step = ARMIJO_INITIAL_STEP
        curvature = float(gradient @ (q.hessian @ gradient))
        while True:  # exact objective change of a quadratic along -gradient
            trials += 1
            change = -step * squared_norm + 0.5 * step ** 2 * curvature
            if change <= -ARMIJO_SUFFICIENT_DECREASE * step * squared_norm:
                break
            step *= ARMIJO_SHRINK
        weights = weights - step * gradient
        gradient = q.hessian @ weights - q.rhs
        squared_norm = float(gradient @ gradient)
        trace.append((perf_counter() - started, weights, np.sqrt(squared_norm), trials))
    return trace, squared_norm <= tolerance_squared


@bead("conjugate_gradient", strand="solvers", after=("ridge_quadratic",),
      title="Linear Conjugate Gradient",
      io=("H, b, H-conjugate", "13 iters / 0.12 ms"),
      tokens=("conjugate gradient", "CG", "13 iterations", "0.12 ms", "orthogonal", "fastest solver"),
      explains="Move along directions that are conjugate under H, so each step never undoes an earlier one.")
def conjugate_gradient(q: Quadratic):
    started = perf_counter()
    weights, gradient, squared_norm, tolerance_squared = _start(q)
    direction = -gradient
    trace = [(perf_counter() - started, weights, np.sqrt(squared_norm), 0)]
    while squared_norm > tolerance_squared and len(trace) <= MAX_ITERATIONS:
        curvature = q.hessian @ direction
        step = squared_norm / (direction @ curvature)
        weights = weights + step * direction
        gradient = q.hessian @ weights - q.rhs
        new_squared_norm = float(gradient @ gradient)
        direction = -gradient + (new_squared_norm / squared_norm) * direction
        squared_norm = new_squared_norm
        trace.append((perf_counter() - started, weights, np.sqrt(squared_norm), 0))
    return trace, squared_norm <= tolerance_squared


SOLVERS = {"Fixed-step GD": fixed_step_gd, "Armijo GD": armijo_gd, "Conjugate gradient": conjugate_gradient}
