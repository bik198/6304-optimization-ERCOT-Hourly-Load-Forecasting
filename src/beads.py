"""Bead registry: tags each pipeline step so the presentation layer can find and trace it.

A bead is one function that performs one transformation. The decorator records the
bead's name, strand, one-sentence purpose, upstream beads, semantic presentation tokens,
input/output shape summary, and source location (file, start line, end line). Calling
the function behaves as before. Inside a `trace()` block, each bead call also records
a compact summary of its inputs and output for the numbers layer of the presentation.

Borrowed functions (such as the Step 0 pipeline in `data_pipeline.py`) are registered
with `register()` so their source files stay unchanged.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

STRAND_MODULES: tuple[str, ...] = (
    "src.s01_system_data",
    "src.s02_hourly_features",
    "src.s03_ridge",
    "src.s04_svr",
    "src.s05_solvers",
    "src.s06_ffn",
    "src.s07_lstm_data",
    "src.s08_lstm_model",
    "src.s09_metrics",
    "src.s10_evaluate",
)


@dataclass(frozen=True)
class Bead:
    name: str
    strand: str
    explains: str
    after: tuple[str, ...]
    file: str
    line: int
    qualname: str
    title: str = ""
    tokens: tuple[str, ...] = ()
    io: tuple[str, str] = ("", "")
    end_line: int = 0


REGISTRY: dict[str, Bead] = {}
_FUNCS: dict[str, Callable] = {}
_trace: list[dict[str, Any]] | None = None


def _describe(fn: Callable) -> tuple[str, int, int, str]:
    """Return `(relative_file, start_line, end_line, qualname)` for an unwrapped function."""
    target = inspect.unwrap(fn)
    path = Path(inspect.getsourcefile(target)).resolve()
    try:
        file = path.relative_to(ROOT).as_posix()
    except ValueError:
        file = path.as_posix()
    lines, start_line = inspect.getsourcelines(target)
    end_line = start_line + len(lines) - 1
    return file, start_line, end_line, target.__qualname__


def source_of(name: str) -> str:
    """Return the exact Python source code lines of registered bead `name`."""
    target = inspect.unwrap(_FUNCS[name])
    return "".join(inspect.getsourcelines(target)[0]).rstrip("\n")


def register(
    name: str,
    fn: Callable,
    *,
    strand: str,
    explains: str,
    after: tuple[str, ...] = (),
    title: str = "",
    tokens: tuple[str, ...] = (),
    io: tuple[str, str] = ("", ""),
) -> Callable:
    """Register `fn` as bead `name` and return a wrapper that reports to an active trace."""
    if name in REGISTRY:
        raise ValueError(f"Bead {name!r} is already registered.")
    file, line, end_line, qualname = _describe(fn)
    display_title = title or name.replace("_", " ").title()
    REGISTRY[name] = Bead(
        name=name,
        strand=strand,
        explains=explains,
        after=tuple(after),
        file=file,
        line=line,
        qualname=qualname,
        title=display_title,
        tokens=tuple(tokens),
        io=(io[0], io[1]),
        end_line=end_line,
    )
    _FUNCS[name] = fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if _trace is not None:
            _trace.append({
                "bead": name,
                "inputs": summarize({"args": args, "kwargs": kwargs}),
                "output": summarize(result),
            })
        return result

    wrapper.bead = REGISTRY[name]
    return wrapper


def bead(
    name: str,
    *,
    strand: str,
    explains: str,
    after: tuple[str, ...] = (),
    title: str = "",
    tokens: tuple[str, ...] = (),
    io: tuple[str, str] = ("", ""),
) -> Callable:
    """Decorator form of `register()` for functions defined in this package."""
    return lambda fn: register(
        name,
        fn,
        strand=strand,
        explains=explains,
        after=after,
        title=title,
        tokens=tokens,
        io=io,
    )


@contextmanager
def trace() -> Iterator[list[dict[str, Any]]]:
    """Collect a summary record for every bead call made inside the block."""
    global _trace
    previous, _trace = _trace, []
    try:
        yield _trace
    finally:
        _trace = previous


def summarize(value: Any, depth: int = 0, head: int = 5) -> Any:
    """Return a JSON-friendly sketch of a value: type, shape and the first few entries."""
    if depth > 3:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "numpy"):  # torch tensor
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        flat = value.ravel()[:head]
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "head": [x.item() if isinstance(x, np.generic) else x for x in flat],
        }
    if hasattr(value, "shape") and hasattr(value, "head"):  # pandas object
        sketch = {"type": type(value).__name__, "shape": list(value.shape)}
        if hasattr(value, "columns"):
            sketch["columns"] = [str(c) for c in value.columns[:20]]
        return sketch
    if isinstance(value, dict):
        return {str(k): summarize(v, depth + 1, head) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "head": [summarize(v, depth + 1, head) for v in value[:head]],
        }
    return {"type": type(value).__name__}


def manifest(include_source: bool = False) -> list[dict[str, Any]]:
    """Every registered bead as a dict, in registration order."""
    out: list[dict[str, Any]] = []
    for b in REGISTRY.values():
        entry = asdict(b)
        if include_source:
            entry["source"] = source_of(b.name)
        out.append(entry)
    return out


def load_all_strands(include_source: bool = False) -> list[dict[str, Any]]:
    """Import all strand modules s01..s10 so every bead is registered, then return manifest."""
    for mod in STRAND_MODULES:
        importlib.import_module(mod)
    return manifest(include_source=include_source)
