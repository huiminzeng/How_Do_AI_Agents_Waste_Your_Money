"""Resolve experiment-level ``run_id`` for trace path hashing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path, None]


def run_id_from_dir(
    experiment_dir: PathLike = None,
    *,
    env_var: str | None = None,
    default: str = "default",
) -> str:
    """Return a stable batch/experiment id for ``tracer.trace(run_id=...)``.

    Resolution order (same convention for all benchmarks):

    1. Non-empty ``env_var`` value (e.g. ``TAU2_RUN_ID``, ``MINI_SWE_RUN_ID``)
    2. Basename of the native experiment output directory (``--save-to``, ``-o``, …)
    3. ``default`` when neither is available
    """
    if env_var:
        val = os.environ.get(env_var)
        if val is not None and str(val).strip():
            return str(val).strip()
    if experiment_dir is not None:
        name = Path(experiment_dir).name
        if name:
            return name
    return default
