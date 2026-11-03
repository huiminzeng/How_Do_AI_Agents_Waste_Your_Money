"""Inject stackable trace context into LLM proxy HTTP requests.

Usage:
    from llm_cost_tracker import tracer
    tracer.install("http://localhost:4000")

    @tracer.trace(benchmark="tau2", run_id="exp-1")
    def run():
        ...

Supported wire format:
    X-Trace-Context: base64(json({...}))

Trace scopes may pass ``reset=True`` to truncate the canonical calls.jsonl once
per scope (first LLM call). A fresh ``_scope_id`` is assigned automatically.

Use ``run_id_from_dir()`` to derive a stable experiment id from the native
output directory or a ``<BENCH>_RUN_ID`` environment override.

The proxy trace logger accepts both:
    - x-trace-meta (legacy plain JSON string)
    - x-trace-context (tracer-native base64 JSON)
"""

from __future__ import annotations

from .context import current, trace
from .patchers import active_patchers, install, uninstall
from .run_id import run_id_from_dir

__all__ = [
    "trace",
    "current",
    "install",
    "uninstall",
    "active_patchers",
    "run_id_from_dir",
]

