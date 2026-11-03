"""Stackable trace context using ContextVar."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
from typing import Any, Callable, TypeVar
from uuid import uuid4

__all__ = ["trace", "current", "BASE_CONTEXT_ENV_VAR"]

_RESET_TRUTHY = frozenset({"true", "1", "yes", "on"})

#: JSON object merged *underneath* every trace scope in this process.
#:
#: Runners are frequently parallelised with a thread pool (tau2 uses
#: ``ThreadPoolExecutor``), and ``ContextVar`` values do not propagate into pool
#: worker threads. A scope entered in the launcher would therefore vanish for
#: every parallel task. Reading a baseline from the environment is immune to
#: that: it is process-wide, so it reaches calls made from any thread.
BASE_CONTEXT_ENV_VAR = "LLM_TRACE_BASE_CONTEXT"

F = TypeVar("F", bound=Callable[..., Any])

_trace_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "llm_cost_tracker_tracer_context", default={}
)


def _base_context() -> dict[str, str]:
    """Process-wide trace fields from ``$LLM_TRACE_BASE_CONTEXT`` (JSON object).

    Malformed or non-object values are ignored rather than raised: a bad
    environment variable must not take down an evaluation run mid-flight.
    """
    raw = os.environ.get(BASE_CONTEXT_ENV_VAR)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def current() -> dict[str, str]:
    """Current context fields, layered over the process-wide baseline.

    Scope fields win over the baseline, so an explicit ``tracer.trace(...)``
    can still override an environment-supplied value.
    """
    return {**_base_context(), **_trace_context.get()}


class trace:
    """Stackable trace scope as context manager and decorator."""

    __slots__ = ("_fields", "_token")

    def __init__(self, **fields: Any) -> None:
        self._fields = _normalize_trace_fields(fields)
        self._token: contextvars.Token | None = None

    def _merged(self) -> dict[str, str]:
        return {**_trace_context.get(), **self._fields}

    def __enter__(self) -> "trace":
        self._token = _trace_context.set(self._merged())
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._token is not None:
            _trace_context.reset(self._token)
            self._token = None
        return False

    def __call__(self, func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = _trace_context.set(self._merged())
                try:
                    return await func(*args, **kwargs)
                finally:
                    _trace_context.reset(token)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            token = _trace_context.set(self._merged())
            try:
                return func(*args, **kwargs)
            finally:
                _trace_context.reset(token)

        return sync_wrapper  # type: ignore[return-value]


def _normalize_trace_fields(fields: dict[str, Any]) -> dict[str, str]:
    """Normalize trace context fields for the proxy header.

    When ``reset=True``, assign a fresh ``_scope_id`` so the proxy truncates the
    canonical log once per trace scope (first LLM call), not once per process.
    """
    out: dict[str, Any] = dict(fields)
    reset_val = out.pop("reset", None)
    reset = str(reset_val).lower() in _RESET_TRUTHY if reset_val is not None else False
    if reset:
        out["reset"] = "true"
        out.setdefault("_scope_id", uuid4().hex[:12])
    normalized = {str(k): str(v) for k, v in out.items() if v is not None}
    return normalized

