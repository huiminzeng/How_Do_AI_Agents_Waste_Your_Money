"""The boundary. The only module that knows how the proxy stores traces.

Everything downstream consumes `Trace` objects and has no idea what a proxy
record looks like. When the proxy's format changes, this file changes and
nothing else does.

artifacts/trajectories/<owner>/<credential>/<benchmark>/<domain>/<model>/trace_<id>/calls.jsonl
is one trajectory; `CELLS` is discovered from that tree at import time, one
cell per (benchmark, domain, requested alias).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .. import paths
from ..schema import ACTION_CHARS, VISIBLE_CHARS, OBSERVATION_CHARS, Trace, Turn
from . import inline_think


@dataclass(frozen=True)
class Cell:
    """One (agent model, benchmark, domain) unit of source data."""

    agent_model: str
    benchmark: str
    domain: str
    #: Always "monitor" (one directory per trace).
    layout: str
    #: Glob under artifacts/trajectories/.
    pattern: str
    #: Only trajectories whose trace_meta.run_id is in this set. None = every run
    #: in the directory. Set when a cell was RE-RUN into the same model directory
    #: (the proxy files every run of a model together), so a superseded run's
    #: trajectories are not judged beside the new ones.
    run_ids: tuple[str, ...] | None = None
    #: Runs to LEAVE OUT (superseded). The opposite of `run_ids`, and the one
    #: to prefer: an include-list silently drops every future run
    #: of the model, which is exactly what a sample grown over several runs
    #: cannot afford.
    exclude_run_ids: tuple[str, ...] | None = None
    #: Only trajectories whose rows carry this `requested_model` (the alias the
    #: harness asked the proxy for). The proxy files every alias of a model in
    #: ONE directory, so a reasoning-effort arm (kimi-k3-low) shares kimi-k3's
    #: directory and is told apart only by this field.
    requested_models: tuple[str, ...] | None = None
    #: The mirror image, for the baseline cell in the same directory: leave out
    #: the arms' trajectories. Rows from before the field existed (None) are
    #: kept, which is right - nothing but the baseline ran then.
    exclude_requested_models: tuple[str, ...] | None = None

    def accepts(self, first_rec: dict) -> bool:
        """Whether a trajectory whose FIRST call row is `first_rec` belongs to
        this cell. One rule for the trace reader and the spend composition."""
        meta = first_rec.get("trace_meta") or {}
        run = str(meta.get("run_id") or "")
        if self.run_ids is not None and run not in self.run_ids:
            return False
        if self.exclude_run_ids and run in self.exclude_run_ids:
            return False
        req = first_rec.get("requested_model")
        if self.requested_models is not None and req not in self.requested_models:
            return False
        if self.exclude_requested_models and req in self.exclude_requested_models:
            return False
        return True

    @property
    def key(self) -> str:
        return f"{self.agent_model}/{self.benchmark}/{self.domain}"


#: Where each cell's traces live. Adding an agent model is a data change here,
#: never a code change downstream.
_MONITOR_TAGS: dict[str, str] = {
    "tau2": "tau2",
    "mini_swe": "swe_bench",
    "claw_eval": "claw_eval",
    "mcp_atlas": "mcp_atlas",
}


def _first_record(f: Path) -> dict[str, Any] | None:
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return None
    return None


def _accepts_any(cell: Cell) -> bool:
    """Whether at least one trajectory on disk passes the cell's pins."""
    for root in paths.MONITOR_ROOTS:
        if not root.exists():
            continue
        for f in root.glob(cell.pattern):
            rec = _first_record(f)
            if rec and cell.accepts(rec):
                return True
    return False


def discover_cells(listed: tuple[Cell, ...] = ()) -> tuple[Cell, ...]:
    """One monitor cell per alias found under each benchmark directory.

    Walks `artifacts/trajectories/<owner>/<credential>/<tag>/<domain>/<model>/`. The proxy
    files every alias of a model in ONE directory (kimi-k3-high lands under
    kimi-k3/), so the cell is named after `requested_model` - the alias the
    harness asked for - and pinned to it. Every run of that alias in the
    directory is read; to judge one run only, pass a `Cell` with `run_ids` set
    (`listed` cells win over discovered ones for the same key when their pins
    match something on disk).
    """
    found: dict[str, Cell] = {}
    for root in paths.MONITOR_ROOTS:
        if not root.exists():
            continue
        for d in root.glob("*/*/*/*/*"):
            if not d.is_dir():
                continue
            tag = d.parent.parent.name
            benchmark = _MONITOR_TAGS.get(tag, tag)       # any other tag is a benchmark of its own
            if "-" in benchmark:
                continue                                  # "-" separates benchmark and domain in cell names
            dom = d.parent.name
            domain = "all" if dom in ("no_domain", "all") or benchmark in ("swe_bench", "claw_eval", "mcp_atlas") else dom
            pattern = f"*/*/{tag}/{dom}/{d.name}/trace_*/calls.jsonl"
            aliases: set[str | None] = set()
            for f in d.glob("trace_*/calls.jsonl"):
                rec = _first_record(f)
                if rec:
                    req = rec.get("requested_model")
                    aliases.add(req.split("/")[-1] if req else None)
            for alias in aliases:
                cell = Cell(alias or d.name, benchmark, domain, "monitor", pattern,
                            requested_models=(alias,) if alias else None)
                if cell.key not in found:
                    found[cell.key] = cell
    keep: dict[str, Cell] = {}
    for key, cell in found.items():
        prior = [c for c in listed if c.key == key]
        if prior and any(_accepts_any(c) for c in prior if c.layout == "monitor"):
            continue
        if prior and any(c.layout == "results" for c in prior):
            continue
        keep[key] = cell
    return tuple(keep.values())


CELLS: tuple[Cell, ...] = discover_cells()


def cells_for(agent_model: str | None = None) -> tuple[Cell, ...]:
    if agent_model is None:
        return CELLS
    return tuple(c for c in CELLS if c.agent_model == agent_model)


# --------------------------------------------------------------- record level


def _is_call(rec: dict[str, Any]) -> bool:
    """Trace files also carry non-call bookkeeping lines."""
    return rec.get("kind") == "call"


def _reasoning_text(rec: dict[str, Any]) -> str:
    msg = rec.get("response_message") or {}
    text = msg.get("reasoning_content")
    if not text and inline_think.is_inline(rec.get("model")):
        text, _ = inline_think.split(msg.get("content"))
    return str(text).strip() if text else ""


def _prev_role(rec: dict[str, Any]) -> str:
    """Who wrote the last message the model saw."""
    msgs = rec.get("request_messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    last = msgs[-1]
    return str(last.get("role") or "") if isinstance(last, dict) else ""


def _prev_observation(rec: dict[str, Any]) -> str:
    """The last thing the model saw before it reasoned.

    Truncated head-only: the informative part of a tool result is at the top,
    and swe_bench observations have a long tail of file dumps.
    """
    msgs = rec.get("request_messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    last = msgs[-1]
    if not isinstance(last, dict):
        return ""
    content = last.get("content")
    if content is None:
        return ""
    text = content if isinstance(content, str) else json.dumps(content)
    if len(text) <= OBSERVATION_CHARS:
        return text
    return text[:OBSERVATION_CHARS] + " ...[truncated]"


def _outgoing_action(rec: dict[str, Any]) -> str:
    """Every tool call this turn, as `name(arguments)` lines.

    All of them, not the first: `tool_called` keeps only the first, and a turn
    that greps three files in parallel is three calls. Rendered as text rather
    than kept structured because every consumer either shows it to a judge or
    pattern-matches it.
    """
    msg = rec.get("response_message")
    if not isinstance(msg, dict):
        return ""
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return ""
    parts = []
    for c in calls:
        fn = (c or {}).get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False) if args is not None else ""
        parts.append(f"{fn.get('name') or '?'}({args})")
    text = "\n".join(parts)
    if len(text) <= ACTION_CHARS:
        return text
    return text[:ACTION_CHARS] + " ...[truncated]"


def _visible_text(rec: dict[str, Any]) -> str:
    msg = rec.get("response_message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if not isinstance(content, str):
        content = "" if content is None else json.dumps(content, ensure_ascii=False)
    elif inline_think.is_inline(rec.get("model")):
        _, content = inline_think.split(content)
    if len(content) <= VISIBLE_CHARS:
        return content
    return content[:VISIBLE_CHARS] + " ...[truncated]"


def _to_turn(rec: dict[str, Any]) -> Turn:
    text = _reasoning_text(rec)
    return Turn(
        trace_id=str(rec.get("trace_id") or ""),
        round=int(rec.get("round") or 0),
        has_text=bool(text),
        reasoning_content=text,
        prev_observation=_prev_observation(rec),
        prev_role=_prev_role(rec),
        # NOTE: the proxy records only the FIRST tool of a multi-tool turn.
        # Downstream must not treat this as a complete tool list.
        tool_called=rec.get("tool_called"),
        output_tokens=int(rec.get("output_tokens") or 0),
        reasoning_tokens=int(inline_think.reasoning_tokens(rec) or 0),
        action=_outgoing_action(rec),
        visible_text=_visible_text(rec),
    )


def _task_id(rec: dict[str, Any]) -> str | None:
    meta = rec.get("trace_meta")
    if isinstance(meta, dict):
        for k in ("task_id", "unit_id", "instance_id", "id"):
            if meta.get(k):
                return str(meta[k])
    name = rec.get("trace_name")
    return str(name) if name else None


def _read_calls(path: Path) -> Iterator[dict[str, Any]]:
    """Stream call records, tolerating a truncated final line."""
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_call(rec):
                yield rec


# ------------------------------------------------------------- trajectories


def _traces_from_monitor(cell: Cell) -> Iterator[Trace]:
    """One directory per trace; each file holds that trajectory in round order."""
    for root in paths.MONITOR_ROOTS:
        if not root.exists():
            continue
        for f in sorted(root.glob(cell.pattern)):
            turns, task_id, trace_id = [], None, f.parent.name.removeprefix("trace_")
            for rec in _read_calls(f):
                if not turns and not cell.accepts(rec):
                    break
                turns.append(_to_turn(rec))
                task_id = task_id or _task_id(rec)
            if not turns:
                continue
            trace_id = turns[0].trace_id or trace_id
            yield Trace(
                trace_id=trace_id,
                agent_model=cell.agent_model,
                benchmark=cell.benchmark,
                domain=cell.domain,
                task_id=task_id,
                turns=turns,
            )


def iter_traces(cell: Cell) -> Iterator[Trace]:
    """Yield every trajectory in a cell, one at a time."""
    if cell.layout == "monitor":
        yield from _traces_from_monitor(cell)
    else:
        raise ValueError(f"unknown layout {cell.layout!r} for {cell.key}")
