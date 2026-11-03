"""The tool calls of one turn as objects, read back from calls.jsonl.

The trajectory files hold each turn's tool calls as ONE rendered string,
`name({arguments})` per call joined by newlines (`sources._outgoing_action`).
That string is what the judge reads and marks, and it stays that way. But the
provider did not bill that string: it billed its own encoding of the call. To
count tokens the way the bill does, the counting stage needs the call back as
an object - name and argument dict - and this module fetches it from the same
record the string was rendered from.

Nothing is rebuilt or rewritten: the judged inputs are untouched, and every
result is checked against them (`render` must reproduce the stored string
exactly before a turn's objects are trusted).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import paths
from ..schema import ACTION_CHARS
from . import sources

_CACHE: dict[tuple, dict[int, list[tuple[str, str]]]] = {}


def _cell(agent_model: str, benchmark: str, domain: str) -> sources.Cell:
    for c in sources.CELLS:
        if (c.agent_model, c.benchmark, c.domain) == (agent_model, benchmark, domain):
            return c
    raise KeyError(f"no source cell for {agent_model}/{benchmark}/{domain}")


def _calls_of(rec: dict[str, Any]) -> list[tuple[str, str]]:
    """(name, arguments-as-string) per call, the way the renderer saw them."""
    msg = rec.get("response_message")
    if not isinstance(msg, dict):
        return []
    out = []
    for c in msg.get("tool_calls") or []:
        fn = (c or {}).get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False) if args is not None else ""
        out.append((fn.get("name") or "?", args))
    return out


def calls_for(agent_model: str, benchmark: str, domain: str,
              trace_id: str) -> dict[int, list[tuple[str, str]]]:
    """round -> [(name, arguments string)] for one trajectory."""
    key = (agent_model, benchmark, domain, trace_id)
    if key in _CACHE:
        return _CACHE[key]
    cell = _cell(agent_model, benchmark, domain)
    out: dict[int, list[tuple[str, str]]] = {}
    if cell.layout == "monitor":
        pattern = cell.pattern.replace("trace_*", f"trace_{trace_id}")
        hits = [f for root in paths.MONITOR_ROOTS
                if root.exists() for f in root.glob(pattern)]
        if not hits:
            raise FileNotFoundError(f"no calls.jsonl for {cell.key} {trace_id}")
        for rec in sources._read_calls(hits[0]):
            out[int(rec.get("round") or 0)] = _calls_of(rec)
    else:
        # Consolidated file layout: one pass, records for this trace only.
        f = paths.RESULTS / cell.pattern
        for rec in sources._read_calls(f):
            if str(rec.get("trace_id") or "") == trace_id:
                out[int(rec.get("round") or 0)] = _calls_of(rec)
    _CACHE[key] = out
    return out


def render(calls: list[tuple[str, str]]) -> str:
    """Exactly `sources._outgoing_action`, so the two can be compared."""
    text = "\n".join(f"{name}({args})" for name, args in calls)
    if len(text) <= ACTION_CHARS:
        return text
    return text[:ACTION_CHARS] + " ...[truncated]"


def segments(calls: list[tuple[str, str]]) -> list[tuple[int, int]]:
    """[start, end) of each call inside `render(calls)` (before truncation)."""
    out, pos = [], 0
    for name, args in calls:
        n = len(name) + 1 + len(args) + 1
        out.append((pos, pos + n))
        pos += n + 1
    return out


# ------------------------------------------------------------------ context
_CTX_CACHE: dict[tuple, dict[str, dict[int, int]]] = {}
_CTX_SCANNED: set[tuple] = set()


def _ctx_of(rec: dict[str, Any]) -> int:
    """Tokens the model read this round: fresh input, cache reads and cache
    writes summed. A cached token is still context the model attended to;
    only its price differs, and price is not what this number is for."""
    return sum(int(rec.get(k) or 0) for k in
               ("input_tokens", "cache_read_tokens", "cache_write_tokens"))


def context_for(agent_model: str, benchmark: str, domain: str,
                trace_id: str) -> dict[int, int]:
    """round -> context tokens for one trajectory, from the same call records
    `calls_for` reads.

    Consolidated files are scanned ONCE per cell and every trace's rounds kept,
    because a per-trace pass over a multi-GB file is too slow.
    Raises when the trajectory has no record: a missing context is a broken
    source, not a zero.
    """
    key = (agent_model, benchmark, domain)
    idx = _CTX_CACHE.setdefault(key, {})
    cell = _cell(agent_model, benchmark, domain)
    if cell.layout == "monitor":
        if trace_id not in idx:
            pattern = cell.pattern.replace("trace_*", f"trace_{trace_id}")
            hits = [f for root in paths.MONITOR_ROOTS
                    if root.exists() for f in root.glob(pattern)]
            if not hits:
                raise FileNotFoundError(f"no calls.jsonl for {cell.key} {trace_id}")
            idx[trace_id] = {int(rec.get("round") or 0): _ctx_of(rec)
                             for rec in sources._read_calls(hits[0])}
    elif key not in _CTX_SCANNED:
        for rec in sources._read_calls(paths.RESULTS / cell.pattern):
            tid = str(rec.get("trace_id") or "")
            idx.setdefault(tid, {})[int(rec.get("round") or 0)] = _ctx_of(rec)
        _CTX_SCANNED.add(key)
    if trace_id not in idx:
        raise KeyError(f"no call records for {cell.key} {trace_id}")
    return idx[trace_id]
