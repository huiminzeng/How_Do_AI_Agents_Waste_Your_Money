"""The standing instructions the agent was given, per cell.

The efficiency judge is asked whether a stretch of output "says again what is
already in the context". It was shown each turn's last incoming message and
nothing else, so it could check a repeat of a tool result but not a repeat of the
policy document - and on tau2 the policy is where the cancellation rules, the
confirmation requirements and the current date live.

Measured, and the reason this is cheap: the system prompt is
IDENTICAL across all four agents within a domain. One document per cell, not per
trajectory and not per model:

    tau2/airline    8,033 characters
    tau2/retail     7,056 characters
    mcp_atlas       none at all - the first message is the user's question, so
                    everything the agent knows was already visible to the judge

So mcp_atlas needs nothing, and tau2 needs two files. Extracted once and stored
as text, rather than read from the full traces on every run.

What this does NOT do is give the judge the tool SCHEMAS. Those are also standing
context, and a repeat of a tool's description would be restatement too. They are
not in the system message on these routes, and chasing them would double the
prompt for a case not yet observed.
"""

from __future__ import annotations

import hashlib
import json

from .. import paths
from .sources import Cell, cells_for, _read_calls

#: How many calls to read before deciding a cell's policy. One would do, since
#: the document is constant, but reading a few and hashing them turns "it is
#: constant" from an assumption into a check that runs every extraction.
SAMPLE_CALLS = 40


def _files(cell: Cell):
    if cell.layout == "monitor":
        for root in paths.MONITOR_ROOTS:
            if root.exists():
                yield from sorted(root.glob(cell.pattern))
    else:
        f = paths.RESULTS / cell.pattern
        if f.exists():
            yield f


def extract_one(cell: Cell) -> tuple[str, int]:
    """This cell's system prompt, and how many distinct ones were seen.

    A count above 1 is the interesting outcome: it means the agents in this cell
    were not given the same instructions, and a single stored document would
    misrepresent some of them. The caller decides what to do; this only reports.
    """
    seen: dict[str, str] = {}
    n = 0
    for f in _files(cell):
        for rec in _read_calls(f):
            msgs = rec.get("request_messages")
            if not isinstance(msgs, list):
                continue
            sysm = [m for m in msgs
                    if isinstance(m, dict) and m.get("role") == "system"]
            if not sysm:
                continue
            text = str(sysm[0].get("content") or "")
            if text:
                seen[hashlib.sha256(text.encode()).hexdigest()] = text
            n += 1
            if n >= SAMPLE_CALLS:
                break
        if n >= SAMPLE_CALLS:
            break
    if not seen:
        return "", 0
    # Longest wins if a cell somehow has more than one: a truncated variant is
    # the failure mode worth guarding against, not a genuinely different policy.
    return max(seen.values(), key=len), len(seen)


#: Benchmarks whose system message differs per trajectory, so a single stored
#: document would misrepresent every trajectory but one.
#:
#: claw_eval is the only one. Its system message is three sections and two of
#: them are the task's own tool environment: the tools THIS task may use
#: (288-724 chars) and their full JSON schemas (280-2,592 chars). Only the
#: trailing rules (1,505 chars) are constant, verified by hashing them across
#: 600 calls spanning all four agents - one distinct value.
#:
#: Trimming to those rules was considered and rejected. Every other benchmark
#: gets the WHOLE system message: tau2 8,033 chars, swe_bench 95, mcp_atlas
#: none because it has none. The rule those three follow is "the judge sees the
#: standing context the agent had", and tau2 has no schemas in it only because
#: tau2 passes tools through the API parameter instead. claw_eval puts them in
#: the message, so for claw_eval they ARE standing context, and a line that
#: restates a tool's parameter documentation is restatement. Trimming would have
#: made claw_eval the one cell where the judge is shown less than the agent had.
PER_TRAJECTORY = ("claw_eval",)


def for_trace(benchmark: str, domain: str, agent_model: str,
              trace_id: str) -> str:
    """This TRAJECTORY's standing instructions, for the benchmarks that need it.

    Falls back to the cell document for everything else, so callers can use this
    unconditionally.

    Cheap because of the layout: claw_eval is `monitor`, one directory per
    trace, named by trace_id - so this reads one small file rather than scanning
    the 483 in the cell. A miss returns the cell document rather than nothing:
    an absent file should degrade to the old behaviour, not to no policy.
    """
    if benchmark not in PER_TRAJECTORY:
        return load(benchmark, domain)
    for cell in cells_for(agent_model):
        if cell.benchmark != benchmark or cell.domain != domain:
            continue
        for f in _files(cell):
            if trace_id not in f.parent.name:
                continue
            for rec in _read_calls(f):
                msgs = rec.get("request_messages")
                if not isinstance(msgs, list):
                    continue
                sysm = [m for m in msgs
                        if isinstance(m, dict) and m.get("role") == "system"]
                if sysm and sysm[0].get("content"):
                    return str(sysm[0]["content"])
            break
    return load(benchmark, domain)


def extract(agent_model: str | None = None) -> dict:
    """Write one policy file per cell that has one. Returns a manifest."""
    out: dict[str, dict] = {}
    for cell in cells_for(agent_model):
        key = paths.cell_seg(cell.benchmark, cell.domain)
        if key in out:
            continue
        text, variants = extract_one(cell)
        if not text:
            out[key] = {"chars": 0, "variants": 0}
            continue
        path = paths.ensure_parent(paths.POLICIES / f"{key}.txt")
        path.write_text(text)
        out[key] = {"chars": len(text), "variants": variants,
                    "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    paths.ensure_parent(paths.POLICIES / "manifest.json").write_text(
        json.dumps(out, indent=2) + "\n")
    return out


def load(benchmark: str, domain: str) -> str:
    """This cell's policy, or "" when it has none.

    An empty string is a real answer, not a missing file: mcp_atlas has no
    standing instructions, and the prompt renders no policy block at all rather
    than an empty one that would invite the judge to reason about its absence.
    """
    path = paths.POLICIES / f"{paths.cell_seg(benchmark, domain)}.txt"
    try:
        return path.read_text()
    except OSError:
        return ""
