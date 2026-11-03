"""Per-trajectory outcomes, read from each benchmark's own record.

Every path below reads the benchmarks' own outputs under benchmarks/<name>/src
(the locations are named in paths.py). `outcome_for` is the entry point: for a
set of extracted trajectories it answers 1.0 (solved) / 0.0 (failed) / None
(not tied to one result) per trace id.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from .. import paths, prices
from .grade_sources import SOLVED_AT

_CACHE: dict[tuple, dict] = {}



def _unwrap(x) -> str:
    x = (x or "").strip()
    if x.startswith("{"):
        try:
            j = json.loads(x)
            if isinstance(j, dict) and "message" in j:
                return str(j["message"]).strip()
        except ValueError:
            pass
    return x


def _squash(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def _subsequence(needles: list[str], hay: list[str]) -> bool:
    """Every needle appears in hay, in order (hay may hold extra items)."""
    i = 0
    for n in needles:
        while i < len(hay) and hay[i] != n:
            i += 1
        if i == len(hay):
            return False
        i += 1
    return True


def _judged(agent, bench, dom, run_dir: Path) -> list[str]:
    ex = json.loads((run_dir / "exact_tokens.json").read_text())
    return sorted({r["trace_id"] for r in ex.get("rounds", [])})


def archive_task_ids(agent: str, bench: str, want) -> dict[str, str]:
    """trace_id -> task id for ids the traces index does not carry.

    The index carries the task id for every trajectory the proxy recorded, so
    there is nothing to recover; kept as the hook the callers expect.
    """
    return {}


def _turns(trace_ids):
    from ..preprocess.sample import iter_index, load_turns
    idx = {r["trace_id"]: r for r in iter_index()}
    return {t: (idx[t], load_turns(idx[t])) for t in trace_ids if t in idx}


# ------------------------------------------------------------------ per bench


def _swe(agent, trace_ids, rows) -> dict:
    hits = sorted(h for n in prices.names(agent) for h in paths.swe_reports(n))
    if not hits:
        return {t: None for t in trace_ids}
    resolved: set = set()
    submitted: set = set()
    for h in hits:                       # every report of the model, merged
        d = json.loads(h.read_text())
        sub = set(d.get("submitted_ids") or [])
        res = set(d.get("resolved_ids") or [])
        submitted |= sub
        resolved = (resolved - sub) | res   # a later report decides its own instances
    extra = archive_task_ids(agent, "swe_bench",
                             [t for t in trace_ids if t not in rows])
    out = {}
    for t in trace_ids:
        task = str(rows[t][0]["task_id"]) if t in rows else extra.get(t)
        out[t] = (1.0 if task in resolved else 0.0) if task in submitted else None
    return out


def _mcp(agent, trace_ids, rows) -> dict:
    hits = sorted(h for n in prices.names(agent) for h in paths.mcp_scored(n))
    if not hits:
        return {t: None for t in trace_ids}
    csv.field_size_limit(sys.maxsize)
    score: dict[str, float] = {}
    for h in hits:                       # every scoring file, later wins a task
        with h.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    score[str(row["TASK"]).strip()] = float(row["coverage_score"])
                except (TypeError, ValueError, KeyError):
                    continue
    cut = SOLVED_AT["mcp_atlas"]
    out = {}
    for t in trace_ids:
        task = str(rows[t][0]["task_id"]) if t in rows else None
        out[t] = (1.0 if score[task] >= cut else 0.0) if task in score else None
    return out


def _tau2(agent, dom, trace_ids, rows) -> dict:
    from ..preprocess import strata
    sims = strata.tau2_simulations(agent, dom)
    if not sims:
        return {t: None for t in trace_ids}
    by_task: dict[str, list] = {}
    for s in sims:
        by_task.setdefault(str(s.get("task_id")), []).append(s)
    out = {}
    for t in trace_ids:
        if t not in rows:
            out[t] = None
            continue
        row, turns = rows[t]
        vis = [_unwrap(x.visible_text) for x in turns if x.visible_text.strip()]
        cands = []
        for s in by_task.get(str(row["task_id"]), []):
            am = [_unwrap(m.get("content")) for m in s.get("messages") or []
                  if m.get("role") == "assistant" and (m.get("content") or "").strip()]
            if vis and _subsequence(vis, am):
                cands.append(s)
        rewards = {float((s.get("reward_info") or {}).get("reward", -1)) for s in cands}
        if len(rewards) == 1:
            out[t] = 1.0 if rewards.pop() >= 1.0 else 0.0
        else:
            out[t] = None
    return out


def _claw_trial_content(path: Path) -> tuple[set, list[str]]:
    ids: set = set()
    texts: list[str] = []
    for line in path.open():
        r = json.loads(line)
        if r.get("type") != "message":
            continue
        m = r.get("message") or {}
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.startswith("["):
            try:
                import ast
                c = ast.literal_eval(c)
            except (ValueError, SyntaxError):
                pass
        if isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and b.get("id"):
                    ids.add(b["id"])
                if b.get("type") == "text" and b.get("text"):
                    texts.append(_squash(b["text"]))
        elif isinstance(c, str) and c.strip():
            texts.append(_squash(c))
    return ids, texts


def _claw_proxy_content(cell, tid) -> tuple[set, list[str]]:
    ids: set = set()
    texts: list[str] = []
    pat = cell.pattern.replace("trace_*", "trace_" + tid)
    for root in paths.MONITOR_ROOTS:
        if not root.exists():
            continue
        for f in root.glob(pat):
            for line in f.open():
                r = json.loads(line)
                if r.get("kind", "call") != "call":
                    continue
                m = r.get("response_message") or {}
                for tc in m.get("tool_calls") or []:
                    if tc.get("id"):
                        ids.add(tc["id"])
                if m.get("content"):
                    texts.append(_squash(str(m["content"])))
    return ids, texts


def _claw(agent, trace_ids, rows) -> dict:
    from ..preprocess.sources import CELLS
    root = paths.CLAW_ROOT
    cell = next((c for c in CELLS if c.agent_model == agent and c.benchmark == "claw_eval"), None)
    # EVERY batch of this model, not the newest only: a model's proxy directory
    # holds all its runs, and the judged trajectory may come from any of them.
    skip = set(cell.exclude_run_ids or ()) if cell else set()
    batches = sorted(str(h) for n in prices.names(agent) for h in paths.claw_batches(n)
                     if h.parent.name not in skip)
    trials: dict[str, list] = {}
    for b in batches:
        for task in json.loads(Path(b).read_text()):
            for tr in task.get("trials") or []:
                if isinstance(tr, dict) and tr.get("trace") and tr.get("task_score") is not None:
                    trials.setdefault(str(task.get("task_id")), []).append(tr)
    cut = SOLVED_AT["claw_eval"]
    out = {}
    for t in trace_ids:
        if t not in rows or cell is None:
            out[t] = None
            continue
        row, _ = rows[t]
        pids, ptexts = _claw_proxy_content(cell, t)
        cands = []
        for tr in trials.get(str(row["task_id"]), []):
            p = root / tr["trace"]
            if not p.exists():
                continue
            ids, texts = _claw_trial_content(p)
            if pids and ids:
                if ids & pids:
                    cands.append(tr)
            elif ptexts and _subsequence(ptexts, texts):
                cands.append(tr)
        if len(cands) > 1 and ptexts:
            # Same ids in several trials (a provider that numbers calls from
            # zero): the reply texts decide.
            narrowed = [tr for tr in cands
                        if _subsequence(ptexts, _claw_trial_content(root / tr["trace"])[1])]
            if narrowed:
                cands = narrowed
        results = {1.0 if float(tr["task_score"]) >= cut else 0.0 for tr in cands}
        out[t] = results.pop() if len(results) == 1 else None
    return out


# ------------------------------------------------------------------- public


def outcomes(agent: str, bench: str, dom: str, run_dir) -> dict[str, float | None]:
    """trace_id -> 1.0 (solved) / 0.0 (failed) / None (not tied to one result),
    for the trajectories exact_tokens.json in `run_dir` counted."""
    from .grade_sources import grade_sources
    run_dir = Path(run_dir)
    key = (agent, bench, dom, str(run_dir), (run_dir / "exact_tokens.json").stat().st_mtime,
           grade_sources(agent, bench, dom))
    if key in _CACHE:
        return _CACHE[key]
    ids = _judged(agent, bench, dom, run_dir)
    out = outcome_for(agent, bench, dom, ids)
    _CACHE[key] = out
    return out


def outcome_for(agent: str, bench: str, dom: str, trace_ids) -> dict[str, float | None]:
    """Same result for an arbitrary set of EXTRACTED trajectories (judged or
    not); the sampler uses it to see which trajectories of a cell's whole pool
    succeeded. Cached per (cell, id set)."""
    from .grade_sources import grade_sources
    ids = sorted(set(trace_ids))
    key = (agent, bench, dom, "ids", hash(tuple(ids)), grade_sources(agent, bench, dom))
    if key in _CACHE:
        return _CACHE[key]
    rows = _turns(ids)
    if bench == "swe_bench":
        out = _swe(agent, ids, rows)
    elif bench == "mcp_atlas":
        out = _mcp(agent, ids, rows)
    elif bench == "tau2":
        out = _tau2(agent, dom, ids, rows)
    elif bench == "claw_eval":
        out = _claw(agent, ids, rows)
    else:
        out = {t: None for t in ids}
    _CACHE[key] = out
    return out

