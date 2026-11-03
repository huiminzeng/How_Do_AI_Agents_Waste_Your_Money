"""Where each benchmark's grades live, and how to read them.

Every function here reads the benchmarks' own outputs under benchmarks/<name>/src
(the locations are named in paths.py), so it works where the runs were made.
`grades()` is the one entry point the rest of the package uses.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

from .. import paths, prices

#: One rule per benchmark: tau2 and swe_bench grade pass/fail; claw_eval's rubric
#: score and mcp_atlas's claim coverage are 0-1, and >= 0.8 counts as solved.
SOLVED_AT = {"claw_eval": 0.8, "mcp_atlas": 0.8}

_CACHE: dict[tuple, dict | None] = {}


def grades(agent_model: str, benchmark: str, domain: str) -> dict[str, float] | None:
    """task_id -> share of that task's graded attempts solved, or None if never graded.

    mcp_atlas and swe_bench run each task once, so the share IS the judged
    trajectory's own result; tau2 and claw_eval run several trials and the trace
    does not say which one a trajectory was, so the task's mean stands in for it.
    Cached per process on the grade files' mtimes.
    """
    key = (agent_model, benchmark, domain, grade_sources(agent_model, benchmark, domain))
    if key not in _CACHE:
        if len(_CACHE) > 200:
            _CACHE.clear()
        _CACHE[key] = _load(agent_model, benchmark, domain)
    return _CACHE[key]


def grade_sources(agent_model: str, benchmark: str, domain: str) -> tuple:
    """(path, mtime) of every grade file this cell reads - the cache key, so a
    scoring file written while the server is up is seen on the next request."""
    files: list[Path] = []
    if benchmark == "tau2":
        from ..preprocess import strata
        files = [Path(f) for f in strata._tau2_reward_files_all().get((agent_model, domain), [])]
    elif benchmark == "mcp_atlas":
        files = [h for n in prices.names(agent_model) for h in paths.mcp_scored(n)]
    elif benchmark == "swe_bench":
        files = [h for n in prices.names(agent_model) for h in paths.swe_reports(n)]
    elif benchmark == "claw_eval":
        files = [h for n in prices.names(agent_model) for h in paths.claw_batches(n)]
    out = []
    for f in sorted(set(files)):
        try:
            out.append((str(f), f.stat().st_mtime))
        except OSError:
            continue
    return tuple(out)




def _by_task(pairs, benchmark: str) -> dict[str, float] | None:
    """(task_id, score) pairs -> task_id -> share of its attempts solved."""
    cut = SOLVED_AT.get(benchmark, 1.0)
    acc: dict[str, list[float]] = {}
    for task, score in pairs:
        acc.setdefault(str(task), []).append(1.0 if score >= cut else 0.0)
    if not acc:
        return None
    return {t: sum(v) / len(v) for t, v in acc.items()}


def _load(agent_model: str, benchmark: str, domain: str) -> dict | None:
    if benchmark == "tau2":
        from ..preprocess import strata
        sims = strata.tau2_simulations(agent_model, domain)
        if not sims:
            return None
        return _by_task([(s.get("task_id"), float(s["reward_info"]["reward"]))
                         for s in sims
                         if (s.get("reward_info") or {}).get("reward") is not None],
                        benchmark)
    if benchmark == "mcp_atlas":
        hits = sorted(h for n in prices.names(agent_model) for h in paths.mcp_scored(n))
        if not hits:
            return None
        csv.field_size_limit(sys.maxsize)
        # Every scoring file of the model, oldest first, later files winning a
        # task scored twice: a run that adds tasks writes its own file.
        per: dict[str, float] = {}
        for h in hits:
            with h.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        per[str(row["TASK"]).strip()] = float(row["coverage_score"])
                    except (TypeError, ValueError, KeyError):
                        continue
        return _by_task(list(per.items()), benchmark)
    if benchmark == "swe_bench":
        hits = sorted(h for n in prices.names(agent_model) for h in paths.swe_reports(n))
        if not hits:
            return None
        # Every report of the model, merged: a later run's report only lists
        # the instances IT ran, so the union is the model's graded set.
        verdict: dict[str, float] = {}
        for h in hits:
            d = json.loads(h.read_text())
            resolved = set(d.get("resolved_ids") or [])
            for t in d.get("submitted_ids") or []:
                verdict[t] = 1.0 if t in resolved else 0.0
        if not verdict:
            return None
        return _by_task(list(verdict.items()), benchmark)
    if benchmark == "claw_eval":
        hits = sorted(h for n in prices.names(agent_model) for h in paths.claw_batches(n))
        if not hits:
            return None
        # Every batch of the model (a model's trace directory holds all its
        # runs, and the sample may come from any of them), superseded batches
        # left out.
        from ..preprocess.sources import CELLS
        cell = next((c for c in CELLS if c.agent_model == agent_model
                     and c.benchmark == "claw_eval"), None)
        skip = set((cell.exclude_run_ids or ()) if cell else ())
        out = []
        for h in hits:
            if h.parent.name in skip:
                continue
            for task in json.loads(h.read_text()):
                for tr in task.get("trials") or []:
                    if isinstance(tr, dict) and tr.get("task_score") is not None:
                        out.append((task.get("task_id"), float(tr["task_score"])))
        return _by_task(out, benchmark)
    return None


