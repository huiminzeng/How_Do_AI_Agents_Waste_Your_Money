"""Repeated tool calls: the same call issued again, whose result is already in
the context.

Counts only. A call that returns what the context already holds is a fact, found
by comparison and not by a judge. What it cost is a token question and is
answered in `exact_tokens`; nothing here is a size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from ..preprocess.duplicates import find
from ..schema import Turn


@dataclass
class ContextWaste:
    n_traces: int = 0
    n_repeats: int = 0
    #: Repeats with nothing but tool output in between - nobody asked.
    n_unasked: int = 0
    #: Same call, DIFFERENT result. Counted here, and ruled on by the judge -
    #: see `add_judged_rechecks`.
    n_rechecks: int = 0
    #: Re-checks the judge marked as removable. Counted separately because it
    #: rests on a judgement rather than on a comparison, and a reader should be
    #: able to take the mechanical figure alone.
    n_rechecks_judged_waste: int = 0
    #: Trajectories that contained at least one repeat.
    traces_affected: int = 0
    worst: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "n_traces": self.n_traces,
            "traces_affected": self.traces_affected,
            "n_repeats": self.n_repeats,
            "n_unasked": self.n_unasked,
            "n_user_asked": self.n_repeats - self.n_unasked,
            "n_rechecks": self.n_rechecks,
            "n_rechecks_judged_waste": self.n_rechecks_judged_waste,
            "worst": self.worst[:10],
        }


def add(cw: ContextWaste, trace_id: str, turns: list[Turn]) -> None:
    cw.n_traces += 1
    report = find(turns)
    cw.n_rechecks += len(report.rechecks)
    repeats = report.all_repeats
    if not repeats:
        return
    cw.traces_affected += 1
    for r in repeats:
        cw.n_repeats += 1
        cw.n_unasked += r.strict
        cw.worst.append({
            "trace_id": trace_id, "round": r.round,
            "first_round": r.first_round, "kind": r.kind,
            "turns_remaining": r.turns_remaining,
            "action": r.action[:120],
        })
    cw.worst.sort(key=lambda d: -d["turns_remaining"])


def add_judged_rechecks(cw: ContextWaste, trace_id: str, turns: list,
                        verdicts: list) -> None:
    """Record a re-check the judge marked removable.

    A repeated call whose result CHANGED is not decidable by comparison, so
    `find()` only flags it and the judge rules. Kept in its own counter rather
    than folded into `n_repeats`, because the mechanical figure is a comparison
    and this one is a judgement. A reader who trusts only the first can still
    have it.
    """
    rechecks = find(turns).rechecks_by_round()
    if not rechecks:
        return
    marked = {v.round for v in verdicts
              if v.stream == "action" and not v.error and v.aligned
              and v.citation_ok and v.spans}
    for rd, rc in rechecks.items():
        if rd not in marked:
            continue
        cw.n_rechecks_judged_waste += 1
        cw.worst.append({
            "trace_id": trace_id, "round": rc.round,
            "first_round": rc.first_round, "kind": "judged-recheck",
            "turns_remaining": rc.turns_remaining,
            "action": rc.action[:120],
        })


def table(cw: ContextWaste, name: str = "") -> str:
    if not cw.n_repeats:
        return (f"  {name}no repeated call returned an identical result"
                + (f"; {cw.n_rechecks} repeated a call and got a different "
                   f"result, left to the judge" if cw.n_rechecks else ""))
    return (f"  {name}{cw.n_repeats} repeated calls in "
            f"{cw.traces_affected}/{cw.n_traces} trajectories "
            f"({cw.n_unasked} unasked, {cw.n_repeats - cw.n_unasked} user-asked)"
            + (f"; {cw.n_rechecks} more repeated a call and got a different "
               f"result, left to the judge" if cw.n_rechecks else ""))


def save(cw: ContextWaste, judge_id: str, run_id: str, agent_model: str,
         benchmark: str, domain: str) -> Path:
    """Filed under the judge's run even though no judge was involved.

    Deliberate: it is computed over exactly the trajectories that run sampled,
    so pairing it with any other run's output half would divide two different
    populations. The file says which run it belongs to so that cannot happen by
    accident.
    """
    path = paths.ensure_parent(
        paths.cell_dir(paths.EFFICIENCY_TASK, judge_id, run_id, agent_model,
                       benchmark, domain) / "context_waste.json")
    path.write_text(json.dumps(cw.to_json(), indent=2) + "\n")
    return path
