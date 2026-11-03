"""Rebuild one cell's token/dollar aggregates from verdicts already on disk.

Why this exists rather than a re-run of `evaluate.py --only efficiency`: that
path re-sends every trajectory `_has_usable` calls unfinished, and 14 of them
are unfinished because the judge exhausts its 131,072-token output budget on
them - twice now, hours apart. Re-running the pass to rebuild an aggregate
would spend those hours again and fail again.

The block below is `evaluate.py`'s own aggregation block, same calls in the
same order, with the judging step removed. The warning in that file - that this
must not live in a separate script - is about the ROUTE: `dollars` and the
figures once came from the character count while the front page came from the
token count, and the two disagreed silently. This script calls
`exact_tokens.build` and passes its output into `dollars.build`, which is the
token route, the same one `evaluate.py` uses. Keep it that way if you edit it.
"""
from __future__ import annotations

import argparse
import sys

from llmaj import paths
from llmaj import tokens as agent_tokens
from llmaj.aggregate import (context_waste, dollars, efficiency_spend,
                             exact_tokens)
from llmaj.evaluate import efficiency as eff_run
from llmaj.preprocess import sample


def rebuild(agent: str, bench: str, dom: str, judge: str, run_id: str,
            limit: int, seed: int, tasks_file: str | None, workers: int) -> None:
    if tasks_file:
        import json as _json
        rows = sample.select_listed_tasks(bench, dom, agent,
                                          _json.load(open(tasks_file)), seed)
    else:
        rows = sample.select(bench, dom, agent, limit, seed)
    if not rows:
        print(f"  {agent}/{bench}/{dom}: no trajectories selected"); return

    sp = efficiency_spend.build(judge, run_id, agent, bench, dom)
    if not sp.n_traces:
        print(f"  {agent}/{bench}/{dom}: no verdicts on disk"); return
    efficiency_spend.save(sp)

    cw = context_waste.ContextWaste()
    for row, turns in sample.iter_selected(rows):
        context_waste.add(cw, row["trace_id"], turns)
        context_waste.add_judged_rechecks(
            cw, row["trace_id"], turns,
            eff_run.verdicts_for(judge, run_id, agent, bench, dom, row["trace_id"]))

    counter = exact_tokens.Counter(agent)
    pr = dollars.prices_for(agent, bench, dom)
    groups = ((turns, eff_run.verdicts_for(judge, run_id, agent, bench, dom,
                                           row["trace_id"]))
              for row, turns in sample.iter_selected(rows))
    ex_cell = exact_tokens.build(agent, bench, dom, counter, groups,
                                 pr.output if pr else 0.0, workers=workers)
    exact_tokens.save(ex_cell, judge, run_id)
    counter.save()
    ex_json = ex_cell.to_json()
    print(exact_tokens.table(ex_cell), flush=True)

    w = dollars.build(agent, bench, dom, {r["trace_id"] for r in rows}, exact=ex_json)
    comp_c = agent_tokens.load(agent, bench, dom)
    dollars.save(w, judge, run_id, agent, bench, dom)
    context_waste.save(cw, judge, run_id, agent, bench, dom)
    print(dollars.table(w), flush=True)
    print(f"     -> {paths.cell_dir('efficiency', judge, run_id, agent, bench, dom)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("agent")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--domain", default="all")
    ap.add_argument("--judge", default="kimi-k3")
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rebuild(a.agent, a.benchmark, a.domain, a.judge, a.run,
            a.limit, a.seed, a.tasks_file, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
