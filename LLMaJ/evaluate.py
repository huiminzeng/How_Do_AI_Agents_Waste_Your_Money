#!/usr/bin/env python3
"""Evaluate one agent model across every benchmark we have traces for.

One of these per screen session. The agent model is the argument that differs
between sessions; everything else has a default.

    python evaluate.py kimi-k3
    python evaluate.py gpt-5.6-terra --judge kimi-k3 --limit 30
    python evaluate.py claude-sonnet-5 --dry-run

Runs both passes where the data allows:

    utility      needs reasoning text, so only kimi-k3 and gemini-3.1-pro have it.
                 gpt-5.6 and claude-sonnet-5 return none at all, and a cell with
                 no reasoning is skipped rather than labelled empty.
    efficiency   every agent: the visible reply and the tool calls are there
                 even when the reasoning is not.

Resumable and interruption-safe: a trajectory whose shard exists is skipped, so
re-running after a crash costs nothing and two sessions on the same agent cannot
duplicate work. Different agents write to different directories and never
contend.

--include-claw-eval is a no-op kept for scripts that pass it.
"""

from __future__ import annotations

import argparse
import os
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmaj import paths  # noqa: E402
from llmaj.aggregate import (context_waste, dollars, efficiency_spend,  # noqa: E402
                             exact_tokens, utility_matrix)
from llmaj.evaluate import efficiency as eff_run  # noqa: E402
from llmaj.evaluate import utility as util_run  # noqa: E402
from llmaj.evaluate.judge import Judge  # noqa: E402
from llmaj.preprocess.extract import iter_index  # noqa: E402
from llmaj.preprocess import sample  # noqa: E402
from llmaj.preprocess.sample import (DEFAULT_SEED, select,  # noqa: E402
                                    select_tasks, select_listed_tasks)
from llmaj.prompts.efficiency_prompt import EFF_PROMPT_VERSION  # noqa: E402
from llmaj.prompts.utility_prompt import PROMPT_VERSION  # noqa: E402
from llmaj import tokens as agent_tokens  # noqa: E402

#: Benchmarks `cells_for` hides unless asked for. Empty; kept as a name because
#: the mechanism is the right place for a benchmark that is present but not yet
#: judgeable.
SKIP_BENCHMARKS: tuple[str, ...] = ()


def preflight(auto_extract: bool, force: bool, agent: str | None = None,
              benchmark: str | None = None, domain: str | None = None,
              re_extract: bool = False) -> int:
    """Make sure the extracted traces exist and are current, before spending.

    Two failures this catches, both of which have happened:

      MISSING   the index is not there at all, and every cell reports "0
                trajectories" - which reads as "nothing to do" rather than as an
                error.
      STALE     source traces were appended AFTER the last extraction, so the
                shards are silently a run behind.

    The check is per REQUESTED cell (this agent, and the
    benchmark/domain asked for), and extraction is incremental:

      - a requested cell that was never extracted is extracted now, on its
        own, without touching the other cells or waiting for their judges;
      - a requested cell whose source traces are newer than its extraction is
        STALE. Refused by default, because re-extracting can change which
        trial of a task `select_tasks` picks, and doing that underneath a
        comparison is how two runs end up measured on different data. Pass
        --re-extract to bring just those cells up to date first, or --stale-ok
        to judge what is there.
    """
    import datetime
    import json as _json
    import os

    from llmaj.preprocess.extract import cell_extracted_at, extract_locked
    from llmaj.preprocess.sources import CELLS

    def _show(stats):
        for st in stats:
            print(f"    {st.cell_key:<44}{st.traces:>5} traj{st.turns:>8} turns")
        print()

    manifest_path = paths.TRACES / "manifest.json"
    if not paths.TRACES_INDEX.exists() or not manifest_path.exists():
        if not auto_extract:
            print("  no extracted traces. Run:\n"
                  "    python evaluate.py <agent> --re-extract\n")
            return 1
        print("  no extracted traces yet — extracting everything (local, free)\n",
              flush=True)
        _show(extract_locked(None))
        return 0

    requested = [c for c in CELLS
                 if (agent is None or c.agent_model == agent)
                 and (benchmark is None or c.benchmark == benchmark)
                 and (domain is None or c.domain == domain)]
    manifest = _json.loads(manifest_path.read_text())

    def _newer(c, cut):
        if c.layout == "monitor":
            files = [f for root in paths.MONITOR_ROOTS
                     if root.exists() for f in root.glob(c.pattern)]
        else:
            f = paths.RESULTS / c.pattern
            files = [f] if f.exists() else []
        return sum(1 for f in files if os.path.getmtime(f) > cut)

    missing, stale = [], {}
    for c in requested:
        cut = cell_extracted_at(manifest, c.key)
        if cut is None:
            missing.append(c)
        else:
            n = _newer(c, cut)
            if n:
                stale[c.key] = (c, n, cut)

    todo = list(missing) + ([v[0] for v in stale.values()] if re_extract else [])
    if todo:
        if not auto_extract:
            print("  cells not extracted (or asked to re-extract): "
                  + ", ".join(c.key for c in todo) + "\n  but --no-extract was given.\n")
            return 1
        print("  extracting " + ", ".join(c.key for c in todo)
              + " (only these cells; nothing else is touched)\n", flush=True)
        _show(extract_locked(cells=todo))
        if re_extract:
            stale = {}
    if stale and not force:
        print("  STALE: source traces changed after these cells were extracted:\n")
        for k, (c, n, cut) in sorted(stale.items()):
            print(f"    {k:<44}{n:>5} files newer than "
                  f"{datetime.datetime.fromtimestamp(cut):%Y-%m-%d %H:%M}")
        print("\n  Judging a stale extract measures data that is a run behind.\n"
              "  Re-extract just these cells:  add --re-extract\n"
              "  Or judge what is there:       add --stale-ok\n")
        return 1
    if stale:
        print(f"  warning: {sum(v[1] for v in stale.values())} source files are newer "
              f"than the extract; --stale-ok given, continuing\n")
    return 0


#: Every agent being judged, read from the cell list rather than written down.
#: The task pool for a paired sample is the intersection over these, so a name
#: missing here silently unpairs the rest - and a hardcoded list would go stale
#: the first time a model is added.
def all_agents() -> tuple[str, ...]:
    from llmaj.preprocess.extract import iter_index
    from llmaj import prices
    # Effort-ablation arms are not models in their own right: they must not shrink
    # the paired task set or count as extra models in the difficulty classes.
    return tuple(sorted({r["agent_model"] for r in iter_index()
                         if not prices.is_arm(r["agent_model"])}))


def cells_for(agent: str, include_claw: bool, benchmark: str | None = None,
              domain: str | None = None) -> list[tuple[str, str, int, int]]:
    """(benchmark, domain, trajectories, reasoning turns), smallest first.

    DISCOVERED from work/input/traces/index.jsonl, not listed in this file. A
    cell appears only once its traces are extracted, so adding a benchmark is a
    re-extract rather than an edit here - and the reverse: a benchmark that looks
    missing has not been extracted yet, which `--list` will show.

    Ordered smallest-first: a session that is going to fail should fail on a
    cheap cell, not after an hour on swe_bench.
    """
    seen: dict[tuple[str, str], list[int]] = {}
    for r in iter_index():
        if r["agent_model"] != agent:
            continue
        if not include_claw and r["benchmark"] in SKIP_BENCHMARKS:
            continue
        if benchmark and r["benchmark"] != benchmark:
            continue
        if domain and r["domain"] != domain:
            continue
        k = (r["benchmark"], r["domain"])
        v = seen.setdefault(k, [0, 0])
        v[0] += 1
        v[1] += r["n_labellable"]
    return sorted(((b, d, n, lab) for (b, d), (n, lab) in seen.items()),
                  key=lambda x: x[2])


def list_cells(include_claw: bool) -> int:
    """Every agent's cells. The answer to "what will this actually run"."""
    from llmaj.preprocess.extract import iter_index as _idx

    seen: dict[str, dict[str, list[int]]] = {}
    for r in _idx():
        if not include_claw and r["benchmark"] in SKIP_BENCHMARKS:
            continue
        cell = f"{r['benchmark']}/{r['domain']}"
        v = seen.setdefault(r["agent_model"], {}).setdefault(cell, [0, 0])
        v[0] += 1
        v[1] += r["n_labellable"]
    cells = sorted({c for a in seen.values() for c in a})
    w = max(len(a) for a in seen) if seen else 20
    print(f"  {'agent':<{w + 2}}" + "".join(f"{c:>20}" for c in cells))
    print("  " + "-" * (w + 2 + 20 * len(cells)))
    for agent in sorted(seen):
        row = ""
        for c in cells:
            v = seen[agent].get(c)
            row += f"{'—':>20}" if not v else f"{f'{v[0]} traj, {v[1]} rsn':>20}"
        print(f"  {agent:<{w + 2}}{row}")
    print("\n  'rsn' is reasoning turns. 0 means the utility pass has nothing to "
          "read there\n  and only efficiency runs.\n")
    return 0


def _load_env() -> None:
    """Fill the environment from the project .env without overriding anything
    already set - the same rule scripts_exact.py uses. The judge key and the
    provider keys the token counters need all live there."""
    import os
    env = paths.PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="the agent model whose traces are judged")
    ap.add_argument("--judge", default="kimi-k3",
                    help="judge alias in LLMaJ/proxy/config.yaml (default kimi-k3)")
    ap.add_argument("--run", default=None,
                    help="run id; defaults to today's date plus 'a', e.g. 0831a. "
                         "It only has to distinguish repeats of the same judge on "
                         "the same agent - both are already directory levels, so "
                         "a run id that named them again ('kimi-gemini-a') just "
                         "restated the path, and a bare 'a' said nothing at all. "
                         "Stable within a day so an interrupted run resumes; pass "
                         "0831b for a deliberate second run")
    ap.add_argument("--target", type=int, default=None,
                    help="grow each cell's sample to this many trajectories with "
                         "balanced top-ups (length, outcome, difficulty); the paired "
                         "base stays --limit tasks. Only adds, never swaps.")
    ap.add_argument("--limit", type=int, default=30,
                    help="trajectories per cell (default 30)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--benchmark", default=None,
                    help="restrict to one benchmark (default: every one with "
                         "extracted traces)")
    ap.add_argument("--domain", default=None,
                    help="restrict to one domain, e.g. retail")
    ap.add_argument("--only", choices=["utility", "action", "efficiency"],
                    default=None)
    ap.add_argument("--list", action="store_true",
                    help="show which cells every agent has, then stop")
    ap.add_argument("--no-top-up", action="store_true",
                    help="skip the extra trajectories that bring each length "
                         "and difficulty group up to a usable size. The random "
                         "sample alone drew 0 of the 2 tasks all four models "
                         "failed on tau2/retail, and 1 of the 4 shortest.")
    ap.add_argument("--tasks-file", default=None,
                    help="JSON list of task ids: judge exactly these tasks (one "
                         "trajectory each) instead of sampling. For re-judging "
                         "a re-run cell on the tasks its superseded judging "
                         "covered - see sample.select_listed_tasks")
    ap.add_argument("--by-trajectory", action="store_true",
                    help="sample trajectories instead of tasks. Draws repeated "
                         "trials of the same task and gives each model a "
                         "different task set; kept only to reproduce a run made "
                         "before task sampling existed.")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent calls, both passes. Every judge session "
                         "shares one upstream endpoint, and six workers were "
                         "enough to draw HTTP 503s there; three ran clean. "
                         "Raise it only with a cell to spare.")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--include-claw-eval", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stale-ok", action="store_true",
                    help="proceed even though source traces are newer than the "
                         "extract")
    ap.add_argument("--re-extract", action="store_true",
                    help="re-extract the requested cells whose source traces are "
                         "newer than their extraction, before judging (only those "
                         "cells; other cells and their judges are untouched)")
    ap.add_argument("--no-extract", action="store_true",
                    help="do not extract automatically when the index is missing")
    a = ap.parse_args()

    rc = preflight(auto_extract=not a.no_extract, force=a.stale_ok,
                   agent=None if a.list else a.agent, benchmark=a.benchmark,
                   domain=a.domain, re_extract=a.re_extract)
    if rc:
        return rc

    if a.list:
        return list_cells(a.include_claw_eval)

    # Date, not a counter: the default has to be STABLE so that re-running the
    # same command after a crash resumes rather than starting a parallel run.
    # An auto-incrementing default would silently do the latter.
    run_id = a.run or f"{datetime.date.today():%m%d}a"
    cells = cells_for(a.agent, a.include_claw_eval, a.benchmark, a.domain)
    agents_all = all_agents()
    if not cells:
        where = f"{a.benchmark or 'any'}/{a.domain or 'any'}"
        print(f"\n  no extracted traces for {a.agent} at {where}.\n"
              f"  what IS extracted:\n")
        list_cells(a.include_claw_eval)
        print(f"  if a benchmark is missing entirely:\n"
              f"    python evaluate.py <agent> --re-extract\n")
        return 1

    # Hold a judge lock naming THESE cells for the rest of the run, so nobody
    # re-extracts them underneath us; other cells may still be extracted
    # (see extract.hold_judge_lock / _refuse_if_judged).
    import atexit
    from llmaj.preprocess.extract import hold_judge_lock
    atexit.register(hold_judge_lock([f"{a.agent}/{bench}/{dom}" for bench, dom, *_ in cells]))

    print(f"\n{'=' * 74}")
    print(f"  agent   {a.agent}")
    print(f"  judge   {a.judge}      run  {run_id}")
    print(f"  prompts utility {PROMPT_VERSION} · efficiency {EFF_PROMPT_VERSION}")
    print(f"  base    {Judge.__init__.__defaults__[0]}")
    print(f"{'=' * 74}\n")

    print(f"  {'cell':<20}{'traj':>7}{'reasoning turns':>18}"
          f"{'traj w/ reasoning':>20}   tasks")
    plan = []
    for bench, dom, n, lab in cells:
        # One selection per cell, shared by both tasks, so their results can be
        # read side by side.
        # One trajectory per TASK, and the same tasks for every model.
        #
        # Sampling trajectories drew 30 that covered only 24-27 tasks on tau2,
        # where every task was run 4 times - so several calls bought a second
        # look at a task already in the sample. It also gave the four models
        # DIFFERENT tasks: 3 of 30 shared on tau2/airline, 0 of 30 on retail and
        # mcp_atlas, which put every cross-model comparison on two different
        # sets of tasks. Both are gone at the same number of calls.
        if a.tasks_file:
            import json as _json
            rows = select_listed_tasks(bench, dom, a.agent,
                                       _json.load(open(a.tasks_file)), a.seed)
        else:
            # Trajectories this run has already judged stay in the sample even
            # if a newer trial of their task arrived since (sample.select_tasks).
            keep = set()
            for task in ("utility", "efficiency"):
                d = paths.cell_dir(task, a.judge, run_id, a.agent, bench, dom)
                if d.exists():
                    keep |= {f.name[:-6] for f in d.glob("meta-*.jsonl")}
            rows = (select(bench, dom, a.agent, a.limit, a.seed) if a.by_trajectory
                    else select_tasks(bench, dom, a.agent, a.limit, a.seed,
                                      pair_across=agents_all,
                                      top_up=not a.no_top_up,
                                      target=a.target, keep=keep))
        n_base = sum(1 for r in rows if r.get("sampled", "base") == "base")
        n_up = len(rows) - n_base
        turns = sum(r["n_labellable"] for r in rows)
        with_reasoning = sum(1 for r in rows if r["n_labellable"] >= 2)
        # No reasoning text means the utility prompt has nothing to read. Skipping
        # is not the same as labelling an empty stream, and the difference matters
        # for every share this feeds: a model that emits no reasoning must not
        # come out looking perfectly efficient at reasoning.
        tasks = []
        if a.only in (None, "utility") and with_reasoning:
            tasks.append("utility")
        # The action side has no reasoning precondition: a turn qualifies when
        # the agent SENT something, which is true of arms that never think.
        # Opt-in for now - it is a new pass and the default run should not
        # start spending on it silently.
        if a.only == "action":
            tasks.append("action")
        if a.only in (None, "efficiency"):
            tasks.append("efficiency")
        shown = f"{len(rows)}" + (f" ({n_base}+{n_up})" if n_up else "")
        print(f"  {bench + '/' + dom:<20}{shown:>11}{turns:>15}"
              f"{with_reasoning:>20}   {', '.join(tasks) or 'nothing to do'}")
        if tasks:
            plan.append((bench, dom, rows, tasks))

    if a.dry_run:
        print("\n  --dry-run: nothing sent\n")
        return 0
    if not plan:
        print("\n  nothing to do\n")
        return 0

    # The agent's own token and cost composition. Local and free - computed from
    # the traces, not from any judge - but it belongs to this run's output: the
    # efficiency shares are a statement about those tokens, and reading one
    # without the other invites comparing a percentage to nothing.
    # One token counter per agent, reused across that agent's cells: it holds
    # a disk cache keyed by the text, so the second cell of a model pays
    # nothing for text it has already counted.
    counters: dict = {}

    print(f"\n  ── token composition ({a.agent}) {'─' * 34}", flush=True)
    for comp in agent_tokens.compose(a.agent):
        if a.benchmark and comp.benchmark != a.benchmark:
            continue
        if a.domain and comp.domain != a.domain:
            continue
        agent_tokens.save([comp])
        # split.png, not pies.png. token_cost_pies split the bill five ways -
        # three input tiers, visible, reasoning - which answers how caching is
        # priced. Directly above an audit whose every denominator is one of
        # three things, the reader had to pool the five in their head. Drawn
        # again below with the audited sample's cost once that is known.
        print(f"     {comp.benchmark}/{comp.domain:<10}{comp.total_tokens:>14,} tok"
              f"  ${comp.total_cost:>9,.2f}  reprice error "
              f"{comp.reprice_error * 100:.4f}%", flush=True)

    started = time.time()
    for bench, dom, rows, tasks in plan:
        for task in tasks:
            print(f"\n  ── {task}  {bench}/{dom}  ({len(rows)} trajectories) "
                  f"{'─' * 20}", flush=True)
            # Stamp which AGENT run these trajectories came from, and refuse if this
            # cell directory was already filled from a different one. See
            # paths.claim_cell: bumping --run is the fix, merging never is.
            # The index rows do not carry it - extraction never stored one - so the
            # source run comes from the CELL DEFINITION in preprocess.sources, where it
            # is pinned. Not from the local `cells` above: that one is a different
            # function of the same name returning (benchmark, domain, n, n) tuples.
            from llmaj.preprocess.sources import cells_for as _src_cells
            srcs = sorted({rid for c in _src_cells(a.agent)
                           if c.benchmark == bench and c.domain == dom
                           for rid in (c.run_ids or ())})
            if srcs:
                paths.claim_cell(
                    paths.cell_dir(task, a.judge, run_id, a.agent, bench, dom), srcs)
            j = Judge(a.judge, timeout=a.timeout)
            t = time.time()
            try:
                if task == "utility":
                    st = util_run.run(j, rows, run_id, overwrite=a.overwrite,
                                      workers=a.workers)
                    print(f"     {time.time() - t:.0f}s  {st.traces} labelled, "
                          f"{st.skipped} done before, {st.retried} retried, "
                          f"{st.no_reasoning} no reasoning, {st.turns} turns, "
                          f"{st.errors} errors", flush=True)
                    # The transition matrix belongs to THIS branch:
                    # `utility_matrix.build` reads the utility tree only.
                    m = utility_matrix.build(a.judge, run_id, a.agent, bench, dom)
                    if m.n_traces:
                        utility_matrix.save(m)
                        print(f"     -> {paths.cell_dir('utility', a.judge, run_id, a.agent, bench, dom)}")
                elif task == "action":
                    from llmaj.evaluate import action as action_run
                    st = action_run.run(j, rows, run_id, overwrite=a.overwrite,
                                        workers=a.workers)
                    print(f"     {time.time() - t:.0f}s  {st.traces} labelled, "
                          f"{st.skipped} done before, {st.retried} retried, "
                          f"{st.no_output} sent nothing, {st.turns} turns, "
                          f"{st.errors} errors", flush=True)
                else:
                    st = eff_run.run(j, rows, run_id, overwrite=a.overwrite,
                                     progress=True, workers=a.workers)
                    print(f"     {time.time() - t:.0f}s  {st.traces} judged, "
                          f"{st.skipped} done before, {st.retried} retried, "
                          f"{st.verdicts} verdicts, "
                          f"{st.misaligned} misaligned, {st.errors} problems, "
                          f"{st.spans} spans ({st.spans_unverified} unverified)"
                          + (f", {st.repeats_injected} repeated calls the "
                             f"judge missed and we added"
                             if st.repeats_injected else ""),
                          flush=True)
                    sp = efficiency_spend.build(a.judge, run_id, a.agent, bench, dom)
                    if sp.n_traces:
                        efficiency_spend.save(sp)
                        print(efficiency_spend.table(sp), flush=True)
                        # The input side, counted mechanically over the SAME
                        # trajectories the judge saw - not the whole cell, or the
                        # two halves would be shares of different populations.
                        cw = context_waste.ContextWaste()
                        for row, turns in sample.iter_selected(rows):
                            context_waste.add(cw, row["trace_id"], turns)
                            # A re-check the judge called removable cost input
                            # tokens too; the output side already counts it.
                            context_waste.add_judged_rechecks(
                                cw, row["trace_id"], turns,
                                eff_run.verdicts_for(a.judge, run_id, a.agent,
                                                     bench, dom,
                                                     row["trace_id"]))
                        # Count the waste in TOKENS before anything is
                        # priced. This has to happen here, inside the same
                        # pass, and not in a separate script run afterwards:
                        # when it lived outside, `dollars` and the figures
                        # front page from this one, and the two pages
                        # disagreed about the same cell with nothing
                        # complaining. Cached by text, so a re-run of a cell
                        # already counted costs no calls.
                        counter = counters.get(a.agent)
                        if counter is None:
                            counter = exact_tokens.Counter(a.agent)
                            counters[a.agent] = counter
                        pr = dollars.prices_for(a.agent, bench, dom)
                        groups = ((turns,
                                   eff_run.verdicts_for(a.judge, run_id,
                                                        a.agent, bench, dom,
                                                        row["trace_id"]))
                                  for row, turns in sample.iter_selected(rows))
                        ex_cell = exact_tokens.build(
                            a.agent, bench, dom, counter, groups,
                            pr.output if pr else 0.0, workers=a.workers)
                        exact_tokens.save(ex_cell, a.judge, run_id)
                        counter.save()
                        ex_json = ex_cell.to_json()
                        print(exact_tokens.table(ex_cell), flush=True)
                        w = dollars.build(a.agent, bench, dom,
                                          {r["trace_id"] for r in rows},
                                          exact=ex_json)
                        print("", context_waste.table(cw), sep="\n", flush=True)
                        print("", dollars.table(w), sep="\n", flush=True)
                        dollars.save(w, a.judge, run_id, a.agent, bench, dom)
                        context_waste.save(cw, a.judge, run_id, a.agent, bench, dom)
                        print(f"     -> {paths.cell_dir('efficiency', a.judge, run_id, a.agent, bench, dom)}")
            except KeyboardInterrupt:
                print("\n  interrupted. Re-run the same command to resume.\n")
                return 130
            except Exception as e:
                # One bad cell must not take the session down: the others are
                # independent, and a partial result is still a result.
                print(f"     FAILED: {e.__class__.__name__}: {e}", flush=True)
            finally:
                j.close()


    print(f"\n{'=' * 74}")
    print(f"  done in {(time.time() - started) / 60:.0f} min")
    print(f"  verdicts and per-cell aggregates   {paths.RESULTS_OUT}")
    print(f"  agent token composition            {paths.AGENT_SPEND}")
    print(f"  judge spend                        {paths.JUDGE_SPEND}")
    print(f"{'=' * 74}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
