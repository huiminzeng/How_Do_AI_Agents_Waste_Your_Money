"""Which group a task belongs to, for topping up a random sample.

The sample is drawn at random and then TOPPED UP, not stratified from the start.
That ordering is the point: the random draw stays intact, so the plain average
over it is still an unbiased estimate of the population, and no reweighting is
needed for the headline number. The top-ups are extra rows, marked as such, and
exist only so a subgroup question has enough rows to answer it.

Stratifying instead of topping up would have forced the opposite trade: every
reported share would need weighting back to the population, and any error in the
weights would move the headline.

TWO GROUPINGS, and the data supports them unevenly.

  difficulty   how many of the four models solved the task. Available for tau2
               wherever at least MIN_GRADED models are graded on the task
               (tau2 rewards, SWE-bench reports, MCP-Atlas coverage, claw-eval
               task scores - see grade_sources.grades).

  length       assistant turns, from the index. Available everywhere.

Measured: a random 30 already covers most groups adequately. What it
misses is small and specific - tau2/retail drew ZERO of the two tasks all four
models failed, and 1 of the 4 shortest trajectories. Those are the cells a
subgroup claim would otherwise rest on nothing.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
from pathlib import Path

from .. import paths, prices

#: Mean reward at or above this counts as "the model solved this task". tau2's
#: reward is 0 or 1 per trial and every task ran 4 trials, so this is "won at
#: least half its attempts" rather than a threshold on a continuous score.
SOLVED_AT = 0.5

#: Assistant-turn boundaries for the length groups. Chosen from the observed
#: distribution rather than round numbers: the tau2 median is 11-12 turns, so
#: these put roughly a fifth of tasks in each tail.
LENGTH_BOUNDS = ((0, 7, "short"), (8, 15, "medium"), (16, 10 ** 9, "long"))

#: Minimum tasks per group before a subgroup claim is worth making. Not a
#: statistical threshold - 8 paired tasks across 4 models is 32 trajectories,
#: which is enough to see a 2x difference and not enough to see a 10% one.
MIN_PER_GROUP = 8

#: Models that must be graded on a task before it gets a difficulty class.
MIN_GRADED = 3


def length_group(n_turns: int) -> str:
    for lo, hi, name in LENGTH_BOUNDS:
        if lo <= n_turns <= hi:
            return name
    return "long"


def tau2_simulations(model: str, domain: str) -> list[dict]:
    """Every simulation of this model on this domain, across ALL its runs.

    A model run twice (the first 38 airline tasks in one run, the 12 missing
    ones later) has two results.json files; newest-wins read the second and
    lost the first. Merged by simulation id, the later file
    winning a duplicate.
    """
    by_id: dict = {}
    for f in _tau2_reward_files_all().get((model, domain), []):
        try:
            sims = (json.loads(Path(f).read_text()) or {}).get("simulations") or []
        except (OSError, ValueError):
            continue
        for s in sims:
            by_id[str(s.get("id") or (s.get("task_id"), s.get("trial"), f))] = s
    return list(by_id.values())


_RUN_NAME = re.compile(r"_(?P<domain>[a-z]+)_llm_agent_(?P<alias>.+?)_user_simulator_")


def _tau2_reward_files_all() -> dict[tuple[str, str], list[str]]:
    """Every results.json per (alias, domain) under benchmarks/tau2-bench, oldest first.

    tau2 names a run `<stamp>_<domain>_llm_agent_<alias>_user_simulator_<user model>`;
    the alias is the model name the cell carries (preprocess/sources.py names
    cells by the alias the harness asked for), with any provider prefix dropped.
    """
    out: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for f in glob.glob(str(paths.TAU2_SIMULATIONS / "*/results.json")):
        m = _RUN_NAME.search(Path(f).parent.name)
        if not m:
            continue
        key = (m.group("alias").split("/")[-1], m.group("domain"))
        out.setdefault(key, []).append((os.path.getmtime(f), f))
    return {k: [f for _, f in sorted(v)] for k, v in out.items()}


def _tau2_reward_files() -> dict[tuple[str, str], str]:
    """Newest results.json per (alias, domain)."""
    return {k: v[-1] for k, v in _tau2_reward_files_all().items() if v}


def task_type_groups(benchmark: str, domain: str) -> dict[str, str]:
    """task_id -> deliverable shape, for benchmarks whose tasks are not alike.

    claw_eval only. Its 161 tasks span 24 `category` values, but the grouping
    that matters is the SHAPE of what was to be produced, and three groups have
    a shape the user's own request does not state - see preprocess.claw_tasks.

    Why this needs to top up rather than be left to the random draw: the shapes
    are wildly uneven in size and in output volume. `execute` is 5 tasks of 161,
    and the one drawn in a test sample carried 45,338 characters against
    8,260-12,403 for the others - four times the weight. A cell share is
    character-weighted, so a single task of that shape moves the whole cell's
    number, and with one row there is nothing to say about it. Topping the group
    up does not stop it moving the number; it makes the movement explainable.

    Everything not in one of the three named shapes is `general`, which is 149
    of the 161 and needs no help.
    """
    if benchmark != "claw_eval":
        return {}
    from .claw_tasks import categories, SHAPE_OF_CATEGORY
    return {task: SHAPE_OF_CATEGORY.get(cat, "general")
            for task, cat in categories().items()}


def difficulty_groups(benchmark: str, domain: str,
                      agents: tuple[str, ...]) -> dict[str, str]:
    """task_id -> "all-failed" | "mixed" | "all-solved", or {} if unavailable.

    A property of the TASK, not of one trajectory: how many of the models
    graded on this benchmark solved it (a model "solved" a task when at least
    half its trials did - tau2 runs 4, claw_eval 3, the others 1). This
    covers every benchmark through `grade_sources.grades`, and
    the grouping is over the models in `agents` that HAVE grades; a task
    ungraded for any of those models is left out rather than guessed. Empty
    is a real answer when fewer than two models are graded.
    """
    from . import grade_sources as scores
    graded = {}
    for model in agents:
        g = scores.grades(model, benchmark, domain)
        if g:
            graded[model] = g
    if len(graded) < 2:
        return {}
    # Per task, over the models graded ON THAT TASK (at least MIN_GRADED of
    # them). A strict intersection over every model shrank the classified set
    # to the 54 tasks the newest models had run; a task three of six models
    # were graded on is still classifiable.
    tasks = set().union(*(set(g) for g in graded.values()))
    out = {}
    for task in tasks:
        gs = [g[task] for g in graded.values() if task in g]
        if len(gs) < MIN_GRADED:
            continue
        solved = sum(1 for v in gs if v >= SOLVED_AT)
        out[task] = ("all-failed" if solved == 0
                     else "all-solved" if solved == len(gs) else "mixed")
    return out
