"""Pick trajectories to label.

Deliberately simple: selection reads only the index, and the unit selected is a
whole trajectory. Never a loose set of turns - a partially labelled trajectory
yields transitions that were never taken.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from typing import Iterator

from .. import paths
from .extract import iter_index
from ..schema import Turn


#: One definition, imported by `evaluate.py` for its --seed default.
#:
#: It was 0 here and 1 there, and nothing failed loudly: a verification script
#: calling select() without a seed got a different three trajectories than the
#: run had judged, and the mismatch read as "the selection is not
#: deterministic". It is - the two callers were just asking different questions.
DEFAULT_SEED = 1


def select(
    benchmark: str | None = None,
    domain: str | None = None,
    agent_model: str | None = None,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    min_labellable: int = 0,
    min_turns: int = 1,
    like_run: tuple[str, str] | None = None,
) -> list[dict]:
    """Index rows for the trajectories to label.

    `min_labellable` counts turns carrying REASONING text. It defaults to 0
    because the two tasks need different things: the utility pass cannot read a
    turn without reasoning, but the efficiency pass judges the visible reply and
    the tool calls, which are there regardless. gpt-5.6 and claude-sonnet-5 emit
    no reasoning at all, so a default of 2 selected zero of their trajectories -
    silently, as "0 traces" rather than as an error.

    One selection per cell, shared by both tasks, so the two can be read side by
    side. The utility pass then skips the trajectories it has nothing to read in.
    Raise `min_labellable` only when selecting FOR the utility pass alone.

    `like_run` is `(judge_id, run_id)`: take exactly the trajectories that run
    labelled, ignoring `limit` and `seed`. This is the only way to make two runs
    comparable by construction. Hash ranking keeps a recomputed sample mostly
    stable, but "mostly" is not a basis for attributing a change to a rule
    rewrite - if the pool grew, some of the difference is different data.
    """
    if like_run is not None:
        want = _trace_ids_of(*like_run, benchmark, domain, agent_model)
        rows = [r for r in iter_index() if r["trace_id"] in want]
        missing = want - {r["trace_id"] for r in rows}
        if missing:
            raise ValueError(
                f"{len(missing)} trajectories labelled by {like_run} are no "
                f"longer in the index, e.g. {sorted(missing)[:3]}. The runs "
                f"cannot be made comparable; re-extract or pick another run."
            )
        rows.sort(key=lambda r: r["trace_id"])
        return rows
    rows = [
        r
        for r in iter_index()
        if (benchmark is None or r["benchmark"] == benchmark)
        and (domain is None or r["domain"] == domain)
        and (agent_model is None or r["agent_model"] == agent_model)
        and r["n_labellable"] >= min_labellable
        and r["n_turns"] >= min_turns
    ]
    rows.sort(key=lambda r: r["trace_id"])
    if limit is not None and limit < len(rows):
        rows = _stable_subset(rows, limit, seed)
        rows.sort(key=lambda r: r["trace_id"])
    return rows


def _stable_tasks(task_ids, limit: int, seed: int) -> list:
    """The `limit` task_ids whose salted hashes sort first.

    Ranking TASKS rather than trajectories, and the same ranking for every model,
    which is what makes the four samples paired.
    """
    salt = str(seed).encode()
    return sorted(task_ids,
                  key=lambda t: hashlib.sha256(
                      salt + str(t).encode()).hexdigest())[:limit]


def select_tasks(
    benchmark: str,
    domain: str,
    agent_model: str,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    min_turns: int = 1,
    pair_across: tuple[str, ...] = (),
    top_up: bool = False,
    min_per_group: int | None = None,
    target: int | None = None,
    keep: set | None = None,
) -> list[dict]:
    """One trajectory per TASK, with the task set shared across models.

    `keep`: trace ids already judged in the run being extended.
    For a task that has one of them, THAT trajectory is taken whatever the
    hash says, so a later run adding a second trial of the same task cannot
    swap the judged trajectory out of the sample.

    `target`: grow the sample to this many trajectories with
    BALANCED top-ups - see `_balance` - after the paired base of `limit` tasks
    and the per-group minimums. Only ever added, never swapped: a trajectory in
    the base stays in the base. A model whose task pool is smaller than `limit`
    (a model still being run) does not constrain the paired task set; it is
    the one that has to catch up, and `plan()` says on which tasks.

    Two problems with sampling trajectories directly, both measured on this data:

    REPEATED TRIALS ARE NOT INDEPENDENT. tau2 ran every task 4 times - 200
    trajectories over 50 tasks on airline, 456 over 114 on retail. Ranking
    trajectories by hash therefore drew 30 trajectories covering only 25 tasks,
    so five of the thirty judge calls bought a second look at a task already in
    the sample. Trials of one task share its difficulty, its tools and the policy
    that constrains it; they are close to the "adjacent prefixes are highly
    correlated, equals paying twice" problem BAGEN describes for prefixes, and
    the fix is the same - stratify at the level the correlation lives at.
    mcp_atlas has one trial per task and is unaffected.

    THE FOUR MODELS DREW DIFFERENT TASKS. Task hashes are model-independent, so
    ranking tasks gives every model the same 30. Variance decomposition on output
    tokens per trajectory: 48% of the variance on tau2/airline is between tasks,
    30% on retail, 39% on mcp_atlas. Paired on task, that share leaves the
    model comparison entirely - worth roughly 1.4-1.9x the sample size, for the
    same number of calls.

    `pair_across` restricts the task pool to tasks every named model actually
    has, so a task missing for one model does not silently unpair the rest. Empty
    means "do not check", which is right when only one model is being selected.

    Which trial is taken is decided by the same hash, on the trace_id: stable,
    and independent of whatever order the index happens to be in. Trial NUMBER
    is not in the index, so "the first trial" is not available to ask for - and
    would be the wrong thing to ask for anyway, since trial 0 of a 4-trial task
    is not more representative than trial 2.
    """
    pool = [r for r in iter_index()
            if r["benchmark"] == benchmark and r["domain"] == domain
            and r["n_turns"] >= min_turns]
    mine = [r for r in pool if r["agent_model"] == agent_model]
    tasks = {r["task_id"] for r in mine}
    # Only models that HAVE this cell constrain the task set.
    #
    # Intersecting over every named model silently emptied swe_bench, which only
    # kimi-k3 was ever run on: the intersection with three absent models is the
    # empty set, and the cell reported "0 trajectories" as though there were no
    # data rather than no partner to pair with. A cell one model has is still
    # worth judging - it just cannot carry a cross-model comparison, and the
    # `paired_with` field on every row says which models it actually pairs to.
    present = [m for m in pair_across
               if m != agent_model
               and any(r["agent_model"] == m for r in pool)]
    if limit:
        # A model that has fewer tasks than the base asks for is still being
        # run; intersecting with it would shrink everyone's sample to its size.
        small = [m for m in present
                 if len({r["task_id"] for r in pool if r["agent_model"] == m}) < limit]
        present = [m for m in present if m not in small]
    for other in present:
        tasks &= {r["task_id"] for r in pool if r["agent_model"] == other}
    chosen = _stable_tasks(tasks, limit, seed) if limit else sorted(tasks)
    want = set(chosen)
    salt = str(seed).encode()
    keep = set(keep or ())
    by_task: dict = {}
    for r in mine:
        if r["task_id"] not in want:
            continue
        # A kept trajectory sorts before every hash.
        h = ("0" if r["trace_id"] in keep else "1") + hashlib.sha256(salt + r["trace_id"].encode()).hexdigest()
        cur = by_task.get(r["task_id"])
        if cur is None or h < cur[0]:
            by_task[r["task_id"]] = (h, r)
    out = [r for _, r in by_task.values()]
    for r in out:
        r["sampled"] = "base"
        # Recorded per row rather than assumed: a reader of the results should
        # not have to know which cells every model was run on to know whether a
        # comparison is paired.
        r["paired_with"] = sorted([agent_model] + present)
    if top_up and limit:
        out += _top_up(benchmark, domain, agent_model, mine, tasks,
                       {r["task_id"] for r in out}, seed, pair_across,
                       min_per_group)
    if target and len(out) < target:
        out += _balance(benchmark, domain, agent_model, mine, pool, out,
                        target, seed, pair_across, keep)
    out.sort(key=lambda r: r["trace_id"])
    return out


#: Balanced top-up floors, as shares of `target`. Each length third and each
#: difficulty class gets at least this share, and each outcome (solved,
#: failed) gets its own; a group the pool cannot fill is taken in full.
BALANCE_FLOORS = {"length": 0.20, "outcome": 0.25, "difficulty": 0.15}


def _trial_for(task, mine: list[dict], salt: bytes, keep: set = frozenset()) -> dict | None:
    """The trajectory `select_tasks` takes for a task: a kept (already judged)
    one if there is one, else the lowest salted hash."""
    cands = [r for r in mine if r["task_id"] == task]
    if not cands:
        return None
    return min(cands, key=lambda r: ("0" if r["trace_id"] in keep else "1")
               + hashlib.sha256(salt + r["trace_id"].encode()).hexdigest())


def _balance(benchmark: str, domain: str, agent_model: str, mine: list[dict],
             pool: list[dict], have: list[dict], target: int, seed: int,
             agents: tuple[str, ...], keep: set | None = None) -> list[dict]:
    """Add tasks until `target`, filling whichever group is furthest below its
    floor, on three dimensions of THIS cell's chosen trajectory:

      length      short / medium / long (strata.LENGTH_BOUNDS on turns)
      outcome     solved / failed, the trajectory's own result (preprocess.outcome_sources)
      difficulty  all-failed / mixed / all-solved, a property of the task
                  (how many graded models solve it; strata.difficulty_groups)

    Among the candidates for a group, tasks that MORE of the other models also
    have come first (then the hash), so the added tasks overlap across models
    as much as the pool allows. When no floor is short, the remaining slots are
    filled in hash order. Every added row is tagged `sampled = "top-up:<dim>=<group>"`
    or "fill"; the base rows are untouched.
    """
    from . import strata
    from .outcome_sources import outcome_for
    salt = str(seed).encode()
    chosen_tasks = {r["task_id"] for r in have}
    pool_tasks = sorted({r["task_id"] for r in mine})
    trial = {t: _trial_for(t, mine, salt, set(keep or ())) for t in pool_tasks}
    trial = {t: r for t, r in trial.items() if r is not None}
    # groups of every pool task, under the trajectory that WOULD be taken.
    # Length is RELATIVE to this cell's pool (terciles of turns), not the
    # fixed tau2 bounds: claw_eval runs are 3-8 turns and swe_bench 15-40, and
    # "long for this cell" is what the balance is about.
    turns = sorted(r["n_turns"] for r in trial.values())
    lo = turns[len(turns) // 3] if turns else 0
    hi = turns[(2 * len(turns)) // 3] if turns else 0
    length = {t: ("short" if r["n_turns"] < lo else "long" if r["n_turns"] >= hi else "medium")
              for t, r in trial.items()}
    oc = outcome_for(agent_model, benchmark, domain, [r["trace_id"] for r in trial.values()])
    outcome = {t: ("solved" if oc.get(r["trace_id"]) == 1.0 else
                   "failed" if oc.get(r["trace_id"]) == 0.0 else None)
               for t, r in trial.items()}
    diff = strata.difficulty_groups(benchmark, domain, agents)
    dims = {"length": length, "outcome": outcome,
            "difficulty": {t: diff.get(t) for t in trial}}
    # how many other models have each task: the overlap preference
    others = collections.Counter(r["task_id"] for r in pool if r["agent_model"] != agent_model)
    floors = {d: math.ceil(BALANCE_FLOORS[d] * target) for d in dims}

    def counts():
        c = {d: collections.Counter() for d in dims}
        for t in chosen_tasks:
            for d, m in dims.items():
                g = m.get(t)
                if g:
                    c[d][g] += 1
        return c

    def groups_of(d):
        return sorted({g for g in dims[d].values() if g})

    added = []
    while len(chosen_tasks) < target:
        c = counts()
        best = None  # (deficit, dim, group)
        for d in dims:
            for g in groups_of(d):
                deficit = floors[d] - c[d][g]
                if deficit > 0 and (best is None or deficit > best[0]):
                    if any(t not in chosen_tasks and dims[d].get(t) == g for t in trial):
                        best = (deficit, d, g)
        if best is None:
            cands = [t for t in trial if t not in chosen_tasks]
            tag = "fill"
        else:
            _, d, g = best
            cands = [t for t in trial if t not in chosen_tasks and dims[d].get(t) == g]
            tag = f"top-up:{d}={g}"
        if not cands:
            break
        cands.sort(key=lambda t: (-others.get(t, 0),
                                  hashlib.sha256(salt + str(t).encode()).hexdigest()))
        t = cands[0]
        chosen_tasks.add(t)
        r = dict(trial[t])
        r["sampled"] = tag
        r["paired_with"] = sorted(m for m in agents if others.get(t, 0) and any(
            x["agent_model"] == m and x["task_id"] == t for x in pool)) or [agent_model]
        added.append(r)
    return added


def plan(benchmark: str, domain: str, agents: tuple[str, ...], target: int,
         limit: int, seed: int = DEFAULT_SEED) -> dict:
    """What each model's `target`-sized sample would be today, and which tasks a
    model still has to RUN for the samples to overlap.

    Returns {"union": [task_ids], "per_model": {m: {"sample": n, "tasks": [...],
    "composition": {...}, "to_run": [...]}}}. `to_run` is the union of every
    model's sample minus the tasks this model has any trajectory for - the
    list to hand to the harness. Models with fewer than `limit` tasks are
    sampled too (over what they have) but do not constrain the others.
    """
    per = {}
    union: set = set()
    for m in agents:
        rows = select_tasks(benchmark, domain, m, limit, seed, pair_across=agents,
                            top_up=True, target=target)
        tasks = {r["task_id"] for r in rows}
        union |= tasks
        comp = collections.Counter(r["sampled"].split(":")[0] if ":" not in r["sampled"]
                                   else r["sampled"] for r in rows)
        per[m] = {"sample": len(rows), "tasks": sorted(tasks), "composition": dict(comp)}
    have = {m: {r["task_id"] for r in iter_index()
                if r["agent_model"] == m and r["benchmark"] == benchmark
                and (r["domain"] or "") == (domain or "")} for m in agents}
    for m in agents:
        per[m]["to_run"] = sorted(union - have[m])
        per[m]["pool_tasks"] = len(have[m])
    return {"benchmark": benchmark, "domain": domain, "target": target, "limit": limit,
            "union": sorted(union), "per_model": per}


def select_listed_tasks(
    benchmark: str,
    domain: str,
    agent_model: str,
    task_ids,
    seed: int = DEFAULT_SEED,
    min_turns: int = 1,
) -> list[dict]:
    """One trajectory per task in `task_ids`, the trial chosen by the same salted
    hash `select_tasks` uses.

    For RE-JUDGING a cell whose trajectories were re-run: the trace ids changed
    (a new run_id), so `like_run` cannot find them, but the TASKS the superseded
    judging covered are known and keeping them is what makes the new cell
    comparable to the one it replaces. The task lists live in
    work/input/sampled_tasks/rerun_<model>_<bench>.tasks.json.

    A task with no trajectory in the cell is reported, not silently dropped.
    """
    want = {str(t) for t in task_ids}
    mine = [r for r in iter_index()
            if r["agent_model"] == agent_model and r["benchmark"] == benchmark
            and (r["domain"] or "") == (domain or "") and r["n_turns"] >= min_turns
            and str(r["task_id"]) in want]
    salt = str(seed).encode()
    by_task: dict = {}
    for r in mine:
        h = hashlib.sha256(salt + r["trace_id"].encode()).hexdigest()
        cur = by_task.get(str(r["task_id"]))
        if cur is None or h < cur[0]:
            by_task[str(r["task_id"])] = (h, r)
    missing = sorted(want - set(by_task))
    if missing:
        print(f"  note: {len(missing)} listed task(s) have no trajectory in "
              f"{agent_model}/{benchmark}/{domain}: {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}")
    out = [r for _, r in by_task.values()]
    for r in out:
        r["sampled"] = "base"
        r["paired_with"] = [agent_model]
    out.sort(key=lambda r: r["trace_id"])
    return out


def _top_up(benchmark: str, domain: str, agent_model: str, mine: list[dict],
            all_tasks: set, chosen: set, seed: int,
            agents: tuple[str, ...], min_per_group: int | None) -> list[dict]:
    """Extra tasks so every group reaches `min_per_group`, marked as top-ups.

    ADDED to the random draw, never substituted into it. The base sample keeps
    its distribution, so the plain average over `sampled == "base"` is still an
    unbiased estimate of the population and needs no reweighting; the top-ups
    exist so a subgroup claim rests on rows rather than on two of them.

    Measured on a random 30: tau2/retail drew ZERO of the two tasks
    all four models failed and 1 of the 4 shortest trajectories. Those are the
    cells this fills. Most groups need nothing - the total top-up is 6 tasks on
    airline and 6 on retail, and a group with fewer tasks than the minimum is
    taken in full rather than reported as short.

    A group whose data is unavailable is skipped, not guessed at: mcp_atlas has
    no scoring file for claude-sonnet-5, so difficulty is absent there and only
    length tops up.
    """
    from . import strata
    floor = strata.MIN_PER_GROUP if min_per_group is None else min_per_group
    lens = {r["task_id"]: r["n_turns"] for r in mine}
    groupings = {"length": {t: strata.length_group(lens.get(t, 0))
                            for t in all_tasks}}
    diff = strata.difficulty_groups(benchmark, domain, agents)
    if diff:
        groupings["difficulty"] = {t: diff[t] for t in all_tasks if t in diff}
    # claw_eval's tasks differ in what they were meant to PRODUCE, and three of
    # the shapes are 2-5 tasks out of 161. Without this the draw takes about one
    # of each, which is enough to move the cell's size-weighted share and
    # not enough to explain it.
    kind = strata.task_type_groups(benchmark, domain)
    if kind:
        groupings["type"] = {t: kind[t] for t in all_tasks if t in kind}

    salt = str(seed).encode()
    extra: set = set()
    why: dict = {}
    for name, mapping in groupings.items():
        members = collections.defaultdict(list)
        for task, g in mapping.items():
            members[g].append(task)
        for g, tasks_in in sorted(members.items()):
            have = sum(1 for t in tasks_in if t in chosen or t in extra)
            need = floor - have
            if need <= 0:
                continue
            rest = sorted(
                (t for t in tasks_in if t not in chosen and t not in extra),
                key=lambda t: hashlib.sha256(salt + str(t).encode()).hexdigest())
            for t in rest[:need]:
                extra.add(t)
                why[t] = f"{name}={g}"

    rows = []
    for task in extra:
        cands = sorted(
            (r for r in mine if r["task_id"] == task),
            key=lambda r: hashlib.sha256(salt + r["trace_id"].encode()).hexdigest())
        if cands:
            r = dict(cands[0])
            r["sampled"] = f"top-up:{why[task]}"
            rows.append(r)
    return rows


def _stable_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """The `limit` trajectories whose salted trace_id hashes sort first.

    NOT `random.sample`: its result depends on the size of the population, so
    the same --limit and --seed select a different set once the pool changes.
    Measured - growing a 500-trajectory pool by 20 replaced 28 of the 30 picks,
    silently. Two prompt versions "compared" across such a change are compared
    on different data, with nothing in the output to say so.

    Hashing depends only on the trace_id. Add a thousand trajectories and every
    previous pick is still selected; a new one enters only if its hash sorts
    ahead, and which ones those are is computable rather than a matter of luck.
    """
    salt = str(seed).encode()

    def rank(row: dict) -> str:
        return hashlib.sha256(salt + row["trace_id"].encode()).hexdigest()

    return sorted(rows, key=rank)[:limit]


def _trace_ids_of(judge_id: str, run_id: str, benchmark: str | None,
                  domain: str | None, agent_model: str | None) -> set[str]:
    """Which trajectories a finished run actually labelled.

    Read from the shard filenames, not from a manifest: the shards ARE the
    record of what was done, so this cannot drift from it.
    """
    root = paths.RESULTS_OUT / judge_id / run_id
    if not root.exists():
        raise ValueError(f"no such run: {root}")
    pattern = f"{agent_model or '*'}/{benchmark or '*'}/{domain or '*'}/*.jsonl"
    ids = {p.stem for p in root.glob(pattern)}
    if not ids:
        raise ValueError(f"{root} has no shards matching {pattern}")
    return ids


def load_turns(row: dict) -> list[Turn]:
    """Read one trajectory's shard back."""
    path = paths.trace_shard(
        row["agent_model"], row["benchmark"], row["domain"], row["trace_id"]
    )
    with path.open() as fh:
        return [Turn.from_json(json.loads(line)) for line in fh if line.strip()]


def iter_selected(rows: list[dict]) -> Iterator[tuple[dict, list[Turn]]]:
    for row in rows:
        yield row, load_turns(row)
