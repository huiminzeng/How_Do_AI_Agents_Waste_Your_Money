"""Stage 3: drive one judge over a set of trajectories.

Resumability is structural: a shard that exists is done, so an interrupted run
resumes with no progress file to keep in sync.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import paths
from .judge import Judge, trace_meta_for
from ..prompts.utility_prompt import PROMPT_VERSION, build, snapshot
from ..preprocess.sample import load_turns
from ..schema import Label, Turn


@dataclass
class LabelStats:
    traces: int = 0
    skipped: int = 0
    turns: int = 0
    errors: int = 0
    #: Selected, but carrying no reasoning text at all - nothing for this pass to
    #: read. Counted rather than silently dropped.
    no_reasoning: int = 0
    #: Shards that existed but held failures, so were labelled again.
    retried: int = 0


def _shard(row: dict, judge_id: str, run_id: str):
    return paths.verdict_shard(
        paths.UTILITY_TASK, judge_id, run_id, row["agent_model"], row["benchmark"],
        row["domain"], row["trace_id"],
    )


def _split_shard(path) -> tuple[list[Label], set[int]]:
    """An existing shard, split into labels worth keeping and rounds to redo.

    Only the failures are re-sent. Re-labelling the whole trajectory would be
    simpler and, on a 30-turn shard with one failure, would pay 30 times for one
    missing answer.

    A label from a DIFFERENT prompt version is a failure for this purpose, and
    that is the load-bearing part. `run_id` is date-based and stable within a
    day, and the shard path does not carry the prompt version, so without this a
    prompt change followed by a same-day re-run reports "done before" for every
    trajectory: the new prompt never goes out, and the summary is built from
    answers to the old question with nothing saying so.

    Mixed versions inside one shard are not merged. If any label is from another
    version the whole trajectory is redone, because a matrix built half from one
    prompt and half from another is a matrix of nothing - and prompt wording was
    measured to move 39% of labels.
    """
    keep: list[Label] = []
    bad: set[int] = set()
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if any(d.get("prompt_version") != PROMPT_VERSION for d in rows):
        return [], {d["round"] for d in rows}
    for d in rows:
        if d.get("error") or not d.get("state"):
            bad.add(d["round"])
        else:
            keep.append(Label.from_json(d))
    return keep, bad


def label_trace(
    judge: Judge,
    row: dict,
    turns: list[Turn],
    run_id: str,
    task: str | None,
    workers: int = 1,
) -> list[Label]:
    """Label one trajectory.

    Turns inside a trajectory are independent classification calls, so they run
    concurrently. Sequentially this is 4s x 25,539 turns = 28 hours per judge;
    the judges are the wall clock here, not us.
    """
    todo = [t for t in turns if t.has_text]
    if not todo:
        return []
    # Passed per call rather than held on the Judge: the same Judge labels
    # several cells in one run, and threads share it.
    meta = trace_meta_for(judge.judge_id, run_id, PROMPT_VERSION,
                          row["agent_model"], row["benchmark"], row["domain"],
                          task="utility")
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(lambda t: judge.label(build(t, task), meta), todo))
    else:
        verdicts = [judge.label(build(t, task), meta) for t in todo]

    out: list[Label] = []
    for t, v in zip(todo, verdicts):
        out.append(
            Label(
                trace_id=row["trace_id"],
                round=t.round,
                judge_id=judge.judge_id,
                judge_run_id=run_id,
                prompt_version=PROMPT_VERSION,
                state=v.state or "",
                confidence=v.confidence,
                rationale=v.rationale,
                error=v.error,
            )
        )
    return out


def run(
    judge: Judge,
    rows: list[dict],
    run_id: str,
    overwrite: bool = False,
    progress: bool = True,
    workers: int = 8,
) -> LabelStats:
    # Before the first call, not after the last: a run that dies halfway still
    # leaves behind what it was asking. Raises if this version was reused for
    # different wording, which is the failure this exists to prevent.
    snapshot()
    stats = LabelStats()
    for i, row in enumerate(rows, 1):
        keep: list[Label] = []
        retry_rounds: set[int] | None = None
        path = _shard(row, judge.judge_id, run_id)
        if path.exists() and not overwrite:
            # A shard that exists is done ONLY if every label in it is usable.
            # Treating any existing shard as finished made errors permanent: one
            # session left 37 of 48 labels as errors, and re-running skipped
            # every one of them. Retrying costs a call; not retrying costs the
            # label forever.
            keep, bad = _split_shard(path)
            if not bad:
                stats.skipped += 1
                continue
            stats.retried += 1
            print(f"  {row['trace_id'][:16]}: re-labelling {len(bad)} of "
                  f"{len(bad) + len(keep)} rounds"
                  + ("" if keep else f" (all of them - shard is from another "
                                     f"prompt version)"), flush=True)
            retry_rounds = bad
        # NOTE: no task statement yet. `task_id` is an opaque identifier in
        # most cells ("2", a hash), and showing that to the judge is worse than
        # showing nothing. Extracting the benchmark's task text is future work;
        # until then the judge relies on the preceding observation, which
        # usually carries the request.
        turns = load_turns(row)
        # No reasoning anywhere in this trajectory means there is nothing this
        # pass can read. Skip it rather than write an empty shard: an empty shard
        # counts as "done" and would make the run look like it covered a
        # trajectory it could not.
        if not any(t.has_text for t in turns):
            stats.no_reasoning += 1
            continue
        if retry_rounds is not None:
            turns = [t for t in turns if t.round in retry_rounds]
        labels = label_trace(judge, row, turns, run_id, None, workers)
        # Keep the good labels from the previous attempt, in round order.
        merged = sorted(keep + labels, key=lambda l: l.round)
        paths.ensure_parent(path)
        with path.open("w") as fh:
            for lab in merged:
                fh.write(json.dumps(lab.to_json()) + "\n")
        stats.traces += 1
        stats.turns += len(labels)
        stats.errors += sum(1 for lab in labels if lab.error)
        if progress and i % 5 == 0:
            print(
                f"  {i}/{len(rows)} traces | {stats.turns} turns | "
                f"{stats.errors} errors",
                flush=True,
            )
    return stats


def iter_labels(judge_id: str, run_id: str, agent_model: str = "*",
                benchmark: str = "*", domain: str = "*"):
    """Read back every label for a (judge, run), optionally filtered."""
    root = (paths.RESULTS_OUT / paths.UTILITY_TASK
            / paths.judge_seg(judge_id) / run_id)
    if not root.exists():
        return
    agent_pat = ("*" if agent_model == "*"
                 else paths.agent_seg(agent_model))
    cell_pat = ("*" if benchmark == "*"
                else paths.cell_seg(benchmark, domain if domain != "*" else "all"))
    for path in sorted(root.glob(f"{agent_pat}/{cell_pat}/*.jsonl")):
        parts = path.relative_to(root).parts  # agent/bench/domain/<trace>.jsonl
        with path.open() as fh:
            rows = [Label.from_json(json.loads(l)) for l in fh if l.strip()]
        bench, dom = paths.split_cell_seg(parts[1])
        yield {"agent_model": paths.strip_role(parts[0]), "benchmark": bench,
               "domain": dom, "labels": rows}
