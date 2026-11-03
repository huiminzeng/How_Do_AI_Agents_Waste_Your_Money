"""Every filesystem location the package uses.

This is the ONLY module allowed to contain a path literal. Everything else asks
here. That makes "can this write somewhere it shouldn't" a one-file review.

Five top-level directories, named for what a reader is looking for rather than
for the stage that produced them:

    input/         what we analyse        - extracted from the traces
    prompts/       what we asked          - one snapshot per version
    results/       what the judge said    - judgements plus their summary
    judge_spend/   what the judging cost

Extracted traces sit under input/, not beside the results: preprocessing output
is the analysis's INPUT. The aggregate for a cell sits beside the judgements it
came from, so everything about one cell is in one directory.

Invariant: we read the project, we write only under WORK.
"""

from __future__ import annotations

import os
from pathlib import Path

# LLMaJ/llmaj/paths.py -> LLMaJ/ -> project root
PKG_ROOT = Path(__file__).resolve().parent.parent
#: The repository root: where artifacts/ (the proxy's traces and this package's
#: output) and benchmarks/ (each benchmark's checkout and its results) live. Defaults to the directory
#: this package is checked out under; LLMAJ_DATA_ROOT points it elsewhere.
PROJECT_ROOT = Path(os.environ["LLMAJ_DATA_ROOT"]).resolve() if os.environ.get("LLMAJ_DATA_ROOT") else PKG_ROOT.parent

# ---------------------------------------------------------------- read-only

#: Everything this repository produces.
ARTIFACTS = PROJECT_ROOT / "artifacts"
#: Every call the tracing proxy recorded, one directory per trajectory.
MONITOR = ARTIFACTS / "trajectories"
#: The trace roots read, in order. One by default; a second root can be added
#: for traces copied in from elsewhere.
MONITOR_ROOTS = (MONITOR,)
#: Kept for callers that branch on a consolidated-file layout; nothing is
#: expected to exist here.
RESULTS = PROJECT_ROOT / "results"

READ_ONLY_ROOTS = MONITOR_ROOTS + (RESULTS,)

#: The benchmarks' own outputs, as benchmarks/<name>/run.sh leaves them. Grades
#: come from here (preprocess/grade_sources.py, preprocess/outcome_sources.py).
BENCHMARKS = PROJECT_ROOT / "benchmarks"
TAU2_SIMULATIONS = BENCHMARKS / "tau2-bench/src/data/simulations"
CLAW_ROOT = BENCHMARKS / "claw-eval/src"


def swe_reports(name: str) -> list[Path]:
    """SWE-bench evaluation reports for a model alias (benchmarks/mini-swe-agent/grade.sh)."""
    return sorted((BENCHMARKS / "mini-swe-agent/src").glob(f"openai__{name}.*.json"))


def mcp_scored(name: str) -> list[Path]:
    """MCP-Atlas claim-coverage files for a model alias (benchmarks/mcp-atlas/score.sh)."""
    return sorted((BENCHMARKS / "mcp-atlas/src").glob(f"scoring_*/scored_{name}.csv"))


def claw_batches(name: str) -> list[Path]:
    """claw-eval batch results for a model alias (one directory per batch)."""
    return sorted((CLAW_ROOT / "traces").glob(f"{name}_*/batch_results.json"))

# --------------------------------------------------------------- write-only

#: Everything this package writes lives under here and `ensure_parent` enforces
#: it, beside the trajectories it was computed from.
WORK = ARTIFACTS / "llmaj"

INPUT = WORK / "input"
TRACES = INPUT / "traces"
TRACES_INDEX = TRACES / "index.jsonl"
AGENT_SPEND = INPUT / "agent_spend"
#: The standing instructions each cell's agents were given. One file per
#: cell, identical across the four agents, so it lives under input/ beside
#: the traces rather than under a judge's run.
POLICIES = INPUT / "policies"

PROMPTS = WORK / "prompts"
RESULTS_OUT = WORK / "results"
JUDGE_SPEND = WORK / "judge_spend"

#: The three classification tasks. Used as a path segment under results/ and
#: figures/, and as part of the bucket the proxy files judge calls under.
#:
#: `utility` and `action` ask the SAME question of the SAME turn from two sides:
#: utility reads the chain of thought with the action hidden, action reads the
#: action and reply with the thought hidden. Separate tasks, not two prompt
#: versions of one, because they must be stored side by side and compared - a
#: version stamp namespaces text, it does not namespace storage.
UTILITY_TASK = "utility"
ACTION_TASK = "action"
EFFICIENCY_TASK = "efficiency"

#: Two model names sit in every result path - the judge and the agent it judged -
#: and bare, they are indistinguishable. `kimi-k3/.../gemini-3.1-pro-preview`
#: reads equally well in either direction, and both models appear in both roles
#: across the tree. The suffixes make the role part of the name.
JUDGE_SUFFIX = "_llmaj"
AGENT_SUFFIX = "_agent"


def judge_seg(judge_id: str) -> str:
    return f"{judge_id}{JUDGE_SUFFIX}"


def agent_seg(agent_model: str) -> str:
    return f"{agent_model}{AGENT_SUFFIX}"


def cell_seg(benchmark: str, domain: str) -> str:
    """One segment for a cell: "tau2-airline", or just "mcp_atlas".

    `domain` is "all" on every benchmark but tau2, so a separate level for it was
    a directory named "all" three times out of four - depth carrying no
    information. Joined on "-", which no benchmark name contains.
    """
    if "-" in benchmark:
        raise ValueError(
            f"benchmark {benchmark!r} contains '-', which is the separator this "
            f"segment is parsed on. Rename it or change the separator."
        )
    return benchmark if domain == "all" else f"{benchmark}-{domain}"


def split_cell_seg(seg: str) -> tuple[str, str]:
    """Inverse of `cell_seg`."""
    benchmark, sep, domain = seg.partition("-")
    return benchmark, (domain if sep else "all")


def strip_role(seg: str) -> str:
    """A judge or agent segment with its role suffix removed."""
    for suffix in (JUDGE_SUFFIX, AGENT_SUFFIX):
        if seg.endswith(suffix):
            return seg[: -len(suffix)]
    return seg


# ------------------------------------------------------------------- input

def trace_shard(agent_model: str, benchmark: str, domain: str,
                trace_id: str) -> Path:
    """One trajectory's extracted turns.

    No role suffix here: input/ holds only agent traces, so there is no second
    model name to confuse it with.
    """
    return TRACES / agent_model / cell_seg(benchmark, domain) / f"{trace_id}.jsonl"


def agent_spend_dir(agent_model: str, benchmark: str, domain: str) -> Path:
    """What the AGENT spent on one cell: counts and repriced dollars.

    Data only. The pies drawn from it live under figures/ - they were written
    here at first, which put 17 PNGs inside a directory named `input/`.
    """
    return AGENT_SPEND / agent_model / cell_seg(benchmark, domain)


# ----------------------------------------------------------------- prompts

def prompt_dir(task: str, prompt_version: str) -> Path:
    """Snapshot of the prompt text a (task, version) refers to.

    `task` is needed because every task numbers from v1 and would otherwise write
    the same file. Written once and never rewritten while any verdict references
    it - see `prompts.utility_prompt.snapshot_texts`.
    """
    return PROMPTS / task / prompt_version


# ----------------------------------------------------------------- results

def cell_dir(task: str, judge_id: str, judge_run_id: str, agent_model: str,
             benchmark: str, domain: str) -> Path:
    """Everything one (task, judge, run, cell) produced.

    Both the per-trajectory judgements and the summary built from them, in one
    directory. `judge_run_id` is part of the path for the same reason the judge
    is: a relabel under revised rules must not overwrite the original. It was
    missing once, and two runs of the same cell silently overwrote each other.
    """
    return (RESULTS_OUT / task / judge_seg(judge_id) / judge_run_id
            / agent_seg(agent_model) / cell_seg(benchmark, domain))


def claim_cell(cell_path: "Path", source_run_ids) -> None:
    """Record which SOURCE run this cell directory was judged from, and refuse a mismatch.

    `judge_run_id` already keeps two judgements of the same cell apart, but only if the
    operator remembers to change it. Reusing it after the underlying traces have been
    re-collected does not overwrite anything - trace ids differ, so the old and new
    verdicts simply sit side by side and the directory silently holds two populations.
    That is how two runs' verdicts would become one directory of twice the size.

    So each cell directory carries a `.source.json` naming the agent run(s) it was built
    from. A later write naming different ones is a mistake - almost always a judge run id
    that should have been bumped - and is refused with both lists rather than merged.
    """
    import json as _json
    want = sorted(source_run_ids or [])
    marker = cell_path / ".source.json"
    if marker.exists():
        have = sorted(_json.loads(marker.read_text()).get("source_run_ids") or [])
        if have and want and have != want:
            raise SystemExit(
                f"{cell_path} already holds verdicts judged from {have}, and this run "
                f"would add verdicts from {want}. Two populations in one directory read "
                f"as one. Pass a new --run instead of merging them."
            )
        return
    cell_path.mkdir(parents=True, exist_ok=True)
    marker.write_text(_json.dumps({"source_run_ids": want}, indent=2) + "\n")


def verdict_shard(task: str, judge_id: str, judge_run_id: str, agent_model: str,
                  benchmark: str, domain: str, trace_id: str) -> Path:
    """One trajectory's judgements from one judge."""
    return cell_dir(task, judge_id, judge_run_id, agent_model, benchmark,
                    domain) / f"{trace_id}.jsonl"


def summary_file(task: str, judge_id: str, judge_run_id: str, agent_model: str,
                 benchmark: str, domain: str) -> Path:
    """The numbers built from one cell's judgements.

    Named `summary.json` in both tasks, not `matrix.json` / `spend.json`: the
    name a reader needs is "the aggregate for this cell", and the task is already
    the first segment of the path.
    """
    return cell_dir(task, judge_id, judge_run_id, agent_model, benchmark,
                    domain) / "summary.json"


def ensure_parent(path: Path) -> Path:
    """Create the parent directory, refusing anything outside WORK."""
    resolved = path.resolve()
    if WORK.resolve() not in resolved.parents:
        raise ValueError(f"refusing to write outside {WORK}: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
