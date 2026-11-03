"""Build the text sent to a judge for one turn.

Enforces the show/withhold rule of TAXONOMY.md 3.1. The withholding is the part
that matters and it is enforced structurally: this module receives a Turn and
emits a string, and it simply never reads `round`, the benchmark, or anything
about what happened after the turn.

PROMPT_VERSION is stamped onto every label. Change the wording, bump it - a
relabel under different wording must be distinguishable from the original.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Turn
from .utility_states import BOUNDARY_RULES, DEFINITIONS, STATES

#: Stamped on every verdict; change the wording, bump it - two different texts
#: must never share one stamp, however cosmetic the difference.
PROMPT_VERSION = "v1"

_SYSTEM = """You classify the internal reasoning of an AI agent.

The agent is working through a task. You are given one turn of its private
reasoning, plus the observation it had just received before producing that
reasoning. Decide which single state the turn belongs to.

Classify what the reasoning is about, not what the agent did next. A turn that
spends most of its words making sense of a result belongs to that activity even
if it closes by naming the next call. The next action is recorded separately;
you are not being asked to infer it.

States
{states}

Rules
{rules}

Answer with JSON only:
{{"state": "<one of the eight>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}"""

_USER = """Task
{task}

The observation it had just received
{observation}

Its reasoning
{reasoning}"""


def system_prompt() -> str:
    states = "\n".join(f"{name:<10} {DEFINITIONS[name]}" for name in STATES)
    rules = "\n".join(f"- {r}" for r in BOUNDARY_RULES)
    return _SYSTEM.format(states=states, rules=rules)


def user_prompt(turn: Turn, task: str | None) -> str:
    """The per-turn half.

    `task` is a one-line statement of what the agent was asked to do. None is
    allowed and renders as "(not available)" rather than being omitted, so the
    judge sees a consistent shape either way.
    """
    return _USER.format(
        task=task or "(not available)",
        observation=turn.prev_observation or "(nothing - this is the first turn)",
        reasoning=turn.reasoning_content,
    )


def build(turn: Turn, task: str | None = None) -> list[dict[str, str]]:
    """Chat messages for one labelling call.

    Withheld on purpose, per TAXONOMY.md 3.1: the outgoing tool call, the
    visible reply, success or failure, the round number, and the benchmark's
    identity. The last two would let the judge label from position or from an
    expectation about the environment, and both are results we intend to
    measure rather than assume.
    """
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(turn, task)},
    ]


class PromptVersionConflict(RuntimeError):
    """A version was reused for different text.

    Raised rather than warned. Silently rewriting the snapshot would leave two
    incomparable batches of labels wearing the same version stamp, and the
    damage is only visible weeks later, when the text that produced the earlier
    batch is already gone.
    """


#: The files a snapshot consists of. Together with a Turn they reproduce, byte
#: for byte, every message any judge was sent under this version.
_SNAPSHOT = {
    "system.txt": lambda: system_prompt(),
    "user_template.txt": lambda: _USER,
}


def records_using(task: str, version: str) -> int:
    """How many of THIS TASK's verdicts are stamped with this prompt version.

    Per task, because the version namespaces are per task: utility v1 and
    efficiency v1 are different texts under prompts/<task>/v1/, and a guard
    that counted both trees would fire on a file it has nothing to do with.

    Zero means the version stamps nothing, and its snapshot can be rewritten
    without stranding a record - see `snapshot_texts`.
    """
    from .. import paths

    n = 0
    root = paths.RESULTS_OUT / task
    if root.exists():
        for shard in root.rglob("*.jsonl"):
            with shard.open() as fh:
                for line in fh:
                    if line.strip() and f'"{version}"' in line:
                        n += json.loads(line).get("prompt_version") == version
    return n


def snapshot_texts(task: str, version: str, texts: dict[str, str],
                   where: str) -> Path:
    """Write `texts` under `version`. Shared by every prompt in the package.

    Called before labelling, not after: a run that dies halfway still leaves
    behind what it was asking. Re-running with unchanged text is a no-op, so it
    is safe to call on every invocation.

    Changed text under an existing version is refused ONLY while that version
    has verdicts on disk. The guard exists so two different texts never wear one
    stamp; a version nothing references stamps nothing, so rewriting it strands
    no record. Making that an explicit check rather than a hand-waved exception
    is the point.
    """
    from .. import paths

    out = paths.prompt_dir(task, version)
    for name, text in texts.items():
        path = paths.ensure_parent(out / name)
        if path.exists():
            if path.read_text() == text:
                continue
            used = records_using(task, version)
            if used:
                raise PromptVersionConflict(
                    f"{path} already holds different text, and {used} {task} "
                    f"verdicts on disk are stamped {version!r}.\n"
                    f"Bump the version in {where}, or restore the wording."
                )
            print(f"  [{task}] {version} snapshot rewritten "
                  f"(0 verdicts reference it)")
        path.write_text(text)
    return out


def snapshot(version: str = PROMPT_VERSION) -> Path:
    return snapshot_texts(
        "utility", version, {k: r() for k, r in _SNAPSHOT.items()},
        "utility_prompt.py")
