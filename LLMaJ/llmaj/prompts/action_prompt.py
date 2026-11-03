"""Build the text sent to a judge for one turn - the ACTION side.

The mirror of `utility_prompt`. That one shows the chain of thought and hides
what the agent did; this one shows what the agent did and hides the chain of
thought. Same eight states, same boundary rules, deliberately: a label from one
side is only comparable with a label from the other if the label space and the
tie-breaks are identical. Only the framing and the input slots differ.

Why the thought is withheld, and it is the whole reason this file exists: agent
reasoning routinely announces its own intent ("now let me run the tests"). A
judge shown both would read the intent off the thought and copy it, and the two
passes would agree by construction. The agreement between them is the thing we
want to measure, so it cannot be manufactured here.

The withholding is structural, as it is on the other side: this module receives
a Turn and never reads `reasoning_content`, `round`, or anything about what
happened after the turn.
"""

from __future__ import annotations

from ..schema import Turn
from .utility_prompt import snapshot_texts
from .utility_states import BOUNDARY_RULES, DEFINITIONS, STATES

#: Stamped on every verdict; change the wording, bump it.
PROMPT_VERSION = "v1"

_SYSTEM = """You classify what an AI agent DID on one turn.

The agent is working through a task. You are given one turn of what it sent
out - the tool calls it made and the reply text it wrote - plus the observation
it had just received before that turn. Decide which single state the turn
belongs to.

You are not shown the agent's private reasoning. Classify the turn by what the
action and the reply accomplish, reading the arguments of a call rather than
only its name: the same command can read, change or check depending on what it
is pointed at.

A turn may carry a reply, a tool call, or both. When it carries both, label the
one that does the work of the turn; a reply that only narrates the call is not
the work.

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

What it sent out - tool calls
{action}

What it sent out - reply text
{visible}"""


def system_prompt() -> str:
    states = "\n".join(f"{name:<10} {DEFINITIONS[name]}" for name in STATES)
    rules = "\n".join(f"- {r}" for r in BOUNDARY_RULES)
    return _SYSTEM.format(states=states, rules=rules)


def user_prompt(turn: Turn, task: str | None) -> str:
    """The per-turn half.

    An absent stream renders as "(none)" rather than being omitted, so the judge
    sees the same four headings on every turn and an empty reply is visibly
    empty rather than looking like a truncated prompt.
    """
    return _USER.format(
        task=task or "(not available)",
        observation=turn.prev_observation or "(nothing - this is the first turn)",
        action=turn.action or "(none - no tool call this turn)",
        visible=turn.visible_text or "(none - no reply text this turn)",
    )


def build(turn: Turn, task: str | None = None) -> list[dict[str, str]]:
    """Chat messages for one labelling call.

    Withheld: the chain of thought, the round number, the benchmark's identity,
    and whether the turn succeeded - the same list as the other side minus the
    action, which is what this side is for.
    """
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(turn, task)},
    ]


#: Together with a Turn these reproduce, byte for byte, every message any judge
#: was sent under this version.
_SNAPSHOT = {
    "system.txt": lambda: system_prompt(),
    "user_template.txt": lambda: _USER,
}


def snapshot(version: str = PROMPT_VERSION):
    """Write the snapshot under prompts/action/<version>/.

    Shares `snapshot_texts` with the utility side, which refuses to rewrite a
    version that already stamps verdicts. The guard counts per TASK, so
    action/v1 and utility/v1 are different namespaces and neither blocks the
    other.
    """
    return snapshot_texts("action", version,
                          {k: r() for k, r in _SNAPSHOT.items()},
                          "action_prompt.py")
