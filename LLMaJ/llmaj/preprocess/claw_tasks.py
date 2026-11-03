"""claw_eval task shapes, for the deliverable sentence.

The judge is told what the task was meant to PRODUCE. claw_eval needs more than
one sentence for that, and the reason is not that it has many categories - it is
that three groups of tasks have a deliverable the user's own request does not
state.

Measured over the 161 tasks in `tasks_general/`, which is all that was
run:

    the task's `prompt.text` IS the first user message, verbatim - 60 of 60
    sampled at similarity 1.00. So for most tasks the most precise statement of
    the deliverable is already in front of the judge, and a hand-written
    sentence can only be vaguer than it. What that sentence should do is point
    at the request and name the shapes a request can take, not replace it.

Three groups are the exception, and they are exceptions because something in the
deliverable is NOT in the user's words:

    terminal (5)   the work happens to files in a container; the artefact is a
                   side effect, not the reply
    coding (2)     the answer is a defect or an execution order in code the user
                   pasted, and the reasoning is part of the deliverable
    safety (5)     these are prompt-injection and over-ask tests. The correct
                   deliverable includes NOT doing what instructions embedded in
                   the content ask for, and refusing part of what the user asks.
                   Nothing in the request says so.

`category` alone does NOT determine the shape, which is why this maps only those
three and lets everything else fall through. Read as a warning against the
obvious refactor: `communication` holds four different shapes - triage
(classify), draft-and-save (text, not sent), contact lookup (an answer) and
send-the-email (an action) - and `productivity` holds three. Mapping all 24
categories to shapes would have handed a third of the tasks a sentence that is
wrong, and a wrong deliverable is worse than a general one: it points the judge
somewhere else instead of at the request.
"""

from __future__ import annotations

import glob
import os
import re

from .. import paths

#: category -> shape, for the three groups whose deliverable the request does
#: not state. Everything else is deliberately absent.
SHAPE_OF_CATEGORY = {
    "terminal": "execute",
    "coding": "review",
    "safety": "guarded",
}

_TASK_DIR = paths.PROJECT_ROOT / "claw-eval" / "tasks_general"
_CACHE: dict[str, str] | None = None


def _load() -> dict[str, str]:
    """task_id -> category, read once from the task definitions on disk."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out: dict[str, str] = {}
    for f in glob.glob(str(_TASK_DIR / "*" / "task.yaml")):
        try:
            text = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        m = re.search(r"^category:\s*(\S+)", text, re.M)
        if m:
            out[os.path.basename(os.path.dirname(f))] = m.group(1)
    _CACHE = out
    return out


def shape(task_id: str | None) -> str | None:
    """Which deliverable group this task belongs to, or None for the majority.

    None is a real answer, not a lookup failure: it means "the user's request
    states the deliverable" and the caller should use the general sentence. An
    unknown task_id also lands here, which is the safe direction - a task we
    cannot identify gets the sentence that tells the judge to read the request.
    """
    if not task_id:
        return None
    return SHAPE_OF_CATEGORY.get(_load().get(task_id, ""))


def categories() -> dict[str, str]:
    return dict(_load())
