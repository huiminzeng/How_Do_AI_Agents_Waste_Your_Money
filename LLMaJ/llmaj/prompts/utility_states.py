"""The eight reasoning states.

Single source of truth. `prompt.py` renders these for the judge and `matrix.py`
indexes transitions by them; importing both from here keeps them from drifting
apart, which would silently corrupt every matrix.

Definitions and boundary rules are the normative text of TAXONOMY.md 2 and 3.
"""

from __future__ import annotations

FRAME = "FRAME"
PLAN = "PLAN"
RETRIEVE = "RETRIEVE"
INTERPRET = "INTERPRET"
DERIVE = "DERIVE"
MUTATE = "MUTATE"
VERIFY = "VERIFY"
RECOVER = "RECOVER"

#: Order is fixed: it is the row/column order of every transition matrix.
STATES: tuple[str, ...] = (
    FRAME,
    PLAN,
    RETRIEVE,
    INTERPRET,
    DERIVE,
    MUTATE,
    VERIFY,
    RECOVER,
)

#: Sentinels for the first and last turn *with reasoning text* in a trajectory.
START = "START"
END = "END"

#: Mass of edges dropped for sample size (TAXONOMY.md 5). A real row entry, not
#: a rounding remainder, so rows still sum to 1.
OTHER = "OTHER"

#: Matrix axes. START never receives, END never emits, but both get a row and a
#: column so the matrix stays square and indexable by a single lookup.
NODES: tuple[str, ...] = (START,) + STATES + (END, OTHER)

INDEX: dict[str, int] = {name: i for i, name in enumerate(NODES)}

DEFINITIONS: dict[str, str] = {
    FRAME: (
        "Establishing what the task is and what constrains it: parsing the "
        "request, restating it, identifying policy, permission or format "
        "requirements that bound the solution."
    ),
    PLAN: (
        "Planning proper: laying out a sequence of steps, breaking the task "
        "into sub-tasks, or weighing alternative approaches against each "
        "other. PLAN must span more than the immediate next action - picking "
        "the next command is not PLAN. FRAME asks what is being asked; PLAN "
        "asks in what order, or by which of several routes."
    ),
    RETRIEVE: (
        "Deciding to read from the environment, or issuing a read. Read-only "
        "intent."
    ),
    INTERPRET: (
        "Making sense of what a tool returned. Occurs after a result arrives "
        "and before the next decision."
    ),
    DERIVE: (
        "Reasoning to a conclusion from facts already in context - arithmetic, "
        "aggregation, inference. No tool involved."
    ),
    MUTATE: (
        "Producing the deliverable itself: composing the new code, the edit, the "
        "patch, or the reply text. Working out what the change should say is "
        "MUTATE even if it has not been executed yet - the content is the "
        "deliverable. Deciding to make a change, or making one, is MUTATE."
    ),
    VERIFY: (
        "Deciding to check, or checking, a result against an expectation: "
        "running tests, re-reading a value to confirm, sanity-checking a "
        "computation."
    ),
    RECOVER: (
        "Reacting to something that went wrong: diagnosing a failure, "
        "retracting, abandoning an approach for another."
    ),
}

#: Tie-breakers for the contested pairs. Applied uniformly rather than argued
#: case by case; see TAXONOMY.md 3.
#:
#: Three rules decide the contested pairs, each the fix for a failure mode
#: seen in piloting:
#:   - Ties break on how much of the reasoning each activity occupies, not on
#:     the action the turn ends in. The action is captured independently by
#:     `tool_called`; a judge that re-derives it labels the action rather than
#:     the reasoning, and INTERPRET and VERIFY become unreachable.
#:   - MUTATE is defined by the deliverable's content, not by executing it, with
#:     an explicit PLAN/MUTATE tie-break; otherwise a turn composing an edit
#:     falls to PLAN.
#:   - PLAN is bounded by level, not subject matter: it needs a sequence, a
#:     decomposition or competing approaches. Choosing the next action belongs
#:     to the activity it serves, as RETRIEVE's definition already claims
#:     "deciding to read".
#: Rewording the tie-breaks moved 39% of labels on one pilot cell. That is a
#: property of the taxonomy worth reporting, not a defect to tune away, and it
#: is why per-judge results are primary and the between-judge distance is the
#: denominator for every claim (TAXONOMY.md 5).
BOUNDARY_RULES: tuple[str, ...] = (
    "RETRIEVE vs MUTATE - decided by intent to change state, not by which tool "
    "exists. A read issued in order to prepare a write is still RETRIEVE.",
    "RETRIEVE vs INTERPRET - if the reasoning is dominated by what came back, "
    "it is INTERPRET; if by what to fetch next, RETRIEVE. When a turn does "
    "both, label by whichever occupies more of the reasoning text, not by what "
    "the turn ends up doing.",
    "DERIVE vs INTERPRET - INTERPRET reads a tool result at face value; DERIVE "
    "combines or transforms facts into something not present in any single "
    "result.",
    "VERIFY vs RECOVER - VERIFY is checking without having found a problem; "
    "once a problem is identified and the turn is about responding to it, it "
    "is RECOVER.",
    "FRAME vs PLAN - FRAME is about the problem's constraints, PLAN about the "
    "solver's next move. A turn spent mostly on what policy, permissions or "
    "the request itself require is FRAME even if it closes by naming an "
    "action.",
    "PLAN vs MUTATE - a turn that spells out the replacement code, the diff, "
    "or the wording of the reply is MUTATE, even when it is phrased as an "
    "intention (\"now let me change line 147 to ...\").",
    "Action selection is not PLAN - a turn that only chooses how, or with "
    "which command, to carry out the next step belongs to the activity that "
    "step serves: a command that edits is MUTATE, one that reads is RETRIEVE, "
    "one that tests is VERIFY. PLAN requires either more than one step or "
    "more than one candidate approach.",
    "Mixed turns - label the activity that occupies most of the reasoning. A "
    "turn that spends most of its text making sense of a result and one clause "
    "announcing the next call is INTERPRET, not RETRIEVE. When a turn "
    "contains nothing but the announcement, label it by what the announced "
    "action is for.",
)


