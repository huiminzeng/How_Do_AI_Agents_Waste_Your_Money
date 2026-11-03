"""Build the text sent to a judge for one whole trajectory.

The unit is a trajectory, not a turn. That is forced by the definitions:
"necessary", "later reverted" and "already in the context" are all claims about
OTHER turns, so a judge shown one turn can only guess. It is also 2.8x cheaper -
the system prompt goes out 30 times instead of 935, and observations stop being
repeated across turns.

Each turn also shows what the agent had just RECEIVED - the user's message or
the previous tool's result, without which the judge produced wrong verdicts:
the judge called a repeated `get_reservation_details` "not-necessary, no changes
have been made", when the message in between was the customer saying "I'm pretty
sure I did purchase the insurance - can you please double check on your end?".
Re-fetching was compliance, not waste. "Was this call necessary" cannot be
answered without knowing what prompted it.

Withheld on purpose:

    whether the trajectory succeeded   scoring a failed run would collapse to
                                       "none of it was necessary", measuring the
                                       outcome again instead of the spend
    the benchmark's name               `deliverable` states what was being
                                       produced, in plain words, so the judge
                                       reasons from the task rather than from a
                                       prior about the benchmark

Shown on purpose, unlike the functional prompt: every tool call, in full. The
functional pass hides the action so a label cannot be a restatement of it. Here
the action is the thing being judged.

EFF_PROMPT_VERSION is stamped on every verdict. Change the wording, bump it.
"""

from __future__ import annotations

from pathlib import Path

from .efficiency_classes import (BOUNDARY_RULES, DEPENDS_FIELD, KEPT_FIELD,
                                 MAX_SPANS, OPENING_CHARS, SPAN_KIND_DESC,
                                 SPAN_KINDS, STREAM_DESC, STREAMS)
from ..schema import Turn

#: The prompt's design, each element the fix for a wrong verdict seen in piloting:
#:   - every turn shows what the agent RECEIVED, marked FROM THE USER or FROM A
#:     TOOL: a repeated call is necessary after "please double check" and
#:     redundant after the same JSON twice;
#:   - `opening` quotes are short (25 characters asked, 20 compared): a model
#:     cannot stop mid-token, so longer copies diverge at the ragged end;
#:   - the judge marks removable stretches and does not choose a class; the
#:     class is computed from coverage, and what it leaves unmarked is its
#:     claim of necessity. Five span kinds (restated, duplicate-fetch, unread,
#:     padding, detour), cap 40 per stream;
#:   - the cell's standing instructions (system prompt) are shown, so
#:     `restated` is checked against the policy rather than guessed - cheap,
#:     because the document is identical across a cell;
#:   - a call repeated with a DIFFERENT result is flagged in the trajectory so
#:     the judge rules on it (a necessary verification or a fruitless retry);
#:     identical repeats are found mechanically.
#: Stamped on every verdict; change the wording, bump it.
EFF_PROMPT_VERSION = "v1"

#: Per-turn render caps.
#:
#: `visible` and `action` are at the storage cap, so nothing the agent produced
#: is hidden from the judge. At 1,000 and 2,000 the audit of those two streams
#: was incomplete in a way the totals did not show: text past
#: the cap entered the denominator and could never be marked, so waste was
#: understated by 0.2% to 9.3% of characters depending on the agent, worst on
#: visible. Measured across all four agents and three cells.
#:
#: `received` and `reasoning` stay clipped, for different reasons. Received text
#: is INPUT - it is shown so "already in the context" is answerable, and a tool
#: dump's tail rarely changes that answer. Reasoning is clipped because it is the
#: longest stream by far and the cap has not been shown to bind; if it turns out
#: to, the same measurement applies.
#:
#: Truncation is marked inline so the judge can see it is quoting from a
#: shortened view.
RENDER_CAPS = {"received": 1200, "reasoning": 8000, "visible": 8000,
               "action": 8000}

_SYSTEM = """You audit how an AI agent spent its output tokens.

You get one complete trajectory: every turn it took, in order. Each turn shows
what the agent had just received - the user's message, or the previous tool's
result - and then up to three streams of its own output:

{streams}

Received text is marked "from the user" or "from a tool". Neither is audited;
both are input, and you are auditing what the agent produced. They are shown
because they are what makes "was this needed" answerable, and in two directions:
a call that repeats an earlier one is milder when the user asked for it in
between, and reasoning that reads the user's own words back is not new content
however well phrased.

Your job for each stream is to mark what could be removed.

Marking
Quote every stretch of the stream that could be deleted without changing the
outcome. Not a few examples - every one. A stream that is load-bearing overall
still contains sentences that carry nothing, and those are what you are looking
for.

Each quote needs a kind:
{kinds}

Copy the text word for word: quotes are located in the source, and one that
cannot be found is recorded as a fabrication rather than a finding.

Up to {max_spans} quotes per stream. If you reach that many, quote the longest
stretches - a cap that binds is reported, so do not compress your answer to fit.

What you keep
Whatever you do not mark is your claim that those characters were load-bearing,
and it is read literally. Two fields state it:

  `{kept}`: one sentence naming what the unmarked remainder does. If you marked
  the whole stream, say plainly that nothing is left.

  `{depends}`: the round number of the work that uses that remainder - the edit
  it enables, the answer it supplies a fact to, the call whose result gets used.
  For reasoning this is often the same round, since a turn's thinking feeds that
  turn's own action. An array; empty when nothing uses any of it.

You are not asked to grade the stream. Necessity is what is left after you mark,
so there is no label to fall back on - a stream you mark nothing in is one you
are asserting was necessary character for character.

Rules
{rules}

Alignment
Every stream's entry must carry `opening`: the first {opening} characters of the
stream you are auditing, copied exactly. Its only job is to prove the entry is
filed against the right turn, so copy it rather than summarising it.

Output
A JSON object with a `turns` array, one entry per turn, in the order given. Each
turn lists only the streams it actually has:

{{"turns": [
  {{"round": <copied from the input>,
    "streams": [
      {{"stream": "reasoning",
        "opening": "<first {opening} characters, word for word>",
        "spans": [{{"quote": "<word for word>", "kind": "restated",
                   "why": "<one sentence>"}},
                  {{"quote": "<word for word>", "kind": "padding",
                   "why": "<one sentence>"}}],
        "{kept}": "<one sentence>",
        "{depends}": [<round numbers>]}},
      {{"stream": "action", "opening": "...", "spans": [],
        "{kept}": "...", "{depends}": []}}
    ]}}
]}}

Copy every round number from the input; they are checked.
"""

_USER_HEAD = """DELIVERABLE
{deliverable}
{policy}
TRAJECTORY
{n} turns follow."""

#: Shown in full, never truncated. A policy clipped in the middle makes "is this
#: sentence already in the context" unanswerable for everything after the cut,
#: which is worse than not showing it: the judge would grade half the document as
#: absent. 8,033 characters is about 2,000 tokens, sent once per trajectory.
_POLICY = """
STANDING INSTRUCTIONS
The agent was given the text below before the conversation started, and it is in
the agent's context on every turn. It is not part of what the agent produced and
you are not auditing it - it is here so that "already in the context" is
something you can check rather than assume.

--- begin standing instructions ---
{policy}
--- end standing instructions ---
"""

_TURN = """
--- round {round} ---{body}"""


def system_prompt() -> str:
    streams = "\n".join(f"  {s:<10} {STREAM_DESC[s]}" for s in STREAMS)
    rules = "\n".join(f"- {r}" for r in BOUNDARY_RULES)
    kinds = "\n".join(f"  {k:<16} {SPAN_KIND_DESC[k]}" for k in SPAN_KINDS)
    return _SYSTEM.format(streams=streams, rules=rules, kinds=kinds,
                          max_spans=MAX_SPANS, opening=OPENING_CHARS,
                          depends=DEPENDS_FIELD, kept=KEPT_FIELD)


def _clip(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return f"{text[:cap]}\n[... {len(text) - cap} more characters not shown]"


def render_turn(turn: Turn, recheck=None) -> str:
    """One turn's streams, or "" when the turn has none.

    A turn with no output at all is skipped rather than rendered empty: there is
    nothing to audit, and an empty block invites the judge to invent a verdict.
    """
    parts = []
    out = []
    if turn.prev_observation:
        # Lower case, and matching the wording of the system prompt, which
        # describes these markers. Shouting in a prompt earns nothing and makes
        # the text harder to read.
        who = {"user": "from the user", "tool": "from a tool",
               "system": "from the system"}.get(turn.prev_role, "received")
        out.append(f"\n{who}:\n"
                   + _clip(turn.prev_observation, RENDER_CAPS["received"]))
    for name, text in (("reasoning", turn.reasoning_content),
                       ("visible", turn.visible_text),
                       ("action", turn.action)):
        if text:
            parts.append(f"\n{name}:\n{_clip(text, RENDER_CAPS[name])}")
    # A turn with only a RECEIVED block has no output to classify.
    if not parts:
        return ""
    if recheck is not None:
        # Stated as a question, not as a verdict. The mechanical pass knows the
        # call was repeated and that the result changed; whether the re-check was
        # needed is the one part it cannot decide.
        parts.append(f"\nnote: {recheck.note()}")
    parts = out + parts
    return _TURN.format(round=turn.round, body="".join(parts))


def user_prompt(turns: list[Turn], deliverable: str, policy: str = "",
                rechecks: dict | None = None) -> str:
    blocks = [b for b in (render_turn(t, (rechecks or {}).get(t.round))
                          for t in turns) if b]
    head = _USER_HEAD.format(
        deliverable=deliverable, n=len(blocks),
        policy=(_POLICY.format(policy=policy.strip()) if policy.strip() else ""))
    return head + "".join(blocks)


def build(turns: list[Turn], deliverable: str, policy: str = "",
          rechecks: dict | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user",
         "content": user_prompt(turns, deliverable, policy, rechecks)},
    ]


def snapshot(version: str = EFF_PROMPT_VERSION) -> Path:
    from .utility_prompt import snapshot_texts

    return snapshot_texts(
        "efficiency", version,
        {"system.txt": system_prompt(), "user_template.txt": _USER_HEAD + _TURN},
        "efficiency_prompt.py",
    )
