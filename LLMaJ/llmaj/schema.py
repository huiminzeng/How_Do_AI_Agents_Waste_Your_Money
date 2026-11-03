"""Record types crossing module boundaries.

`sources.py` produces Trace/Turn; everything downstream consumes them and never
sees a proxy trace. Label is what a judge returns.

Kept as plain dataclasses with explicit (de)serialisation: the on-disk form is
part of the contract between stages that run days apart, so it should not drift
with a library's defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

#: Preceding observations are truncated before storage. Long enough for the
#: INTERPRET/DERIVE/VERIFY boundaries, short enough that swe_bench file dumps do
#: not dominate the shard. See TAXONOMY.md 3.1.
OBSERVATION_CHARS = 1500

#: The outgoing action is kept whole up to this. It is not an observation - a
#: truncated edit cannot be judged for necessity, which is the point of storing
#: it. Longest single action measured: 6,619 chars (a full-file rewrite).
ACTION_CHARS = 8000

#: Visible replies are short everywhere measured (median 84 chars, longest
#: 1,771), so this only guards against a pathological case.
VISIBLE_CHARS = 8000


@dataclass
class Turn:
    """One LLM call inside a trajectory."""

    trace_id: str
    round: int
    #: False when the model emitted no reasoning text. Kept rather than dropped
    #: because a textless turn *breaks* a transition chain (TAXONOMY.md 6) and
    #: the break can only be seen if the turn is present.
    has_text: bool
    reasoning_content: str = ""
    #: Last entry of request_messages: what the model saw before reasoning.
    #: Truncated to OBSERVATION_CHARS.
    prev_observation: str = ""
    #: Who wrote it - "user", "tool", "system", or "" when unknown. Kept because
    #: the two are read completely differently: a customer saying "please double
    #: check" makes a repeated call necessary, while a tool returning the same
    #: JSON twice makes it redundant; without the role both appear as one
    #: undifferentiated block.
    #:
    #: NEITHER is ever classified. Both are input to the agent, and the question
    #: both judges ask is what the agent PRODUCED was worth.
    prev_role: str = ""
    #: Recorded but never shown to a judge - it is the independent cross-check
    #: for the RETRIEVE/MUTATE boundary (TAXONOMY.md 3).
    #:
    #: On swe_bench this carries no information at all: 935 of 935 turns say
    #: "bash". The cross-check it was supposed to provide lives in `action`.
    tool_called: str | None = None
    output_tokens: int = 0
    reasoning_tokens: int = 0
    #: What the model sent out this turn: every tool call, name and arguments.
    #: Withheld from the functional prompt exactly as
    #: `tool_called` is - it is the independent signal, and a judge that sees it
    #: labels the action rather than the reasoning. The efficiency pass, which
    #: asks whether a call was necessary, cannot work without it.
    action: str = ""
    #: The reply text, separate from the action. On swe_bench this is empty on
    #: 36% of turns and 6% of output tokens; on sonnet/tau2 it is 86%. It is
    #: the only stream all four models have, so it is the only one on which they
    #: can be compared.
    visible_text: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Turn":
        return cls(**d)


@dataclass
class Trace:
    """One trajectory: the unit of analysis everywhere in this codebase.

    Transitions are counted inside a Trace, START/END are per-Trace, and the
    bootstrap resamples Traces. 
    """

    trace_id: str
    agent_model: str
    benchmark: str
    #: "all" where the benchmark has no domain, so path depth stays uniform.
    domain: str
    task_id: str | None = None
    turns: list[Turn] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_labellable(self) -> int:
        return sum(1 for t in self.turns if t.has_text)

    @property
    def n_transitions(self) -> int:
        """Adjacent rounds where BOTH turns carry text.

        A textless turn breaks the chain rather than being bridged; bridging
        would fabricate an edge that was never taken (TAXONOMY.md 6).
        """
        n = 0
        for a, b in zip(self.turns, self.turns[1:]):
            if a.has_text and b.has_text and b.round == a.round + 1:
                n += 1
        return n

    def index_row(self) -> dict[str, Any]:
        """The line this trajectory contributes to turns/index.jsonl.

        Carries the counts so the sample-size table and stratified sampling can
        be computed without opening a single shard.
        """
        return {
            "trace_id": self.trace_id,
            "agent_model": self.agent_model,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "task_id": self.task_id,
            "n_turns": self.n_turns,
            "n_labellable": self.n_labellable,
            "n_transitions": self.n_transitions,
        }

    def iter_json(self) -> Iterator[dict[str, Any]]:
        for t in self.turns:
            yield t.to_json()


@dataclass
class Label:
    """One judge's verdict on one turn."""

    trace_id: str
    round: int
    judge_id: str
    judge_run_id: str
    prompt_version: str
    state: str
    confidence: float | None = None
    rationale: str = ""
    #: Set when the judge failed or returned something unusable; `state` is then
    #: meaningless and the turn is excluded from matrices.
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Label":
        return cls(**d)


@dataclass
class Span:
    """A stretch of text the judge says could be deleted.

    `verified` is computed here, not reported by the judge: the quote either is
    or is not a substring of the stream it came from. A quote that is not is a
    fabrication, and the rate of those is the one direct read we have on how far
    the rest of a verdict can be trusted - the utility pass has no equivalent,
    having measured its own confidence scores to be useless.
    """

    quote: str
    why: str = ""
    #: One of SPAN_KINDS. Never summed into a single "waste" number: `restated`
    #: means write less and `detour` means do less, and the two have different
    #: fixes.
    kind: str = ""
    verified: bool = False
    #: "judge" or "mechanical". A `duplicate-fetch` is the one kind with ground
    #: truth - the same call returning the same bytes is a fact, not an opinion -
    #: so `preprocess/duplicates.py` finds them and they are injected here rather
    #: than left to the judge. Kept as a field so the two can be told apart: the
    #: judge's own marks stay in the record and are scored against the injected
    #: ones for recall, which is the only per-kind reliability number we can get.
    source: str = "judge"

    @property
    def n_chars(self) -> int:
        """Only verified spans have a length that means anything."""
        return len(self.quote) if self.verified else 0

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "n_chars": self.n_chars}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Span":
        return cls(quote=d.get("quote", ""), why=d.get("why", ""),
                   kind=d.get("kind", ""), verified=bool(d.get("verified")))


@dataclass
class EffVerdict:
    """One judge's verdict on ONE STREAM of one turn.

    A turn carries up to three streams and each gets its own record, so the
    rollup can weight by size per stream. `n_chars` is the length of the
    stream as the judge saw it, which is what makes the totals a statement about
    tokens rather than about how many turns there were.
    """

    trace_id: str
    round: int
    #: "reasoning" | "visible" | "action"
    stream: str
    judge_id: str
    judge_run_id: str
    prompt_version: str
    cls: str = ""
    why: str = ""
    n_chars: int = 0
    #: The stream's opening, copied back by the judge, and whether it matched.
    #: This is the only field that proves a verdict sits on the turn it was meant
    #: for: round numbers are copied, so a hallucinated round is caught but a
    #: shifted array is not - round 3 exists and its stream is non-empty, so
    #: every other check passes. A verdict with aligned=False is excluded from
    #: the rollup, because it cannot be attributed to anything.
    opening: str = ""
    aligned: bool = False
    #: Rounds the judge says depend on this stream. Empty is a finding, not a
    #: failure: it means nothing later relies on it, which is what makes the
    #: stream ok-to-have or not-necessary.
    depended_on_by: list[int] = field(default_factory=list)
    #: One sentence naming what the UNMARKED remainder does.
    kept: str = ""
    #: Older spelling of the same claim, still read so old verdicts load.
    what_breaks: str = ""
    #: Summed length of the VERIFIED spans, and that over `n_chars`. The judge
    #: never states a class: `cls` below is computed from this share, so the
    #: measurement is the marking and the label is a summary of it.
    waste_chars: int = 0
    waste_share: float = 0.0
    #: True when the judge produced MAX_SPANS quotes for this stream, so the list
    #: may be short of what it would have marked. Never silently applied.
    spans_capped: bool = False
    #: Whether every named round exists AND comes later. A citation that points
    #: nowhere is the whole reason to require one - without this check, naming a
    #: round would be as free as asserting necessity.
    citation_ok: bool = True
    spans: list[Span] = field(default_factory=list)
    #: Set when the judge failed or returned something unusable for this stream;
    #: `cls` is then meaningless and the record is excluded from the rollup.
    error: str | None = None

    @property
    def usable(self) -> bool:
        return not self.error and self.aligned and self.citation_ok

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["spans"] = [s.to_json() for s in self.spans]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "EffVerdict":
        d = dict(d)
        # Older records carry it; it was the SUM of quote lengths where
        # `waste_chars` is their union, and two quantities one letter apart
        # in the same record is a trap rather than a convenience.
        d.pop("wasted_chars", None)
        spans = [Span.from_json(x) for x in d.pop("spans", [])]
        return cls(**d, spans=spans)
