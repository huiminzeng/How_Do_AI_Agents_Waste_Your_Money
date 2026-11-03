"""Verdicts -> counts per stream: how many verdicts, how many spans, how many
rejected, and the span kinds the judge named.

Token and dollar figures live in `exact_tokens`, counted with the model's own
tokenizer. Nothing here is a size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .. import paths
from ..prompts.efficiency_classes import (CLASSES, KIND_SEVERITY, SPAN_KINDS,
                                          STREAMS)
from ..evaluate.efficiency import find_span


@dataclass
class Spend:
    judge_id: str
    judge_run_id: str
    agent_model: str
    benchmark: str
    domain: str
    #: stream -> how many verdicts
    verdicts: dict[str, int] = field(default_factory=dict)
    n_traces: int = 0
    #: Rejected VERDICTS, not trajectories - a verdict whose call errored, or
    #: that could not be aligned to its turn, or whose citation failed. One
    #: trajectory can contribute several, and a trajectory can contribute one
    #: and still be usable. The viewer said "6 of 38 trajectories produced no
    #: usable verdict" off this field when the true trajectory count was 4.
    n_errors: int = 0
    #: Trajectories left with NO usable verdict at all. This is the sampling
    #: hole - those trajectories are in every denominator and in no numerator -
    #: and it is the number a reader needs, so it is counted here rather than
    #: inferred from `n_errors`.
    n_traces_unusable: int = 0
    spans_total: int = 0
    spans_unverified: int = 0
    n_misaligned: int = 0
    #: stream -> kind -> how many spans the judge named of that kind. Never
    #: summed into one "waste" number: the five kinds have different fixes, and
    #: `restated` (write less) and `detour` (do less) are not the same finding.
    #: The size of each kind is in `exact_tokens`, in tokens.
    span_by_kind: dict = field(default_factory=dict)
    #: Streams that marked nothing at all AND named no round. An unmarked
    #: stream is a claim that every token was load-bearing, so a stream that
    #: also declines to say who used it has asserted necessity and shown nothing.
    unmarked_uncited: int = 0
    #: Streams where the judge produced MAX_SPANS quotes, so its list may be
    #: short of what it would have marked. Reported, never silently applied.
    spans_capped: int = 0

    def add(self, stream: str, cls: str, n_spans: int, n_bad: int,
            by_kind: dict | None = None, unmarked_uncited: bool = False,
            capped: bool = False) -> None:
        for k, v in (by_kind or {}).items():
            self.span_by_kind.setdefault(stream, {})
            self.span_by_kind[stream][k] = self.span_by_kind[stream].get(k, 0) + v
        self.unmarked_uncited += unmarked_uncited
        self.spans_capped += capped
        self.verdicts[stream] = self.verdicts.get(stream, 0) + 1
        self.spans_total += n_spans
        self.spans_unverified += n_bad

    @property
    def fabrication_rate(self) -> float:
        """Share of quotes that were not found in the source.

        The only direct read on the judge's reliability, and the reason spans are
        quoted verbatim rather than described. High here means the class shares
        above should not be trusted either.
        """
        if not self.spans_total:
            return 0.0
        return self.spans_unverified / self.spans_total

    def to_json(self) -> dict:
        return {
            "judge_id": self.judge_id,
            "judge_run_id": self.judge_run_id,
            "agent_model": self.agent_model,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "n_traces": self.n_traces,
            "n_errors": self.n_errors,
            "n_traces_unusable": self.n_traces_unusable,
            "n_misaligned": self.n_misaligned,
            "streams": {
                s: {
                    "verdicts": self.verdicts.get(s, 0),
                    "waste_by_kind": self.span_by_kind.get(s, {}),
                }
                for s in STREAMS if self.verdicts.get(s)
            },
            "span_by_kind": self.span_by_kind,
            "unmarked_uncited": self.unmarked_uncited,
            "spans_capped": self.spans_capped,
            "spans_total": self.spans_total,
            "spans_unverified": self.spans_unverified,
            "fabrication_rate": self.fabrication_rate,
        }


def build(judge_id: str, run_id: str, agent_model: str, benchmark: str,
          domain: str) -> Spend:
    from ..evaluate.efficiency import iter_verdicts
    from ..preprocess.sample import load_turns

    sp = Spend(judge_id, run_id, agent_model, benchmark, domain)
    for group in iter_verdicts(judge_id, run_id, agent_model, benchmark, domain):
        sp.n_traces += 1
        # The token view needs the trajectory back: the billed counts live on the
        # turns, not on the verdicts. A trajectory whose shard has gone missing
        # contributes no tokens rather than failing the rollup.
        try:
            turns = load_turns({"agent_model": group["agent_model"],
                                "benchmark": group["benchmark"],
                                "domain": group["domain"],
                                "trace_id": group["trace_id"]})
        except (OSError, KeyError):
            turns = []
        turns_by_round = {t.round: t for t in turns} if turns else None
        # Decided BEFORE the token rollup, because a trajectory the judge never
        # answered for must not contribute a denominator. Three trajectories
        # returned an empty response - 58, 28 and 88 rows of swe_bench never
        # answered - and keeping them asserts they wasted nothing, which is not a
        # finding but a run failure. They are counted in
        # `n_traces_unusable` and excluded from the cell.
        usable_any = any(not v.error and v.aligned and v.citation_ok
                         and v.cls in CLASSES for v in group["verdicts"])
        if not usable_any:
            sp.n_traces_unusable += 1
        for v in group["verdicts"]:
            # An unaligned verdict cannot be attributed to a turn, so it is not
            # evidence about anything and must not enter the totals.
            if v.error or not v.aligned or not v.citation_ok or v.cls not in CLASSES:
                sp.n_errors += 1
                sp.n_misaligned += bool(not v.error and not v.aligned)
                continue
            # Per-kind from the merged ranges, not from summed quote lengths:
            # nested quotes would otherwise make the parts exceed the whole, and
            # the whole is what the figures divide by. Needs the source text, so
            # the turn is read back rather than trusted from the record.
            text = ""
            if turns_by_round is not None:
                t = turns_by_round.get(v.round)
                if t is not None:
                    text = {"reasoning": t.reasoning_content,
                            "visible": t.visible_text,
                            "action": t.action}.get(v.stream, "") or ""
            by_kind = {}
            for s_ in v.spans:
                if s_.kind:
                    by_kind[s_.kind] = by_kind.get(s_.kind, 0) + 1
            # Located-ness and the waste total are re-derived here for the same
            # reason `waste_tokens.build` re-derives them: the stored values
            # carry whichever locator the judging run shipped with. Reading
            # `s.verified` here would have kept reporting a 2.12% fabrication
            # rate after the locator brought it down to 0.18%.
            if text:
                unlocated = sum(1 for s_ in v.spans
                                if s_.quote and find_span(text, s_.quote) is None)
            else:
                unlocated = sum(1 for s_ in v.spans if not s_.verified)
            sp.add(v.stream, v.cls, len(v.spans), unlocated, by_kind,
                   unmarked_uncited=not v.spans and not v.depended_on_by,
                   capped=v.spans_capped)
    return sp


def save(sp: Spend) -> "paths.Path":
    path = paths.ensure_parent(
        paths.summary_file(paths.EFFICIENCY_TASK, sp.judge_id, sp.judge_run_id, sp.agent_model,
                           sp.benchmark, sp.domain))
    path.write_text(json.dumps(sp.to_json(), indent=2) + "\n")
    return path


def table(sp: Spend) -> str:
    """The summary a run prints when it finishes.

    Columns are the five waste kinds. The three classes are not printed: they
    are a coarsening of exactly this row, and printing both invites reading the
    label instead of the number.
    """
    lines = [
        f"  judge={sp.judge_id}  run={sp.judge_run_id}  "
        f"cell={sp.agent_model}/{sp.benchmark}/{sp.domain}",
        f"  {sp.n_traces} trajectories, {sp.n_errors} rejected "
        f"({sp.n_misaligned} misaligned), "
        f"{sp.spans_unverified}/{sp.spans_total} quotes not found in source "
        f"({sp.fabrication_rate * 100:.0f}%)",
    ]
    if sp.unmarked_uncited or sp.spans_capped:
        lines.append(
            f"  {sp.unmarked_uncited} streams marked nothing and named nobody"
            + (f", {sp.spans_capped} hit the span cap" if sp.spans_capped
               else ""))
    short = {"restated": "restated", "duplicate-fetch": "dup-call",
             "padding": "padding", "detour": "detour", "unread": "unread"}
    lines += [
        "",
        f"  {'stream':<11}{'verdicts':>10}"
        + "".join(f"{short[k]:>10}" for k in SPAN_KINDS),
        "  " + "-" * (21 + 10 * len(SPAN_KINDS)),
    ]
    for s in STREAMS:
        n = sp.verdicts.get(s, 0)
        if not n:
            continue
        kinds = sp.span_by_kind.get(s, {})
        lines.append(
            f"  {s:<11}{n:>10,}"
            + "".join(f"{kinds.get(k, 0):>10,}" for k in SPAN_KINDS))

    return "\n".join(lines)
