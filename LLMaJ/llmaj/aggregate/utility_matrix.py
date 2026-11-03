"""Stage 4: labels -> transition matrix.

Implements TAXONOMY.md 5 verbatim. The rules matter more than the arithmetic:

  - a transition needs BOTH endpoints labelled AND adjacent rounds. A textless
    turn between two labelled turns breaks the chain; bridging it would
    fabricate an edge that was never taken.
  - START and END are the first and last labelled turn of a trajectory.
  - rows are normalised BEFORE any edge is dropped for sample size. Dropping
    first would inflate whatever survives.
  - the n < MIN_EDGE_N residual stays in the row as explicit OTHER mass, so
    rows still sum to 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .. import paths
from ..schema import Label
from ..prompts.utility_states import END, INDEX, NODES, OTHER, START

#: Below this, an edge is not estimable and is not drawn (TAXONOMY.md 5).
MIN_EDGE_N = 10
#: A row with more mass than this in OTHER is mostly tail; the drawn edges
#: misrepresent it and it gets flagged.
OTHER_FLAG = 0.20


@dataclass
class TransitionMatrix:
    judge_id: str
    #: Which labelling run produced it. Part of the identity, not metadata: the
    #: same cell labelled under revised rules is a different matrix.
    judge_run_id: str
    agent_model: str
    benchmark: str
    domain: str
    #: Every transition observed, keyed "SRC>DST", each entry
    #: [trace_id, source round, destination round]. START and END have no turn
    #: of their own, so their side is None.
    #:
    #: This is the record; `counts` is derived from it. Storing a tally beside
    #: the evidence would be two implementations of the same rule, free to
    #: disagree once "adjacent" is ever redefined - the failure that made three
    #: swe_bench round-trip figures wrong.
    edges: dict[str, list[list]] = field(default_factory=dict)
    n_traces: int = 0
    n_labels: int = 0
    n_errors: int = 0
    n_broken: int = 0  # adjacent labels separated by a textless turn

    def add(self, src: str, dst: str, trace_id: str,
            src_round: int | None, dst_round: int | None) -> None:
        self.edges.setdefault(f"{src}>{dst}", []).append(
            [trace_id, src_round, dst_round]
        )

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """Tally, recomputed from `edges` on every access.

        A property rather than a field so the two can never drift: the number
        in a figure and the list of turns behind it come from one object.
        """
        out: dict[str, dict[str, int]] = {}
        for key, cases in self.edges.items():
            src, dst = key.split(">", 1)
            out.setdefault(src, {})[dst] = len(cases)
        return out

    def cases(self, src: str, dst: str) -> list[list]:
        """The turns behind one edge. The reason `edges` exists."""
        return self.edges.get(f"{src}>{dst}", [])

    def row_total(self, src: str) -> int:
        return sum(self.counts.get(src, {}).values())

    def probabilities(self, min_edge_n: int | None = None) -> dict[str, dict[str, float]]:
        """Normalise, then fold small edges into OTHER.

        Order is load-bearing: normalising after the drop would redistribute the
        dropped mass across the survivors instead of accounting for it.

        `min_edge_n` overrides MIN_EDGE_N. Pass 1 to keep every observed edge -
        useful for a preliminary look, where the default threshold empties the
        graph, but any figure drawn that way must say so: those edges are not
        estimable and their probabilities will move a lot with more data.
        """
        floor = MIN_EDGE_N if min_edge_n is None else min_edge_n
        out: dict[str, dict[str, float]] = {}
        for src, row in self.counts.items():
            total = sum(row.values())
            if not total:
                continue
            probs, other = {}, 0.0
            for dst, n in row.items():
                p = n / total
                if n >= floor:
                    probs[dst] = p
                else:
                    other += p
            if other > 0:
                probs[OTHER] = other
            out[src] = probs
        return out

    def flagged_rows(self) -> list[str]:
        return [s for s, r in self.probabilities().items()
                if r.get(OTHER, 0.0) > OTHER_FLAG]

    def to_json(self) -> dict:
        """The record, and only the record.

        `probabilities` is deliberately NOT stored. It is derived from `counts`
        at a threshold, and which threshold depends on the question - the
        figures keep every observed edge, the default drops anything under
        MIN_EDGE_N. Storing one of them made a file that disagreed with its own
        figure: on kimi-k3/mcp_atlas the stored rows were all `OTHER: 1.00`
        while flow.png showed six edges, both correct and neither labelled.

        `counts` and `edges` are threshold-free, so they cannot go stale. Any
        reader that wants probabilities calls `probabilities(n)` and knows what
        it asked for.
        """
        return {
            "judge_id": self.judge_id,
            "judge_run_id": self.judge_run_id,
            "agent_model": self.agent_model,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "nodes": list(NODES),
            #: The default an unqualified `probabilities()` would apply. Not a
            #: property of this matrix, and not what any figure used.
            "default_min_edge_n": MIN_EDGE_N,
            "n_traces": self.n_traces,
            "n_labels": self.n_labels,
            "n_errors": self.n_errors,
            "n_broken": self.n_broken,
            "counts": self.counts,
            "edges": self.edges,
        }


def _usable(labels: list[Label]) -> list[Label]:
    """Ordered, error-free labels."""
    return sorted(
        (l for l in labels if not l.error and l.state in INDEX),
        key=lambda l: l.round,
    )


def add_trace(m: TransitionMatrix, labels: list[Label]) -> None:
    seq = _usable(labels)
    m.n_labels += len(seq)
    m.n_errors += sum(1 for l in labels if l.error)
    if not seq:
        return
    m.n_traces += 1
    tid = seq[0].trace_id
    m.add(START, seq[0].state, tid, None, seq[0].round)
    for a, b in zip(seq, seq[1:]):
        if b.round == a.round + 1:
            m.add(a.state, b.state, tid, a.round, b.round)
        else:
            # A textless turn sat between them. Not an edge.
            m.n_broken += 1
    m.add(seq[-1].state, END, tid, seq[-1].round, None)


def build(judge_id: str, run_id: str, agent_model: str, benchmark: str,
          domain: str, task: str = "") -> TransitionMatrix:
    """`task` selects which judging to read; the utility tree by default.

    The transition RULES are the same on both sides - both label turns from
    the same trajectory, and "adjacent" means the same thing - so the two
    differ only in which pass produced the labels. Duplicating this function
    per side would be two copies of TAXONOMY.md 5, free to drift.
    """
    if task and task == paths.ACTION_TASK:
        from ..evaluate.action import iter_labels
    else:
        from ..evaluate.utility import iter_labels

    m = TransitionMatrix(judge_id, run_id, agent_model, benchmark, domain)
    for group in iter_labels(judge_id, run_id, agent_model, benchmark, domain):
        add_trace(m, group["labels"])
    return m


def save(m: TransitionMatrix) -> None:
    path = paths.ensure_parent(
        paths.summary_file(paths.UTILITY_TASK, m.judge_id, m.judge_run_id,
                           m.agent_model, m.benchmark, m.domain)
    )
    path.write_text(json.dumps(m.to_json(), indent=2) + "\n")
