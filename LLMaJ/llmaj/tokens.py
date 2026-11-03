"""Token and cost composition per cell.

Five categories, chosen so that count-share and cost-share are directly comparable:

    input        prompt tokens billed at full rate (cache misses)
    cache_read   prompt tokens served from cache
    cache_write  prompt tokens written into the cache
    visible      output tokens that became text or tool calls
    reasoning    output tokens spent thinking

`visible` is derived as `output_tokens - reasoning_tokens`, which is exact: verified
against Gemini's own tokenizer on 12 calls, difference 0 every time.
`reasoning_tokens` is *inside* `output_tokens`, never additional to it - treating them as
separate would double-count the output side.

The point of splitting count from cost is that the two look nothing alike. `cache_read`
dominates the token count and is priced at a tenth of the input rate, so a pie of counts
and a pie of dollars put their mass in different places. That gap is the measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

from . import prices
from .preprocess.sources import Cell, cells_for, _read_calls  # noqa: F401
from .preprocess import inline_think
from . import paths

CATEGORIES = ("input", "cache_read", "cache_write", "visible", "reasoning")

#: Which price applies to which category. `visible` and `reasoning` are both output.
PRICE_KEY = {
    "input": "input",
    "cache_read": "cache_read",
    "cache_write": "cache_write",
    "visible": "output",
    "reasoning": "output",
}


@dataclass
class Composition:
    """Token counts and repriced dollars for one cell."""

    cell_key: str
    model: str
    benchmark: str
    domain: str
    counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in CATEGORIES})
    n_calls: int = 0
    #: What the proxy recorded at the time, summed. The check on our repricing.
    recorded_cost: float = 0.0

    def add_call(self, rec: dict) -> None:
        out = int(rec.get("output_tokens") or 0)
        reasoning = int(rec.get("reasoning_tokens") or 0)
        # A few providers report reasoning_tokens > output_tokens (seen once in
        # 5,538 kimi-k3 calls). Clamp rather than emit a negative slice.
        reasoning = min(reasoning, out)
        self.counts["input"] += int(rec.get("input_tokens") or 0)
        self.counts["cache_read"] += int(rec.get("cache_read_tokens") or 0)
        self.counts["cache_write"] += int(rec.get("cache_write_tokens") or 0)
        self.counts["reasoning"] += reasoning
        self.counts["visible"] += out - reasoning
        self.recorded_cost += float(
            rec.get("provider_cost_usd") or rec.get("cost_usd") or 0.0
        )
        self.n_calls += 1

    def dollars(self) -> dict[str, float]:
        r = prices.rates(self.model)
        if r is None:
            return {c: 0.0 for c in CATEGORIES}
        return {
            c: self.counts[c] / 1e6 * r[PRICE_KEY[c]] for c in CATEGORIES
        }

    @property
    def total_tokens(self) -> int:
        return sum(self.counts.values())

    @property
    def total_cost(self) -> float:
        return sum(self.dollars().values())

    def share_counts(self) -> dict[str, float]:
        t = self.total_tokens or 1
        return {c: self.counts[c] / t for c in CATEGORIES}

    def share_dollars(self) -> dict[str, float]:
        d = self.dollars()
        t = sum(d.values()) or 1.0
        return {c: d[c] / t for c in CATEGORIES}

    @property
    def reprice_error(self) -> float:
        """Relative gap between our repricing and the recorded cost.

        Large values mean `prices.py` has drifted from what the proxy charged, or
        the cell mixes models. Either way the pies are not trustworthy.
        """
        if not self.recorded_cost:
            return 0.0
        return (self.total_cost - self.recorded_cost) / self.recorded_cost

    def to_json(self) -> dict:
        return {
            "cell_key": self.cell_key,
            "model": self.model,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "n_calls": self.n_calls,
            "counts": self.counts,
            "dollars": self.dollars(),
            "share_counts": self.share_counts(),
            "share_dollars": self.share_dollars(),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "recorded_cost": self.recorded_cost,
            "reprice_error": self.reprice_error,
        }


def compose(agent_model: str | None = None) -> list[Composition]:
    """Aggregate every cell. Reads raw call records directly for the token fields."""
    import glob

    out: list[Composition] = []
    for cell in cells_for(agent_model):
        comp = Composition(cell.key, cell.agent_model, cell.benchmark, cell.domain)
        if cell.layout == "monitor":
            files: Iterator = (
                f
                for root in paths.MONITOR_ROOTS
                if root.exists()
                for f in sorted(root.glob(cell.pattern))
            )
        else:
            f = paths.RESULTS / cell.pattern
            files = iter([f] if f.exists() else [])
        for f in files:
            first = True
            for rec in _read_calls(f):
                if first:
                    first = False
                    if not cell.accepts(rec):
                        break
                # Inline-think models (inline_think.py) have no proxy
                # reasoning_tokens; the reconstructed figure goes in its place.
                comp.add_call({**rec, "reasoning_tokens": inline_think.reasoning_tokens(rec)})
        if comp.n_calls:
            out.append(comp)
    return out


def save(comps: list[Composition]) -> None:
    """One file per cell, under its own directory.

    A single combined file forced every reader to load all cells and filter, and
    hid the axes the layout is supposed to make explicit.
    """
    for c in comps:
        path = paths.ensure_parent(
            paths.agent_spend_dir(c.model, c.benchmark, c.domain) / "composition.json"
        )
        path.write_text(json.dumps(c.to_json(), indent=2) + "\n")


def load(agent_model: str, benchmark: str, domain: str) -> Composition | None:
    path = paths.agent_spend_dir(agent_model, benchmark, domain) / "composition.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    c = Composition(d["cell_key"], d["model"], d["benchmark"], d["domain"])
    c.counts = d["counts"]
    c.n_calls = d["n_calls"]
    c.recorded_cost = d["recorded_cost"]
    return c


